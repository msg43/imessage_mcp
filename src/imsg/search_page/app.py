"""The search page's web application (Starlette), with its access control.

Request path, outermost first:

1. `GuardMiddleware` (every request, including static files and the login
   page): a repeated `Host` header is 400; a `Host` not in
   `search_page.allowed_hosts` is 403; nothing else runs for either. It
   also stamps the security headers on every response: a strict Content
   Security Policy (this origin's script and style only, no inline code),
   `nosniff`, a `same-origin` referrer policy, framing denied, `no-store`
   caching. (`no-referrer` would make browsers send `Origin: null` on the
   page's own form posts, which the same-site check must refuse.)
2. Per route: every route except `/login` and `/static/*` needs a valid
   session (`Session` via `_session`); HTML routes redirect to the login
   page, everything else answers 401.
3. Every `POST` (login, logout, labels) checks the request is same-site
   and carries the right CSRF token.

Blocking work (Postgres, SQLite, model API calls, media conversion) runs
in Starlette's thread pool, so one slow request never stalls the others.
Access logging is off on purpose: a search URL carries the search text.
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, quote, urlsplit

import psycopg
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from imsg.eval.runner import (
    AT4_MIN_POOLED_JUDGMENTS,
    AT4_MIN_QUERIES,
    AT4_MIN_QUERIES_WITH_A_POSITIVE,
)
from imsg.mcp.transport import find_duplicate_header, validate_host
from imsg.retrieval.query import analyze_query
from imsg.search_page import html as views
from imsg.search_page import labels as label_store
from imsg.search_page.attachments import (
    AttachmentRecord,
    cache_file,
    lookup_attachment,
    needs_audio_preview,
    needs_image_preview,
    serve_plan,
)
from imsg.search_page.auth import (
    LoginGuard,
    PasswordFile,
    Session,
    SessionStore,
    cookie_header,
    cookie_is_secure,
    login_cookie_name,
    new_login_token,
    same_site_request,
    session_cookie_name,
    tokens_match,
)
from imsg.search_page.errors import ModelApiUnavailable, SearchInputError, SecretFileError
from imsg.search_page.highlight import QueryMatcher
from imsg.search_page.search import (
    CHANNEL_ATTACHMENT_TEXT,
    SORT_ORDERS,
    Hit,
    QueryVectors,
    ResultCache,
    SearchRequest,
    SearchResult,
    SearchSettings,
    SemanticStatus,
    ThreadHits,
    apply_rerank,
    rerank_candidates,
    run_fulltext,
    run_semantic,
    segment_texts,
)
from imsg.search_page.threads import (
    MessageView,
    attachment_chunk_text,
    chat_by_thread_key,
    chat_views,
    decorate,
    is_opaque_key,
    messages_by_id,
    segment_messages,
    thread_page,
    thread_window,
)

if TYPE_CHECKING:
    import apsw

    from imsg.search_page.config import SearchPageConfig
    from imsg.search_page.media import MediaConverter
    from imsg.search_page.model_api_client import ModelApiClient

logger = logging.getLogger("imsg.search_page")

HTML_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "media-src 'self'; frame-src 'self'; object-src 'none'; base-uri 'none'; "
    "form-action 'self'; frame-ancestors 'none'"
)
MAX_FORM_BYTES = 16 * 1024
THREAD_WINDOW_EACH_SIDE = 40
THREAD_PAGE_SIZE = 60
PEOPLE_TTL_SECONDS = 900.0
STATIC_FILES = {"app.js": "text/javascript; charset=utf-8", "app.css": "text/css; charset=utf-8"}
SEGMENT_KEY_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
RERANK_DOC_CHARS = 2000


# --------------------------------------------------------------------------
# resource pools
# --------------------------------------------------------------------------


class ConnectionPool:
    """A few Postgres connections, each used by one request at a time.
    Connections are made lazily, verified by `connect`, and replaced when
    broken."""

    def __init__(self, connect: Callable[[], psycopg.Connection], size: int) -> None:
        self._connect = connect
        self._slots: queue.LifoQueue[psycopg.Connection | None] = queue.LifoQueue()
        for _ in range(size):
            self._slots.put(None)

    @contextlib.contextmanager
    def connection(self, timeout: float = 30.0) -> Iterator[psycopg.Connection]:
        conn = self._slots.get(timeout=timeout)
        try:
            if conn is None or conn.closed or conn.broken:
                conn = self._connect()
            yield conn
        except psycopg.OperationalError:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()
            conn = None
            raise
        finally:
            if (
                conn is not None
                and not conn.closed
                and conn.info.transaction_status != psycopg.pq.TransactionStatus.IDLE
            ):
                with contextlib.suppress(Exception):
                    conn.rollback()
            self._slots.put(conn)

    def close(self) -> None:
        while True:
            try:
                conn = self._slots.get_nowait()
            except queue.Empty:
                return
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()


class FtsReaders:
    """Read-only SQLite connections to the full-text sidecar. A sidecar
    rebuilt by `imsg fts rebuild` arrives by rename, so each checkout
    compares the file's identity with the one it opened and reopens when
    they differ."""

    def __init__(self, open_reader: Callable[[], apsw.Connection], path: Path, size: int) -> None:
        self._open = open_reader
        self._path = path
        self._slots: queue.LifoQueue[tuple[apsw.Connection, tuple[int, int]] | None] = (
            queue.LifoQueue()
        )
        for _ in range(size):
            self._slots.put(None)

    def _identity(self) -> tuple[int, int]:
        info = self._path.stat()
        return (info.st_dev, info.st_ino)

    @contextlib.contextmanager
    def reader(self, timeout: float = 30.0) -> Iterator[apsw.Connection]:
        slot = self._slots.get(timeout=timeout)
        try:
            identity = self._identity()
            if slot is None or slot[1] != identity:
                if slot is not None:
                    with contextlib.suppress(Exception):
                        slot[0].close()
                slot = (self._open(), identity)
            yield slot[0]
        finally:
            self._slots.put(slot)


class PeopleDirectory:
    """Everyone, with how many messages they sent, cached for autocomplete."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded_at = 0.0
        self._rows: list[tuple[str, str, int]] = []

    def _load(self, pg: psycopg.Connection) -> None:
        with pg.cursor() as cur:
            cur.execute(
                """
                SELECT p.display_name, p.short_name, coalesce(c.n, 0)
                FROM person p
                LEFT JOIN (
                    SELECT sender_person_id, count(*) AS n FROM message
                    WHERE sender_person_id IS NOT NULL GROUP BY sender_person_id
                ) c ON c.sender_person_id = p.person_id
                WHERE NOT p.is_owner
                """
            )
            rows = [(str(d), str(s), int(n)) for d, s, n in cur.fetchall()]
        rows.sort(key=lambda r: (-r[2], r[0].lower()))
        self._rows = rows
        self._loaded_at = time.monotonic()

    def search(self, pg: psycopg.Connection, text: str, limit: int = 12) -> list[tuple[str, str, int]]:
        with self._lock:
            if not self._rows or time.monotonic() - self._loaded_at > PEOPLE_TTL_SECONDS:
                self._load(pg)
            rows = self._rows
        needle = text.strip().lower()
        if not needle:
            return rows[:limit]
        prefix = [r for r in rows if r[0].lower().startswith(needle) or r[1].lower().startswith(needle)]
        inner = [
            r
            for r in rows
            if (needle in r[0].lower() or needle in r[1].lower()) and r not in prefix[:limit]
        ]
        return (prefix + inner)[:limit]


# --------------------------------------------------------------------------
# dependencies
# --------------------------------------------------------------------------


@dataclass(slots=True)
class AppDeps:
    page: SearchPageConfig
    settings: SearchSettings
    data_root: Path
    pool: ConnectionPool
    fts: FtsReaders
    passwords: PasswordFile
    sessions: SessionStore
    login_guard: LoginGuard
    media: MediaConverter
    model_api: ModelApiClient | None
    cache: ResultCache = field(default_factory=ResultCache)
    people: PeopleDirectory = field(default_factory=PeopleDirectory)

    @property
    def timezone(self) -> str:
        return self.settings.timezone


# --------------------------------------------------------------------------
# the outer guard: Host allowlist and security headers
# --------------------------------------------------------------------------


class GuardMiddleware:
    def __init__(self, app: ASGIApp, *, allowed_hosts: list[str]) -> None:
        self._app = app
        self._allowed_hosts = tuple(allowed_hosts)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        raw_headers = scope.get("headers", [])
        if find_duplicate_header(raw_headers, frozenset({"host"})) is not None:
            await self._plain(send, 400, "duplicate Host header")
            return
        host = Headers(raw=list(raw_headers)).get("host")
        if not validate_host(host, self._allowed_hosts):
            await self._plain(send, 403, "unknown host")
            return
        path = str(scope.get("path", ""))

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {name.lower() for name, _ in headers}

                def add(name: bytes, value: str) -> None:
                    if name not in present:
                        headers.append((name, value.encode("latin-1")))

                add(b"content-security-policy", HTML_CSP)
                add(b"x-content-type-options", "nosniff")
                add(b"referrer-policy", "same-origin")
                add(b"x-frame-options", "SAMEORIGIN" if path.startswith("/att/") else "DENY")
                add(b"cache-control", "no-store")
                add(b"permissions-policy", "camera=(), microphone=(), geolocation=()")
                add(b"cross-origin-opener-policy", "same-origin")
                message = {**message, "headers": headers}
            await send(message)

        await self._app(scope, receive, send_with_headers)

    async def _plain(self, send: Send, status: int, text: str) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"text/plain; charset=utf-8"),
                    (b"cache-control", b"no-store"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": text.encode("utf-8")})


# --------------------------------------------------------------------------
# request helpers
# --------------------------------------------------------------------------


def _deps(request: Request) -> AppDeps:
    deps = request.app.state.deps
    assert isinstance(deps, AppDeps)
    return deps


def _secure(request: Request) -> bool:
    deps = _deps(request)
    return cookie_is_secure(
        deps.page.cookie_secure,
        scheme=request.url.scheme,
        host=request.headers.get("host", ""),
        https_hosts=deps.page.https_hosts,
    )


def _raw_session_token(request: Request) -> str | None:
    return request.cookies.get(session_cookie_name(True)) or request.cookies.get(
        session_cookie_name(False)
    )


def _session(request: Request) -> Session | None:
    deps = _deps(request)
    try:
        fingerprint = deps.passwords.fingerprint()
    except SecretFileError:
        return None
    return deps.sessions.get(_raw_session_token(request), fingerprint)


def _client(request: Request) -> str:
    deps = _deps(request)
    client = request.client.host if request.client else "unknown"
    forwarded = request.headers.get("x-forwarded-for")
    if (
        forwarded
        and client in ("127.0.0.1", "::1")
        and request.headers.get("host", "").lower() in deps.page.https_hosts
    ):
        return forwarded.split(",")[-1].strip() or client
    return client


def _safe_next(value: str | None) -> str:
    if not value or not value.startswith("/") or value.startswith("//") or "\\" in value:
        return "/"
    parts = urlsplit(value)
    if parts.scheme or parts.netloc:
        return "/"
    return value


def _login_redirect(request: Request) -> Response:
    target = request.url.path + (f"?{request.url.query}" if request.url.query else "")
    return RedirectResponse(f"/login?next={quote(target, safe='')}", status_code=303)


def _unauthorized() -> Response:
    return JSONResponse({"error": "sign in required"}, status_code=401)


def _csrf_ok(request: Request, session: Session, provided: str | None) -> bool:
    return same_site_request(
        host=request.headers.get("host", ""),
        origin=request.headers.get("origin"),
        sec_fetch_site=request.headers.get("sec-fetch-site"),
    ) and tokens_match(session.csrf_token, provided)


async def _limited_body(request: Request, limit: int) -> bytes | None:
    declared = request.headers.get("content-length")
    if declared is not None and (not declared.isdigit() or int(declared) > limit):
        return None
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _page_ctx(request: Request, session: Session, title: str = "Messages") -> views.PageContext:
    deps = _deps(request)
    return views.PageContext(
        csrf_token=session.csrf_token,
        timezone=deps.timezone,
        title=title,
        semantic_available=deps.model_api is not None and deps.settings.semantic_enabled,
    )


# --------------------------------------------------------------------------
# search form parsing and result presentation
# --------------------------------------------------------------------------


def _form_state(request: Request) -> views.FormState:
    params = request.query_params
    sort = params.get("sort", "relevance")
    att = params.get("att", "any")
    return views.FormState(
        query=params.get("q", "")[:2000],
        people=params.get("people", "")[:500],
        date_from=params.get("from", "")[:10],
        date_to=params.get("to", "")[:10],
        attachments=att if att in ("any", "with", "without") else "any",
        sort=sort if sort in SORT_ORDERS else "relevance",
    )


def _search_request(form: views.FormState) -> SearchRequest:
    people = tuple(p.strip() for p in form.people.split(",") if p.strip())
    before: str | None = None
    if form.date_to:
        try:
            before = (date.fromisoformat(form.date_to) + timedelta(days=1)).isoformat()
        except ValueError as exc:
            raise SearchInputError("The 'to' date is not a valid date.") from exc
    if form.date_from:
        try:
            date.fromisoformat(form.date_from)
        except ValueError as exc:
            raise SearchInputError("The 'from' date is not a valid date.") from exc
    return SearchRequest(
        query=form.query,
        people=people,
        after=form.date_from or None,
        before=before,
        attachments=form.attachments,  # type: ignore[arg-type]
    )


def _result_for(deps: AppDeps, pg: psycopg.Connection, request: SearchRequest) -> SearchResult:
    validated = request.validated()
    key = validated.cache_key()
    cached = deps.cache.get(key)
    if cached is not None:
        return cached
    with deps.fts.reader() as fts:
        result = run_fulltext(pg, fts, validated, deps.settings)
    if result.semantic.state == "pending" and deps.model_api is None:
        result.semantic = SemanticStatus("disabled", note="the model API is off in config")
    deps.cache.put(key, result)
    return result


def _in_range(messages: list[MessageView], result: SearchResult) -> list[MessageView]:
    """Only the messages inside the search's date range, when it has one:
    a segment that overlaps the range may begin or end outside it."""
    after, before = result.filters.after, result.filters.before
    if after is None and before is None:
        return messages
    return [
        m
        for m in messages
        if (after is None or m.sent_at >= after) and (before is None or m.sent_at < before)
    ]


def _matching_span(
    messages: list[MessageView], hit: Hit, matcher: QueryMatcher
) -> tuple[datetime, datetime] | None:
    """When the shown messages that match were sent (all shown messages
    when none matches the words, as for a hit found by meaning)."""
    attachment_messages = {m for m, _a in hit.matched_attachments}
    matching = [
        m for m in messages if matcher.matched_terms(m.text) or m.message_id in attachment_messages
    ]
    chosen = matching or messages
    if not chosen:
        return None
    return chosen[0].sent_at, chosen[-1].sent_at


def _pick_messages(messages: list[MessageView], hit: Hit, matcher: QueryMatcher, limit: int = 3) -> list[MessageView]:
    if not messages:
        return []
    attachment_messages = {m for m, _a in hit.matched_attachments}
    scored: list[tuple[int, int, MessageView]] = []
    for index, message in enumerate(messages):
        terms = matcher.matched_terms(message.text)
        if terms or message.message_id in attachment_messages:
            scored.append((-(terms + (1 if message.message_id in attachment_messages else 0)), index, message))
    if not scored:
        return messages[:2]
    scored.sort()
    chosen = sorted(scored[:limit], key=lambda t: t[1])
    return [m for _s, _i, m in chosen]


def _segment_keys(pg: psycopg.Connection, segment_ids: list[int]) -> dict[int, str]:
    if not segment_ids:
        return {}
    with pg.cursor() as cur:
        cur.execute(
            "SELECT segment_id, stable_key FROM segment WHERE segment_id = ANY(%(ids)s::bigint[])",
            {"ids": segment_ids},
        )
        return {int(a): str(b) for a, b in cur.fetchall()}


def build_thread_views(
    deps: AppDeps,
    pg: psycopg.Connection,
    result: SearchResult,
    threads: list[ThreadHits],
    *,
    matcher: QueryMatcher,
    query_id: str | None,
    hit_offset: int | None = None,
) -> list[views.ThreadResultView]:
    """`hit_offset` set: the owner asked for all of each conversation's
    hits, and gets the 50 from that offset."""
    per_thread = deps.page.hits_per_thread
    chats = chat_views(pg, [t.chat_id for t in threads])

    def window(t: ThreadHits) -> list[Hit]:
        if hit_offset is None:
            return t.hits[:per_thread]
        return t.hits[hit_offset : hit_offset + views.HITS_PAGE]

    shown: list[tuple[ThreadHits, list[Hit]]] = [
        (t, window(t)) for t in threads if t.chat_id in chats
    ]
    hits = [h for _t, hs in shown for h in hs]
    segment_ids = [h.segment_id for h in hits if h.segment_id is not None]
    message_ids = [h.message_id for h in hits if h.message_id is not None]
    by_segment = {
        segment_id: _in_range(messages, result)
        for segment_id, messages in segment_messages(
            pg, segment_ids, index_unsent=deps.settings.index_unsent
        ).items()
    }
    direct = messages_by_id(pg, message_ids)
    keys = _segment_keys(pg, segment_ids)
    picked: dict[str, list[MessageView]] = {}
    to_decorate: dict[int, MessageView] = {}
    for hit in hits:
        if hit.segment_id is not None:
            chosen = _pick_messages(by_segment.get(hit.segment_id, []), hit, matcher)
        else:
            chosen = [direct[hit.message_id]] if hit.message_id in direct else []
        picked[hit.key] = chosen
        for message in chosen:
            to_decorate[message.message_id] = message
    decorate(pg, list(to_decorate.values()))
    snippet_ids = {
        att_id
        for h in hits
        if CHANNEL_ATTACHMENT_TEXT in h.ranks
        for _m, att_id in h.matched_attachments
    }
    chunk_text = attachment_chunk_text(pg, snippet_ids)
    seg_labels, msg_labels = label_store.labels_for_hits(
        pg, query_id, segment_ids=segment_ids, message_ids=message_ids
    )
    out: list[views.ThreadResultView] = []
    for thread, thread_hits in shown:
        hit_views: list[views.HitView] = []
        for hit in thread_hits:
            messages = picked.get(hit.key, [])
            snippets: list[tuple[str, str]] = []
            if CHANNEL_ATTACHMENT_TEXT in hit.ranks:
                names = {a.attachment_id: (a.filename or "attachment") for m in messages for a in m.attachments}
                for _m, att_id in sorted(hit.matched_attachments):
                    text = chunk_text.get(att_id)
                    if text:
                        snippets.append((names.get(att_id, "attachment"), matcher.snippet(text)))
            if hit.segment_id is not None:
                label_key = keys.get(hit.segment_id, "")
                kind = "segment"
                label = seg_labels.get(hit.segment_id)
            else:
                label_key = messages[0].message_key if messages else ""
                kind = "message"
                label = msg_labels.get(hit.message_id or -1)
            hit_views.append(
                views.HitView(
                    hit=hit,
                    kind=kind,
                    label_key=label_key,
                    messages=messages,
                    anchor_key=messages[0].message_key if messages else None,
                    snippets=snippets[:3],
                    label=label,
                    span=_matching_span(messages, hit, matcher),
                )
            )
        next_offset: int | None = None
        if hit_offset is not None and hit_offset + len(thread_hits) < thread.count:
            next_offset = hit_offset + len(thread_hits)
        out.append(
            views.ThreadResultView(
                chat=chats[thread.chat_id],
                count=thread.count,
                hits=hit_views,
                latest_at=thread.latest_at,
                shown_all=hit_offset is not None,
                next_offset=next_offset,
                shown_from=hit_offset or 0,
            )
        )
    return out


def _status(result: SearchResult, sort: str) -> views.StatusView:
    threads = result.threads(sort)  # type: ignore[arg-type]
    return views.StatusView(
        total_hits=result.total_hits,
        total_threads=len(threads),
        counts=dict(result.counts),
        capped=frozenset(result.capped),
        semantic_state=result.semantic.state,
        semantic_note=result.semantic.note,
        timings_ms=dict(result.timings_ms),
        hidden_non_content=result.hidden_non_content,
    )


def _page_slice(deps: AppDeps, result: SearchResult, sort: str, page: int) -> tuple[list[ThreadHits], bool]:
    size = deps.page.page_threads
    threads = result.threads(sort)  # type: ignore[arg-type]
    start = (page - 1) * size
    return threads[start : start + size], start + size < len(threads)


def _page_number(request: Request) -> int:
    raw = request.query_params.get("page", "1")
    return max(1, min(int(raw), 100000)) if raw.isdigit() else 1


def _render_results(
    deps: AppDeps,
    pg: psycopg.Connection,
    result: SearchResult,
    form: views.FormState,
    page: int,
    query_id: str | None,
) -> str:
    matcher = QueryMatcher.for_query(result.analyzed)
    slice_, more = _page_slice(deps, result, form.sort, page)
    thread_views = build_thread_views(deps, pg, result, slice_, matcher=matcher, query_id=query_id)
    extra = {"sem": "1"} if result.semantic.state == "done" else {}
    next_url = form.url("/search/page", page=page + 1, **extra) if more else None
    full_url = form.url("/search", page=page + 1) if more else None
    return views.results_fragment(
        thread_views,
        tz=deps.timezone,
        matcher=matcher,
        form=form,
        next_page_url=next_url,
        next_full_url=full_url,
    )


def _ensure_semantic(deps: AppDeps, pg: psycopg.Connection, result: SearchResult, sort: str) -> None:
    """Run the semantic channels (and the optional rerank) for `result`
    once; later callers see the stored outcome."""
    with result.lock:
        if result.semantic.state == "pending":
            if deps.model_api is None or not deps.settings.semantic_enabled:
                result.semantic = SemanticStatus(
                    "disabled" if not deps.settings.semantic_enabled else "unavailable",
                    note=None if not deps.settings.semantic_enabled else "the model API is not configured",
                )
            else:
                try:
                    embedded = deps.model_api.embed(
                        result.analyzed.phrase, multimodal=deps.settings.multimodal_enabled
                    )
                except ModelApiUnavailable as exc:
                    result.semantic = SemanticStatus("unavailable", note=exc.reason)
                else:
                    result.timings_ms["embed"] = sum(embedded.timings_ms.values())
                    run_semantic(
                        pg,
                        result,
                        QueryVectors(text=embedded.text_vector, multimodal=embedded.multimodal_vector),
                        deps.settings,
                    )
        if sort == "rerank" and not result.reranked and deps.model_api is not None:
            candidates = rerank_candidates(result, deps.page.rerank_top)
            texts = segment_texts(
                pg, [h.segment_id for h in candidates if h.segment_id is not None], max_chars=RERANK_DOC_CHARS
            )
            candidates = [h for h in candidates if h.segment_id in texts]
            if candidates:
                try:
                    scores = deps.model_api.rerank(
                        result.analyzed.phrase, [texts[h.segment_id or 0] for h in candidates]
                    )
                except ModelApiUnavailable as exc:
                    note = f"rerank unavailable: {exc.reason}"
                    result.semantic.note = f"{result.semantic.note}; {note}" if result.semantic.note else note
                else:
                    apply_rerank(result, candidates, scores)


# --------------------------------------------------------------------------
# routes: pages
# --------------------------------------------------------------------------


def home(request: Request) -> Response:
    session = _session(request)
    if session is None:
        return _login_redirect(request)
    ctx = _page_ctx(request, session)
    return HTMLResponse(
        views.search_page(
            ctx,
            form=views.FormState(),
            status=None,
            results="",
            error=None,
            semantic_url=None,
            search_key=None,
        )
    )


def search_view(request: Request) -> Response:
    session = _session(request)
    if session is None:
        return _login_redirect(request)
    deps = _deps(request)
    ctx = _page_ctx(request, session, title="Search · Messages")
    form = _form_state(request)
    if not form.query.strip():
        return RedirectResponse("/", status_code=303)
    started = time.perf_counter()
    try:
        search_request = _search_request(form)
        with deps.pool.connection() as pg:
            result = _result_for(deps, pg, search_request)
            query_id = label_store.find_query_id(pg, result.request.query)
            with result.lock:
                results_html = _render_results(
                    deps, pg, result, form, _page_number(request), query_id
                )
                status = _status(result, form.sort)
    except SearchInputError as exc:
        return HTMLResponse(
            views.search_page(
                ctx, form=form, status=None, results="", error=str(exc), semantic_url=None, search_key=None
            ),
            status_code=400,
        )
    semantic_url = None
    if result.semantic.state == "pending" or (form.sort == "rerank" and not result.reranked):
        semantic_url = form.url("/api/semantic")
    page_ms = (time.perf_counter() - started) * 1000
    response = HTMLResponse(
        views.search_page(
            ctx,
            form=form,
            status=status,
            results=results_html,
            error=None,
            semantic_url=semantic_url,
            search_key=result.request.cache_key(),
        )
    )
    response.headers["server-timing"] = (
        f"fulltext;dur={result.timings_ms.get('fulltext_total', 0):.1f}, page;dur={page_ms:.1f}"
    )
    return response


def search_page_fragment(request: Request) -> Response:
    session = _session(request)
    if session is None:
        return _unauthorized()
    deps = _deps(request)
    form = _form_state(request)
    try:
        search_request = _search_request(form)
        with deps.pool.connection() as pg:
            result = _result_for(deps, pg, search_request)
            if request.query_params.get("sem") == "1" or form.sort == "rerank":
                _ensure_semantic(deps, pg, result, form.sort)
            query_id = label_store.find_query_id(pg, result.request.query)
            with result.lock:
                body = _render_results(deps, pg, result, form, _page_number(request), query_id)
    except SearchInputError as exc:
        return PlainTextResponse(str(exc), status_code=400)
    return HTMLResponse(body)


def search_thread_hits(request: Request) -> Response:
    session = _session(request)
    if session is None:
        return _unauthorized()
    deps = _deps(request)
    form = _form_state(request)
    thread_key = request.query_params.get("thread", "")
    if not is_opaque_key(thread_key):
        return PlainTextResponse("not found", status_code=404)
    raw_offset = request.query_params.get("offset", "0")
    if not raw_offset.isdigit() or int(raw_offset) > 10_000_000:
        return PlainTextResponse("bad offset", status_code=400)
    offset = int(raw_offset)
    try:
        search_request = _search_request(form)
        with deps.pool.connection() as pg:
            result = _result_for(deps, pg, search_request)
            chat = chat_by_thread_key(pg, thread_key)
            matcher = QueryMatcher.for_query(result.analyzed)
            query_id = label_store.find_query_id(pg, result.request.query)
            with result.lock:
                group = result.thread(chat.chat_id, form.sort) if chat else None  # type: ignore[arg-type]
                if chat is None or group is None:
                    return PlainTextResponse("not found", status_code=404)
                [view] = build_thread_views(
                    deps, pg, result, [group], matcher=matcher, query_id=query_id, hit_offset=offset
                )
    except SearchInputError as exc:
        return PlainTextResponse(str(exc), status_code=400)
    render = views.thread_result_html if offset == 0 else views.thread_hits_html
    return HTMLResponse(render(view, tz=deps.timezone, matcher=matcher, form=form, query=form.query))


def semantic_api(request: Request) -> Response:
    session = _session(request)
    if session is None:
        return _unauthorized()
    deps = _deps(request)
    form = _form_state(request)
    started = time.perf_counter()
    try:
        search_request = _search_request(form)
        with deps.pool.connection() as pg:
            result = _result_for(deps, pg, search_request)
            _ensure_semantic(deps, pg, result, form.sort)
            with result.lock:
                status = _status(result, form.sort)
    except SearchInputError as exc:
        return JSONResponse({"state": "error", "note": str(exc)}, status_code=400)
    return JSONResponse(
        {
            "state": result.semantic.state,
            "note": result.semantic.note,
            "added_hits": result.semantic.added_hits,
            "added_threads": result.semantic.added_threads,
            "reranked": result.reranked,
            "total_hits": result.total_hits,
            "status_html": views.status_html(status),
            "page1_url": form.url("/search/page", page=1, sem="1"),
            "timings_ms": {k: round(v, 1) for k, v in result.timings_ms.items()},
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }
    )


async def label_api(request: Request) -> Response:
    session = _session(request)
    if session is None:
        return _unauthorized()
    if not _csrf_ok(request, session, request.headers.get("x-csrf-token")):
        return JSONResponse({"error": "CSRF check failed"}, status_code=403)
    raw = await _limited_body(request, MAX_FORM_BYTES)
    if raw is None:
        return JSONResponse({"error": "request too large"}, status_code=413)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    if not isinstance(payload, dict):
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    query = payload.get("q")
    kind = payload.get("kind")
    key = payload.get("key")
    grade = payload.get("grade")
    if not isinstance(query, str) or not query.strip() or len(query) > 1000:
        return JSONResponse({"error": "missing query"}, status_code=400)
    if kind == "segment":
        valid_key = isinstance(key, str) and 0 < len(key) <= 128 and set(key) <= SEGMENT_KEY_CHARS
    elif kind == "message":
        valid_key = isinstance(key, str) and is_opaque_key(key)
    else:
        valid_key = False
    if not valid_key or grade not in (None, 0, 2):
        return JSONResponse({"error": "invalid label"}, status_code=400)
    deps = _deps(request)

    def write() -> label_store.LabelOutcome:
        with deps.pool.connection() as pg:
            return label_store.write_label(
                pg, query_text=query, kind=kind, key=str(key), grade=grade  # type: ignore[arg-type]
            )

    try:
        outcome = await run_in_threadpool(write)
    except label_store.UnknownHitError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    return JSONResponse(
        {
            "query_id": outcome.query_id,
            "grade": outcome.grade,
            "counts": {
                "total": outcome.counts.total,
                "relevant": outcome.counts.relevant,
                "not_relevant": outcome.counts.not_relevant,
            },
        }
    )


def labels_view(request: Request) -> Response:
    session = _session(request)
    if session is None:
        return _login_redirect(request)
    deps = _deps(request)
    ctx = _page_ctx(request, session, title="Labels · Messages")
    with deps.pool.connection() as pg:
        check = label_store.baseline_progress(pg)
        queries = label_store.labelled_queries(pg)
    return HTMLResponse(
        views.labels_page(
            ctx,
            query_count=check.query_count,
            judgment_count=check.pooled_judgment_count,
            queries_with_relevant=check.queries_with_a_positive,
            passed=check.passed,
            minimums=(AT4_MIN_QUERIES, AT4_MIN_POOLED_JUDGMENTS, AT4_MIN_QUERIES_WITH_A_POSITIVE),
            queries=queries,
        )
    )


def people_api(request: Request) -> Response:
    session = _session(request)
    if session is None:
        return _unauthorized()
    deps = _deps(request)
    text = request.query_params.get("q", "")[:100]
    with deps.pool.connection() as pg:
        rows = deps.people.search(pg, text)
    return JSONResponse([{"name": d, "short": s, "count": n} for d, s, n in rows])


def thread_view(request: Request) -> Response:
    session = _session(request)
    if session is None:
        return _login_redirect(request)
    deps = _deps(request)
    ctx = _page_ctx(request, session, title="Conversation · Messages")
    thread_key = str(request.path_params.get("thread_key", ""))
    anchor = request.query_params.get("anchor") or None
    query = request.query_params.get("q", "")[:1000]
    with deps.pool.connection() as pg:
        chat = chat_by_thread_key(pg, thread_key)
        if chat is None:
            return HTMLResponse(
                views.error_page(ctx, status=404, message="No such conversation."), status_code=404
            )
        window = thread_window(
            pg,
            chat,
            anchor_key=anchor if anchor and is_opaque_key(anchor) else None,
            before=THREAD_WINDOW_EACH_SIDE,
            after=THREAD_WINDOW_EACH_SIDE,
            index_unsent=deps.settings.index_unsent,
        )
    matcher = QueryMatcher.for_query(analyze_query(query)) if query.strip() else None
    messages_html = views.messages_with_day_breaks(
        window.messages,
        tz=deps.timezone,
        matcher=matcher,
        group=chat.kind == "group" or chat.is_holding,
        anchor_key=window.anchor_key,
        thread_key=chat.thread_key,
        conversation=chat.title,
    )
    back_url = views.FormState(query=query).url() if query.strip() else None
    return HTMLResponse(
        views.thread_page(
            ctx,
            chat=chat,
            messages_html=messages_html,
            has_older=window.has_older,
            has_newer=window.has_newer,
            first_key=window.messages[0].message_key if window.messages else None,
            last_key=window.messages[-1].message_key if window.messages else None,
            query=query,
            back_url=back_url,
        )
    )


def thread_messages(request: Request) -> Response:
    session = _session(request)
    if session is None:
        return _unauthorized()
    deps = _deps(request)
    thread_key = str(request.path_params.get("thread_key", ""))
    cursor = request.query_params.get("cursor", "")
    direction = request.query_params.get("dir", "older")
    query = request.query_params.get("q", "")[:1000]
    if not is_opaque_key(cursor) or direction not in ("older", "newer"):
        return JSONResponse({"error": "bad cursor"}, status_code=400)
    with deps.pool.connection() as pg:
        chat = chat_by_thread_key(pg, thread_key)
        if chat is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        window = thread_page(
            pg,
            chat,
            cursor_key=cursor,
            older=direction == "older",
            limit=THREAD_PAGE_SIZE,
            index_unsent=deps.settings.index_unsent,
        )
    matcher = QueryMatcher.for_query(analyze_query(query)) if query.strip() else None
    body = views.messages_with_day_breaks(
        window.messages,
        tz=deps.timezone,
        matcher=matcher,
        group=chat.kind == "group" or chat.is_holding,
        anchor_key=None,
        thread_key=chat.thread_key,
        conversation=chat.title,
    )
    messages = window.messages
    next_cursor = (messages[0] if direction == "older" else messages[-1]).message_key if messages else None
    more = window.has_older if direction == "older" else window.has_newer
    return JSONResponse({"html": body, "cursor": next_cursor, "more": bool(more and next_cursor)})


# --------------------------------------------------------------------------
# routes: attachments
# --------------------------------------------------------------------------


def _attachment(request: Request) -> tuple[AttachmentRecord, Path] | Response:
    session = _session(request)
    if session is None:
        return _unauthorized()
    deps = _deps(request)
    key = str(request.path_params.get("key", ""))
    with deps.pool.connection() as pg:
        record = lookup_attachment(pg, key)
    if record is None:
        return PlainTextResponse("not found", status_code=404)
    path = cache_file(deps.data_root, record.sha256)
    if path is None:
        return PlainTextResponse("not available", status_code=404)
    return record, path


def _file(path: Path, media_type: str, headers: dict[str, str]) -> Response:
    return FileResponse(path, media_type=media_type, headers=headers)


_DERIVED_HEADERS = {
    "Cache-Control": "private, max-age=86400",
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "sandbox; default-src 'none'; img-src 'self'; media-src 'self'",
    "Cross-Origin-Resource-Policy": "same-origin",
}


def attachment_original(request: Request) -> Response:
    found = _attachment(request)
    if isinstance(found, Response):
        return found
    record, path = found
    plan = serve_plan(record, download=request.query_params.get("download") == "1")
    return _file(path, plan.media_type, plan.headers)


def attachment_thumb(request: Request) -> Response:
    found = _attachment(request)
    if isinstance(found, Response):
        return found
    record, path = found
    deps = _deps(request)
    assert record.sha256 is not None
    if record.kind == "image":
        thumb = deps.media.image_thumbnail(path, record.sha256)
    elif record.kind == "video":
        thumb = deps.media.video_poster(path, record.sha256)
    else:
        thumb = None
    if thumb is None:
        return PlainTextResponse("no thumbnail", status_code=404)
    return _file(thumb, "image/jpeg", dict(_DERIVED_HEADERS))


def attachment_poster(request: Request) -> Response:
    found = _attachment(request)
    if isinstance(found, Response):
        return found
    record, path = found
    deps = _deps(request)
    assert record.sha256 is not None
    poster = deps.media.video_poster(path, record.sha256) if record.kind == "video" else None
    if poster is None:
        return PlainTextResponse("no poster", status_code=404)
    return _file(poster, "image/jpeg", dict(_DERIVED_HEADERS))


def attachment_view(request: Request) -> Response:
    found = _attachment(request)
    if isinstance(found, Response):
        return found
    record, path = found
    deps = _deps(request)
    assert record.sha256 is not None
    if record.kind == "image" and needs_image_preview(record):
        preview = deps.media.image_preview(path, record.sha256)
        if preview is not None:
            return _file(preview, "image/jpeg", dict(_DERIVED_HEADERS))
    plan = serve_plan(record, download=False)
    return _file(path, plan.media_type, plan.headers)


def attachment_audio(request: Request) -> Response:
    found = _attachment(request)
    if isinstance(found, Response):
        return found
    record, path = found
    deps = _deps(request)
    assert record.sha256 is not None
    if needs_audio_preview(record):
        converted = deps.media.audio_preview(path, record.sha256)
        if converted is not None:
            return _file(converted, "audio/mp4", dict(_DERIVED_HEADERS))
    plan = serve_plan(record, download=False)
    return _file(path, plan.media_type, plan.headers)


# --------------------------------------------------------------------------
# routes: login, logout, static
# --------------------------------------------------------------------------


def _set_cookie(response: Response, name: str, value: str, *, max_age: int, secure: bool) -> None:
    response.headers.append("set-cookie", cookie_header(name, value, max_age=max_age, secure=secure))


def login_get(request: Request) -> Response:
    next_url = _safe_next(request.query_params.get("next"))
    if _session(request) is not None:
        return RedirectResponse(next_url, status_code=303)
    secure = _secure(request)
    token = new_login_token()
    response = HTMLResponse(views.login_page(login_token=token, next_url=next_url, error=None))
    _set_cookie(response, login_cookie_name(secure), token, max_age=3600, secure=secure)
    return response


async def login_post(request: Request) -> Response:
    deps = _deps(request)
    secure = _secure(request)
    raw = await _limited_body(request, MAX_FORM_BYTES)
    if raw is None:
        return PlainTextResponse("request too large", status_code=413)
    form = {k: v[0] for k, v in parse_qs(raw.decode("utf-8", "replace"), max_num_fields=10).items()}
    next_url = _safe_next(form.get("next"))
    cookie_token = request.cookies.get(login_cookie_name(secure)) or request.cookies.get(
        login_cookie_name(not secure)
    )

    def refuse(message: str, status: int) -> Response:
        token = new_login_token()
        response = HTMLResponse(
            views.login_page(login_token=token, next_url=next_url, error=message), status_code=status
        )
        _set_cookie(response, login_cookie_name(secure), token, max_age=3600, secure=secure)
        return response

    if not same_site_request(
        host=request.headers.get("host", ""),
        origin=request.headers.get("origin"),
        sec_fetch_site=request.headers.get("sec-fetch-site"),
    ) or not tokens_match(cookie_token, form.get("login_token")):
        return refuse("The sign-in form expired. Try again.", 403)
    password = form.get("password", "")
    try:
        verdict, fingerprint = await run_in_threadpool(deps.login_guard.attempt, _client(request), password)
    except SecretFileError:
        logger.error("search page: the password file is missing or unsafe")
        return refuse("Sign-in is not set up on this server.", 503)
    if verdict == "throttled":
        return refuse("Too many failed attempts. Wait a few minutes and try again.", 429)
    if verdict != "ok":
        return refuse("Wrong password.", 401)
    raw_token, _session_row = await run_in_threadpool(deps.sessions.create, fingerprint)
    response = RedirectResponse(next_url, status_code=303)
    max_age = deps.page.session_days * 86400
    _set_cookie(response, session_cookie_name(secure), raw_token, max_age=max_age, secure=secure)
    _set_cookie(response, login_cookie_name(secure), "", max_age=0, secure=secure)
    return response


async def logout(request: Request) -> Response:
    deps = _deps(request)
    session = _session(request)
    raw = await _limited_body(request, MAX_FORM_BYTES)
    form = {k: v[0] for k, v in parse_qs((raw or b"").decode("utf-8", "replace"), max_num_fields=10).items()}
    if session is None:
        return RedirectResponse("/login", status_code=303)
    if not _csrf_ok(request, session, form.get("csrf_token")):
        return PlainTextResponse("CSRF check failed", status_code=403)
    await run_in_threadpool(deps.sessions.revoke, _raw_session_token(request))
    secure = _secure(request)
    response = RedirectResponse("/login", status_code=303)
    _set_cookie(response, session_cookie_name(secure), "", max_age=0, secure=secure)
    return response


def static_file(request: Request) -> Response:
    name = str(request.path_params.get("name", ""))
    media_type = STATIC_FILES.get(name)
    if media_type is None:
        return PlainTextResponse("not found", status_code=404)
    body = resources.files("imsg.search_page").joinpath("static", name).read_bytes()
    return Response(
        body,
        media_type=media_type,
        headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"},
    )


def build_app(deps: AppDeps) -> ASGIApp:
    routes = [
        Route("/", home, methods=["GET"]),
        Route("/search", search_view, methods=["GET"]),
        Route("/search/page", search_page_fragment, methods=["GET"]),
        Route("/search/thread", search_thread_hits, methods=["GET"]),
        Route("/api/semantic", semantic_api, methods=["GET"]),
        Route("/api/label", label_api, methods=["POST"]),
        Route("/api/people", people_api, methods=["GET"]),
        Route("/labels", labels_view, methods=["GET"]),
        Route("/thread/{thread_key}", thread_view, methods=["GET"]),
        Route("/thread/{thread_key}/messages", thread_messages, methods=["GET"]),
        Route("/att/{key}", attachment_original, methods=["GET"]),
        Route("/att/{key}/thumb", attachment_thumb, methods=["GET"]),
        Route("/att/{key}/view", attachment_view, methods=["GET"]),
        Route("/att/{key}/poster", attachment_poster, methods=["GET"]),
        Route("/att/{key}/audio", attachment_audio, methods=["GET"]),
        Route("/login", login_get, methods=["GET"]),
        Route("/login", login_post, methods=["POST"]),
        Route("/logout", logout, methods=["POST"]),
        Route("/static/{name}", static_file, methods=["GET"]),
    ]
    app = Starlette(routes=routes)
    app.state.deps = deps
    return GuardMiddleware(app, allowed_hosts=deps.page.allowed_hosts)


__all__ = [
    "HTML_CSP",
    "AppDeps",
    "ConnectionPool",
    "FtsReaders",
    "GuardMiddleware",
    "PeopleDirectory",
    "build_app",
    "build_thread_views",
]
