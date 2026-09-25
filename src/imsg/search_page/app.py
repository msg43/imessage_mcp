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
import dataclasses
import hashlib
import json
import logging
import queue
import secrets
import threading
import time
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, quote, urlsplit
from zoneinfo import ZoneInfo

import psycopg
from starlette.applications import Starlette
from starlette.background import BackgroundTask
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
from imsg.retrieval.errors import PersonAmbiguousError, PersonNotFoundError
from imsg.retrieval.filters import MAX_PEOPLE_FILTER
from imsg.retrieval.people import resolve_people
from imsg.retrieval.query import analyze_query
from imsg.search_page import browse, grading
from imsg.search_page import cases as case_store
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
from imsg.search_page.details import message_details
from imsg.search_page.errors import ModelApiUnavailable, SearchInputError, SecretFileError
from imsg.search_page.highlight import QueryMatcher
from imsg.search_page.search import (
    CHANNEL_ATTACHMENT_SEMANTIC,
    CHANNEL_ATTACHMENT_TEXT,
    CHANNEL_FILTERS,
    CHANNEL_IMAGE,
    CHANNEL_LABELS,
    CHANNEL_SEMANTIC,
    CHANNEL_TEXT,
    CHANNEL_UNINDEXED,
    FULL_SCOPE,
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
from imsg.search_page.secret_files import ensure_private_dir
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
    return views.FormState.from_mapping(request.query_params)


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
        sender=form.sender,
        direction=form.direction,  # type: ignore[arg-type]
        chat_kind=form.chat_kind,  # type: ignore[arg-type]
        thread=form.thread,
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


def _passing(messages: list[MessageView], result: SearchResult) -> list[MessageView]:
    """Only the messages the per-message filters keep (dates, who sent it),
    when the search has any: a segment that overlaps the range may begin
    or end outside it, and holds other people's messages too."""
    filters = result.filters
    if filters.after is None and filters.before is None and not filters.by_sender:
        return messages
    return [
        m
        for m in messages
        if filters.keeps(
            sent_at=m.sent_at, is_from_me=m.is_from_me, sender_person_id=m.sender_person_id
        )
    ]


def matcher_for(result: SearchResult) -> QueryMatcher:
    """The words' matcher; one that matches nothing for a search by filters alone."""
    return QueryMatcher.for_query(result.analyzed) if result.analyzed is not None else QueryMatcher("bm25")


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
        segment_id: _passing(messages, result)
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


def _thread_view_keys(thread_views: list[views.ThreadResultView]) -> set[str]:
    return {m.message_key for t in thread_views for h in t.hits for m in h.messages}


def _review_state(pg: psycopg.Connection, form: views.FormState) -> views.ReviewState | None:
    saved = case_store.saved_search_for(pg, form.query, form.params())
    if saved is None:
        return None
    return views.ReviewState(
        saved.search_id, saved.reviewed, case_id=saved.case_id, case_name=saved.case_name, name=saved.name
    )


def _render_results(
    deps: AppDeps,
    pg: psycopg.Connection,
    result: SearchResult,
    form: views.FormState,
    page: int,
    query_id: str | None,
    review: views.ReviewState | None = None,
) -> str:
    matcher = matcher_for(result)
    slice_, more = _page_slice(deps, result, form.sort, page)
    thread_views = build_thread_views(deps, pg, result, slice_, matcher=matcher, query_id=query_id)
    marks = case_store.marks(pg, _thread_view_keys(thread_views))
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
        marks=marks,
        review=review,
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
                    assert result.analyzed is not None  # a search with no words is never pending
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
        if (
            sort == "rerank"
            and not result.reranked
            and deps.model_api is not None
            and result.analyzed is not None
        ):
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
    if not form.query.strip() and not form.has_filters:
        return RedirectResponse("/", status_code=303)
    started = time.perf_counter()
    try:
        search_request = _search_request(form)
        with deps.pool.connection() as pg:
            result = _result_for(deps, pg, search_request)
            query_id = label_store.find_query_id(pg, result.request.query)
            review = _review_state(pg, form)
            active = case_store.active_case(pg)
            with result.lock:
                results_html = _render_results(
                    deps, pg, result, form, _page_number(request), query_id, review
                )
                status = _status(result, form.sort)
            box = views.case_box(
                form, active=active, review=review, total_threads=status.total_threads
            )
            only_in = chat_by_thread_key(pg, form.thread) if form.thread else None
    except SearchInputError as exc:
        return HTMLResponse(
            views.search_page(
                ctx, form=form, status=None, results="", error=str(exc), semantic_url=None, search_key=None
            ),
            status_code=400,
        )
    semantic_url = None
    if result.semantic.state == "pending" or (
        form.sort == "rerank" and not result.reranked and result.analyzed is not None
    ):
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
            case_box=box,
            thread_title=only_in.title if only_in is not None else None,
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
            review = _review_state(pg, form)
            with result.lock:
                body = _render_results(
                    deps, pg, result, form, _page_number(request), query_id, review
                )
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
            matcher = matcher_for(result)
            query_id = label_store.find_query_id(pg, result.request.query)
            with result.lock:
                group = result.thread(chat.chat_id, form.sort) if chat else None  # type: ignore[arg-type]
                if chat is None or group is None:
                    return PlainTextResponse("not found", status_code=404)
                [view] = build_thread_views(
                    deps, pg, result, [group], matcher=matcher, query_id=query_id, hit_offset=offset
                )
            marks = case_store.marks(pg, _thread_view_keys([view]))
            review = _review_state(pg, form)
    except SearchInputError as exc:
        return PlainTextResponse(str(exc), status_code=400)
    render = views.thread_result_html if offset == 0 else views.thread_hits_html
    return HTMLResponse(
        render(
            view, tz=deps.timezone, matcher=matcher, form=form, query=form.query, marks=marks, review=review
        )
    )


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
        graded_lists = grading.load_graded_lists(pg)
    return HTMLResponse(
        views.labels_page(
            ctx,
            query_count=check.query_count,
            judgment_count=check.pooled_judgment_count,
            queries_with_relevant=check.queries_with_a_positive,
            passed=check.passed,
            minimums=(AT4_MIN_QUERIES, AT4_MIN_POOLED_JUDGMENTS, AT4_MIN_QUERIES_WITH_A_POSITIVE),
            queries=queries,
            graded_lists=graded_lists,
        )
    )


# --------------------------------------------------------------------------
# routes: grading mode
# --------------------------------------------------------------------------


def _form_fields(raw: bytes) -> dict[str, str]:
    return {
        k: v[0]
        for k, v in parse_qs(raw.decode("utf-8", "replace"), max_num_fields=20).items()
    }


async def grade_start(request: Request) -> Response:
    """Store the search's fused candidate list and open its grading view.
    A POST with the CSRF token: it writes to the database."""
    session = _session(request)
    if session is None:
        return _login_redirect(request)
    raw = await _limited_body(request, MAX_FORM_BYTES)
    if raw is None:
        return PlainTextResponse("request too large", status_code=413)
    fields = _form_fields(raw)
    if not _csrf_ok(request, session, fields.get("csrf_token")):
        return PlainTextResponse("CSRF check failed", status_code=403)
    deps = _deps(request)
    form = dataclasses.replace(views.FormState.from_mapping(fields), sort="relevance")
    filters: dict[str, object] = {}
    people = [p.strip() for p in form.people.split(",") if p.strip()]
    if people:
        filters["people"] = people
    if form.date_from:
        filters["from"] = form.date_from
    if form.date_to:
        filters["to"] = form.date_to
    if form.attachments != "any":
        filters["attachments"] = form.attachments
    if form.sender.strip():
        filters["sender"] = form.sender.strip()
    if form.direction != "any":
        filters["direction"] = form.direction
    if form.chat_kind != "any":
        filters["conversations"] = form.chat_kind
    if form.thread:
        filters["conversation"] = form.thread

    def create() -> int:
        if not form.query.strip():
            raise SearchInputError("Grading needs a search with words.")
        search_request = _search_request(form)
        with deps.pool.connection() as pg:
            result = _result_for(deps, pg, search_request)
            _ensure_semantic(deps, pg, result, "relevance")
            with result.lock:
                ranking = {
                    "rrf_k": result.rrf_k,
                    "semantic": result.semantic.state,
                    "semantic_note": result.semantic.note,
                    "text_min_similarity": deps.settings.text_min_similarity,
                    "multimodal_min_similarity": deps.settings.multimodal_min_similarity,
                    "counts": dict(result.counts),
                    "capped": sorted(result.capped),
                    "total_hits": result.total_hits,
                    "segment_hits": sum(1 for h in result.hits.values() if h.segment_id is not None),
                }
                return grading.create_candidate_list(
                    pg,
                    result,
                    query_text=result.request.query,
                    filters=filters,
                    ranking=ranking,
                    seed=secrets.randbelow(2**31),
                )

    ctx = _page_ctx(request, session, title="Grade · Messages")
    try:
        list_id = await run_in_threadpool(create)
    except SearchInputError as exc:
        return HTMLResponse(
            views.search_page(
                ctx, form=form, status=None, results="", error=str(exc), semantic_url=None, search_key=None
            ),
            status_code=400,
        )
    return RedirectResponse(f"/grade/{list_id}", status_code=303)


def grade_view(request: Request) -> Response:
    session = _session(request)
    if session is None:
        return _login_redirect(request)
    deps = _deps(request)
    raw_id = str(request.path_params.get("list_id", ""))
    if not raw_id.isdigit() or len(raw_id) > 18:
        return PlainTextResponse("not found", status_code=404)
    positions = (
        grading.MAX_POSITIONS if request.query_params.get("more") == "1" else grading.GRADED_POSITIONS
    )
    ctx = _page_ctx(request, session, title="Grade · Messages")
    with deps.pool.connection() as pg:
        graded = grading.load_graded_list(pg, int(raw_id))
        if graded is None:
            return HTMLResponse(
                views.error_page(ctx, status=404, message="No such graded search."), status_code=404
            )
        shown = graded.in_view(positions)
        segment_ids = [c.current_segment_id for c in shown if c.current_segment_id is not None]
        by_segment = segment_messages(pg, segment_ids, index_unsent=deps.settings.index_unsent)
        decorate(pg, [m for ms in by_segment.values() for m in ms])
        chats = chat_views(pg, {ms[0].chat_id for ms in by_segment.values() if ms})
        texts = attachment_chunk_text(
            pg, {a.attachment_id for ms in by_segment.values() for m in ms for a in m.attachments}
        )
    candidate_views = []
    for candidate in shown:
        messages = by_segment.get(candidate.current_segment_id or -1, [])
        chat = chats.get(messages[0].chat_id) if messages else None
        own = {a.attachment_id for m in messages for a in m.attachments}
        candidate_views.append(
            views.CandidateView(
                candidate=candidate,
                chat=chat,
                messages=messages,
                attachment_text={k: v for k, v in texts.items() if k in own},
            )
        )
    matcher = QueryMatcher.for_query(analyze_query(graded.query_text))
    return HTMLResponse(
        views.grading_page(
            ctx,
            graded=graded,
            views=candidate_views,
            positions=positions,
            tz=deps.timezone,
            matcher=matcher,
        )
    )


async def grade_api(request: Request) -> Response:
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
    list_id = payload.get("list")
    anchor = payload.get("anchor")
    grade = payload.get("grade")
    if (
        not isinstance(list_id, int)
        or isinstance(list_id, bool)
        or not isinstance(anchor, str)
        or not 0 < len(anchor) <= 200
        or (grade is not None and (isinstance(grade, bool) or grade not in grading.GRADES))
    ):
        return JSONResponse({"error": "invalid grade"}, status_code=400)
    deps = _deps(request)

    def write() -> grading.GradeOutcome:
        with deps.pool.connection() as pg:
            return grading.write_grade(pg, list_id=list_id, anchor_guid=anchor, grade=grade)

    try:
        outcome = await run_in_threadpool(write)
    except grading.UnknownCandidateError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    return JSONResponse(
        {
            "grade": outcome.grade,
            "graded": outcome.graded,
            "graded_extra": outcome.graded_extra,
            "total": outcome.total,
        }
    )


def message_details_view(request: Request) -> Response:
    """The Details panel for one message, as an HTML fragment."""
    session = _session(request)
    if session is None:
        return _unauthorized()
    deps = _deps(request)
    key = str(request.path_params.get("message_key", ""))
    with deps.pool.connection() as pg:
        details = message_details(
            pg,
            key,
            index_unsent=deps.settings.index_unsent,
            show_edit_history=deps.page.details.show_edit_history,
            show_raw_handles=deps.page.details.show_raw_handles,
        )
    if details is None:
        return PlainTextResponse("not found", status_code=404)
    return HTMLResponse(views.details_html(details, tz=deps.timezone))


# --------------------------------------------------------------------------
# routes: timeline and media
# --------------------------------------------------------------------------


def _browse_form(request: Request) -> views.BrowseForm:
    params = request.query_params
    media_type = params.get("type", "all")
    return views.BrowseForm(
        date_from=params.get("from", "")[:10],
        date_to=params.get("to", "")[:10],
        people=params.get("people", "")[:500],
        sender=params.get("sender", "")[:200],
        query=params.get("q", "")[:1000],
        media_type=media_type[:20],
    )


def _parse_day(value: str, label: str) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise SearchInputError(f"The '{label}' date is not a valid date.") from exc


def _resolve_browse_filters(
    pg: psycopg.Connection, form: views.BrowseForm, timezone: str
) -> tuple[browse.BrowseFilters, date | None, date | None]:
    """People and sender resolved with the retrieval layer's own person
    matching; dates parsed but not yet defaulted."""
    first = _parse_day(form.date_from, "from")
    last = _parse_day(form.date_to, "to")
    first, last = (first or last), (last or first)  # one date alone is that one day
    if first is not None and last is not None and first > last:
        raise SearchInputError("The 'from' date must be before the 'to' date.")
    names = [p.strip() for p in form.people.split(",") if p.strip()]
    if len(names) > MAX_PEOPLE_FILTER:
        raise SearchInputError(f"Filter by at most {MAX_PEOPLE_FILTER} people.")
    try:
        people = resolve_people(pg, FULL_SCOPE, names) if names else ()
        sender: int | None = None
        sender_is_owner = form.sender.strip().lower() in ("me", "owner")
        if form.sender.strip() and not sender_is_owner:
            [sender] = resolve_people(pg, FULL_SCOPE, [form.sender.strip()])
    except PersonAmbiguousError as exc:
        found = ", ".join(f"{c.display_name} ({c.short_name})" for c in exc.candidates)
        raise SearchInputError(f"More than one person matches {exc.query!r}: {found}. Pick one.") from exc
    except PersonNotFoundError as exc:
        raise SearchInputError(f"No person matches {exc.query!r}.") from exc
    start = end = None
    if first is not None and last is not None:
        start, end = browse.day_bounds(first, last, timezone)
    return (
        browse.BrowseFilters(
            people=tuple(people), sender=sender, sender_is_owner=sender_is_owner, start=start, end=end
        ),
        first,
        last,
    )


def _range_heading(first: date, last: date, total: int, conversations: int) -> str:
    def label(d: date) -> str:
        return d.strftime("%a %d %b %Y")

    span = label(first) if first == last else f"{label(first)} \u2013 {label(last)}"
    return (
        f"{span} \u00b7 {total:,} message{'s' if total != 1 else ''}"
        + (f" in {conversations:,} conversation{'s' if conversations != 1 else ''}" if total else "")
    )


def timeline_view(request: Request) -> Response:
    session = _session(request)
    if session is None:
        return _login_redirect(request)
    deps = _deps(request)
    ctx = _page_ctx(request, session, title="Timeline · Messages")
    form = _browse_form(request)
    tz = deps.timezone
    try:
        with deps.pool.connection() as pg:
            filters, first, last = _resolve_browse_filters(pg, form, tz)
            if first is None and last is None:
                latest = browse.latest_day(pg, filters, index_unsent=deps.settings.index_unsent, timezone=tz)
                if latest is None:
                    return HTMLResponse(
                        views.timeline_page(
                            ctx, form=form, heading="No messages match.", day_counts=[],
                            rows_html="", next_url=None, error=None,
                        )
                    )
                first = last = latest
                start, end = browse.day_bounds(latest, latest, tz)
                filters = dataclasses.replace(filters, start=start, end=end)
                form = dataclasses.replace(form, date_from=latest.isoformat(), date_to=latest.isoformat())
            assert first is not None and last is not None
            counts = browse.day_counts(pg, filters, index_unsent=deps.settings.index_unsent, timezone=tz)
            page = browse.timeline_page(pg, filters, index_unsent=deps.settings.index_unsent, cursor=None)
            chats = chat_views(pg, {m.chat_id for m in page.messages})
            marks = case_store.marks(pg, [m.message_key for m in page.messages])
            conversations = browse.conversation_count(
                pg, filters, index_unsent=deps.settings.index_unsent
            )
    except SearchInputError as exc:
        return HTMLResponse(
            views.timeline_page(
                ctx, form=form, heading="", day_counts=[], rows_html="", next_url=None, error=str(exc)
            ),
            status_code=400,
        )
    total = sum(n for _d, n in counts)
    matcher = QueryMatcher.for_query(analyze_query(form.query)) if form.query.strip() else None
    rows = views.timeline_rows_html(
        page.messages, chats=chats, tz=tz, matcher=matcher, previous_day=None, marks=marks
    )
    day_links = [
        (
            d.strftime("%a %d %b %Y"),
            dataclasses.replace(form, date_from=d.isoformat(), date_to=d.isoformat()).url("/timeline"),
            n,
        )
        for d, n in counts
    ]
    next_url = form.url("/timeline/page", cursor=page.next_cursor) if page.next_cursor else None
    return HTMLResponse(
        views.timeline_page(
            ctx,
            form=form,
            heading=_range_heading(first, last, total, conversations),
            day_counts=day_links,
            rows_html=rows,
            next_url=next_url,
            error=None,
        )
    )


def timeline_more(request: Request) -> Response:
    """The next 200 timeline rows, for the page script's endless scroll."""
    session = _session(request)
    if session is None:
        return _unauthorized()
    deps = _deps(request)
    form = _browse_form(request)
    cursor = request.query_params.get("cursor", "")
    position = browse.decode_cursor(cursor, 1)
    if position is None:
        return PlainTextResponse("bad cursor", status_code=400)
    tz = deps.timezone
    try:
        with deps.pool.connection() as pg:
            filters, first, last = _resolve_browse_filters(pg, form, tz)
            if first is None and last is None:
                return PlainTextResponse("a date range is required", status_code=400)
            page = browse.timeline_page(pg, filters, index_unsent=deps.settings.index_unsent, cursor=cursor)
            chats = chat_views(pg, {m.chat_id for m in page.messages})
            marks = case_store.marks(pg, [m.message_key for m in page.messages])
    except SearchInputError as exc:
        return PlainTextResponse(str(exc), status_code=400)
    matcher = QueryMatcher.for_query(analyze_query(form.query)) if form.query.strip() else None
    previous_day = views.fmt_date(position[0], tz)
    body = views.timeline_rows_html(
        page.messages, chats=chats, tz=tz, matcher=matcher, previous_day=previous_day, marks=marks
    )
    if page.next_cursor:
        next_url = form.url("/timeline/page", cursor=page.next_cursor)
        body += (
            f'<div class="sentinel" data-next="{views.esc(next_url)}"><a href="{views.esc(next_url)}" '
            'class="load-more">Later messages</a></div>'
        )
    return HTMLResponse(body)


def _media_kind(form: views.BrowseForm) -> str | None:
    if form.media_type == "all":
        return None
    if form.media_type not in browse.MEDIA_KINDS:
        raise SearchInputError("Unknown file type.")
    return form.media_type


def media_view(request: Request) -> Response:
    session = _session(request)
    if session is None:
        return _login_redirect(request)
    deps = _deps(request)
    ctx = _page_ctx(request, session, title="Media · Messages")
    form = _browse_form(request)
    try:
        kind = _media_kind(form)
        with deps.pool.connection() as pg:
            filters, _first, _last = _resolve_browse_filters(pg, form, deps.timezone)
            counts = browse.media_counts(pg, filters, index_unsent=deps.settings.index_unsent)
            page = browse.media_page(
                pg, filters, kind=kind, index_unsent=deps.settings.index_unsent, cursor=None
            )
            chats = chat_views(pg, {item.chat_id for item in page.items})
            marks = case_store.marks(pg, [item.message_key for item in page.items])
    except SearchInputError as exc:
        return HTMLResponse(
            views.media_page(
                ctx, form=form, counts={}, labels=browse.MEDIA_KIND_LABELS, tiles_html="",
                next_url=None, error=str(exc),
            ),
            status_code=400,
        )
    tiles = views.media_tiles_html(page.items, chats=chats, tz=deps.timezone, marks=marks)
    next_url = form.url("/media/page", cursor=page.next_cursor) if page.next_cursor else None
    return HTMLResponse(
        views.media_page(
            ctx, form=form, counts=counts, labels=browse.MEDIA_KIND_LABELS, tiles_html=tiles,
            next_url=next_url, error=None,
        )
    )


def media_more(request: Request) -> Response:
    session = _session(request)
    if session is None:
        return _unauthorized()
    deps = _deps(request)
    form = _browse_form(request)
    cursor = request.query_params.get("cursor", "")
    if browse.decode_cursor(cursor, 2) is None:
        return PlainTextResponse("bad cursor", status_code=400)
    try:
        kind = _media_kind(form)
        with deps.pool.connection() as pg:
            filters, _first, _last = _resolve_browse_filters(pg, form, deps.timezone)
            page = browse.media_page(
                pg, filters, kind=kind, index_unsent=deps.settings.index_unsent, cursor=cursor
            )
            chats = chat_views(pg, {item.chat_id for item in page.items})
            marks = case_store.marks(pg, [item.message_key for item in page.items])
    except SearchInputError as exc:
        return PlainTextResponse(str(exc), status_code=400)
    body = views.media_tiles_html(page.items, chats=chats, tz=deps.timezone, marks=marks)
    if page.next_cursor:
        next_url = form.url("/media/page", cursor=page.next_cursor)
        body += (
            f'<div class="sentinel" data-next="{views.esc(next_url)}"><a href="{views.esc(next_url)}" '
            'class="load-more">More</a></div>'
        )
    return HTMLResponse(body)


# --------------------------------------------------------------------------
# routes: evidence cases
# --------------------------------------------------------------------------

MAX_DOWNLOAD_FILE_BYTES = 2 * 1024 * 1024 * 1024
"""The original files a case download may carry: 2 GiB. The zip is
written on the encrypted volume, then streamed and deleted."""
DOWNLOAD_STALE_SECONDS = 3600.0
_SLUG_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")


class _ExactTimes:
    def __init__(self, timezone: str) -> None:
        self._tz = timezone

    def exact(self, dt: datetime) -> str:
        return views.exact_time(dt, self._tz)


def _default_case_name(timezone: str) -> str:
    return "Case started " + views.fmt_date(datetime.now(UTC), timezone)


async def _form_post(request: Request) -> tuple[Session, dict[str, str]] | Response:
    """A form POST from the page: signed in, same-site, CSRF token right."""
    session = _session(request)
    if session is None:
        return _login_redirect(request)
    raw = await _limited_body(request, 64 * 1024)
    if raw is None:
        return PlainTextResponse("request too large", status_code=413)
    fields = _form_fields(raw)
    if not _csrf_ok(request, session, fields.get("csrf_token")):
        return PlainTextResponse("CSRF check failed", status_code=403)
    return session, fields


async def _json_post(request: Request) -> tuple[Session, dict[str, object]] | Response:
    """A script POST from the page: signed in, same-site, CSRF header right."""
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
    return session, payload


def _id_param(request: Request, name: str) -> int | None:
    raw = str(request.path_params.get(name, ""))
    return int(raw) if raw.isdigit() and len(raw) <= 18 else None


def cases_view(request: Request) -> Response:
    session = _session(request)
    if session is None:
        return _login_redirect(request)
    deps = _deps(request)
    ctx = _page_ctx(request, session, title="Cases · Messages")
    with deps.pool.connection() as pg:
        found = case_store.list_cases(pg)
    return HTMLResponse(views.cases_page(ctx, cases=found, error=None))


async def case_create(request: Request) -> Response:
    posted = await _form_post(request)
    if isinstance(posted, Response):
        return posted
    session, fields = posted
    deps = _deps(request)

    def create() -> int:
        with deps.pool.connection() as pg:
            return case_store.create_case(pg, fields.get("name", ""))

    try:
        case_id = await run_in_threadpool(create)
    except case_store.CaseError as exc:
        def listing() -> list[case_store.CaseSummary]:
            with deps.pool.connection() as pg:
                return case_store.list_cases(pg)

        ctx = _page_ctx(request, session, title="Cases · Messages")
        found = await run_in_threadpool(listing)
        return HTMLResponse(views.cases_page(ctx, cases=found, error=str(exc)), status_code=400)
    return RedirectResponse(f"/case/{case_id}", status_code=303)


def _coverage(
    deps: AppDeps, pg: psycopg.Connection, case_id: int
) -> list[case_store.SearchCoverage]:
    """Each saved search with how many conversations it finds now."""
    out: list[case_store.SearchCoverage] = []
    for search in case_store.saved_searches(pg, case_id):
        form = views.FormState.from_mapping({**search.params, "q": search.query_text})
        try:
            result = _result_for(deps, pg, _search_request(form))
            with result.lock:
                conversations: int | None = len(result.threads("relevance"))
        except SearchInputError:
            conversations = None
        out.append(case_store.SearchCoverage(search=search, conversations=conversations))
    return out


def case_view(request: Request) -> Response:
    session = _session(request)
    if session is None:
        return _login_redirect(request)
    deps = _deps(request)
    ctx = _page_ctx(request, session, title="Case · Messages")
    case_id = _id_param(request, "case_id")
    with deps.pool.connection() as pg:
        case = case_store.get_case(pg, case_id) if case_id is not None else None
        if case is None:
            return HTMLResponse(views.error_page(ctx, status=404, message="No such case."), status_code=404)
        items = case_store.case_items(
            pg,
            case.case_id,
            index_unsent=deps.settings.index_unsent,
            show_edit_history=deps.page.details.show_edit_history,
            show_raw_handles=deps.page.details.show_raw_handles,
        )
        coverage = _coverage(deps, pg, case.case_id)
    return HTMLResponse(
        views.case_page(
            ctx,
            case=case,
            items=items,
            coverage=coverage,
            tz=deps.timezone,
            show_raw_handles=deps.page.details.show_raw_handles,
            confirm_delete=request.query_params.get("delete") == "1",
            error=None,
        )
    )


def _case_action(
    name: str, action: Callable[[psycopg.Connection, int, dict[str, str]], str]
) -> Callable[[Request], Any]:
    """A form POST on one case (or one of its items or searches) that ends
    with a redirect to the page `action` returns."""

    async def endpoint(request: Request) -> Response:
        posted = await _form_post(request)
        if isinstance(posted, Response):
            return posted
        _session_row, fields = posted
        target_id = _id_param(request, name)
        if target_id is None:
            return PlainTextResponse("not found", status_code=404)
        deps = _deps(request)

        def run() -> str:
            with deps.pool.connection() as pg:
                return action(pg, target_id, fields)

        try:
            location = await run_in_threadpool(run)
        except case_store.CaseError as exc:
            return PlainTextResponse(str(exc), status_code=400)
        return RedirectResponse(location, status_code=303)

    return endpoint


def _activate(pg: psycopg.Connection, case_id: int, _fields: dict[str, str]) -> str:
    case_store.activate_case(pg, case_id)
    return f"/case/{case_id}"


def _rename(pg: psycopg.Connection, case_id: int, fields: dict[str, str]) -> str:
    case_store.rename_case(pg, case_id, fields.get("name", ""))
    return f"/case/{case_id}"


def _notes(pg: psycopg.Connection, case_id: int, fields: dict[str, str]) -> str:
    case_store.set_case_notes(pg, case_id, fields.get("notes", ""))
    return f"/case/{case_id}"


def _delete(pg: psycopg.Connection, case_id: int, fields: dict[str, str]) -> str:
    if fields.get("confirm") != "1":
        return f"/case/{case_id}?delete=1"
    case_store.delete_case(pg, case_id)
    return "/case"


def _item_note(pg: psycopg.Connection, item_id: int, fields: dict[str, str]) -> str:
    case_id = case_store.set_item_note(pg, item_id, fields.get("note", ""))
    return f"/case/{case_id}#item-{item_id}"


def _item_remove(pg: psycopg.Connection, item_id: int, _fields: dict[str, str]) -> str:
    return f"/case/{case_store.remove_item(pg, item_id)}"


def _search_remove(pg: psycopg.Connection, search_id: int, _fields: dict[str, str]) -> str:
    case_id = case_store.remove_search(pg, search_id)
    return f"/case/{case_id}" if case_id is not None else "/saved"


def _saved_remove(pg: psycopg.Connection, search_id: int, _fields: dict[str, str]) -> str:
    case_store.remove_search(pg, search_id)
    return "/saved"


def _saved_rename(pg: psycopg.Connection, search_id: int, fields: dict[str, str]) -> str:
    case_store.rename_search(pg, search_id, fields.get("name", ""))
    return f"/saved#saved-{search_id}"


# --------------------------------------------------------------------------
# saved searches and "Download results"
# --------------------------------------------------------------------------


def saved_view(request: Request) -> Response:
    """Every saved search, on its own or in a case."""
    session = _session(request)
    if session is None:
        return _login_redirect(request)
    deps = _deps(request)
    ctx = _page_ctx(request, session, title="Saved searches · Messages")
    with deps.pool.connection() as pg:
        searches = case_store.all_saved_searches(pg)
        conversations: dict[str, str] = {}
        for key in {s.params["in"] for s in searches if s.params.get("in")}:
            chat = chat_by_thread_key(pg, key)
            if chat is not None:
                conversations[key] = chat.title
    return HTMLResponse(
        views.saved_page(ctx, searches=searches, conversations=conversations, tz=deps.timezone, error=None)
    )


async def saved_api(request: Request) -> Response:
    """Save the shown search on its own, outside any case."""
    posted = await _json_post(request)
    if isinstance(posted, Response):
        return posted
    _session_row, payload = posted
    query = payload.get("q", "")
    name = payload.get("name", "")
    if not isinstance(query, str) or not isinstance(name, str):
        return JSONResponse({"error": "invalid request"}, status_code=400)
    deps = _deps(request)

    def save() -> int:
        with deps.pool.connection() as pg:
            return case_store.save_search_alone(pg, query_text=query, params=payload, name=name)

    try:
        search_id = await run_in_threadpool(save)
    except case_store.CaseError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse({"search_id": search_id})


FOUND_BY = {
    CHANNEL_TEXT: "the words",
    CHANNEL_ATTACHMENT_TEXT: "a file's text",
    CHANNEL_UNINDEXED: "the words (not yet indexed)",
    CHANNEL_SEMANTIC: "meaning",
    CHANNEL_ATTACHMENT_SEMANTIC: "a file's meaning",
    CHANNEL_IMAGE: "an image",
    CHANNEL_FILTERS: "the filters",
}
DOWNLOAD_THREAD_BATCH = 200


def _download_selection(
    deps: AppDeps, pg: psycopg.Connection, result: SearchResult, sort: str
) -> tuple[list[str], dict[str, str], dict[str, int], bool, int]:
    """Which messages "Download results" writes, in the results' order of
    conversations and, within each, the order they were sent: for a hit on
    one message, that message; for a passage, the messages in it that the
    filters keep and that match the words or a matching file, or, when
    none does (found by meaning or by the filters alone), every message
    in it the filters keep. Returns the message keys, how each was found,
    its conversation's place, whether the limit stopped it, and how many
    conversations the results hold."""
    matcher = matcher_for(result)
    threads = result.threads(sort)  # type: ignore[arg-type]
    keys: list[str] = []
    found_by: dict[str, str] = {}
    place: dict[str, int] = {}
    seen: set[str] = set()
    n = 0
    for start in range(0, len(threads), DOWNLOAD_THREAD_BATCH):
        batch = threads[start : start + DOWNLOAD_THREAD_BATCH]
        known = chat_views(pg, [t.chat_id for t in batch])
        batch = [t for t in batch if t.chat_id in known]
        hits = [h for t in batch for h in t.hits]
        by_segment = segment_messages(
            pg, [h.segment_id for h in hits if h.segment_id is not None],
            index_unsent=deps.settings.index_unsent,
        )
        direct = messages_by_id(pg, [h.message_id for h in hits if h.message_id is not None])
        for thread in batch:
            n += 1
            chosen: dict[str, tuple[datetime, int, str]] = {}
            for hit in thread.hits:
                how = ", ".join(FOUND_BY.get(c, c) for c in hit.channels)
                if hit.segment_id is not None:
                    messages = _passing(by_segment.get(hit.segment_id, []), result)
                    attached = {m for m, _a in hit.matched_attachments}
                    matching = [
                        m for m in messages if matcher.matched_terms(m.text) or m.message_id in attached
                    ]
                    picked = matching or messages
                else:
                    picked = [direct[hit.message_id]] if hit.message_id in direct else []
                for m in picked:
                    if m.message_key not in chosen:
                        chosen[m.message_key] = (m.sent_at, m.message_id, how)
            for key, (_at, _id, how) in sorted(chosen.items(), key=lambda kv: (kv[1][0], kv[1][1])):
                if key in seen:
                    continue
                if len(keys) >= case_store.MAX_DOWNLOAD_MESSAGES:
                    return keys, found_by, place, True, len(threads)
                seen.add(key)
                keys.append(key)
                found_by[key] = how
                place[key] = n
    return keys, found_by, place, False, len(threads)


def search_download(request: Request) -> Response:
    """Every result of a search as Markdown, CSV or JSON, with the case
    download's exact citations. Called "download" on the page: "export"
    means the Gemini pipeline in this project."""
    session = _session(request)
    if session is None:
        return _login_redirect(request)
    deps = _deps(request)
    ctx = _page_ctx(request, session, title="Download · Messages")
    fmt = request.query_params.get("fmt", "md")
    if fmt not in case_store.DOWNLOAD_FORMATS:
        return PlainTextResponse("unknown format", status_code=400)
    form = _form_state(request)
    sort = form.sort if form.sort != "rerank" else "relevance"
    try:
        search_request = _search_request(form)
        with deps.pool.connection() as pg:
            result = _result_for(deps, pg, search_request)
            if result.analyzed is not None:
                _ensure_semantic(deps, pg, result, sort)
            with result.lock:
                keys, found_by, place, truncated, conversations = _download_selection(
                    deps, pg, result, sort
                )
                total_hits = result.total_hits
                capped = tuple(sorted(CHANNEL_LABELS.get(c, c) for c in result.capped))
                semantic = {
                    "done": "done",
                    "disabled": "off",
                    "unavailable": "unavailable",
                    "pending": "not run",
                }.get(result.semantic.state, result.semantic.state)
                if result.semantic.note:
                    semantic += f" ({result.semantic.note})"
            items = case_store.message_items(
                pg,
                keys,
                index_unsent=deps.settings.index_unsent,
                show_edit_history=deps.page.details.show_edit_history,
                show_raw_handles=deps.page.details.show_raw_handles,
            )
            only_in = chat_by_thread_key(pg, form.thread) if form.thread else None
    except SearchInputError as exc:
        return HTMLResponse(views.error_page(ctx, status=400, message=str(exc)), status_code=400)
    now = datetime.now(UTC)
    text = case_store.render_search_download(
        fmt,
        search=case_store.SearchDownload(
            query_text=result.request.query,
            filters=views.describe_search_filters(
                form.filter_params(), conversation=only_in.title if only_in else None
            ),
            conversations=conversations,
            hits=total_hits,
            semantic=semantic,
            capped=capped,
            truncated=truncated,
        ),
        items=items,
        found_by=found_by,
        conversation_result=place,
        formatter=_ExactTimes(deps.timezone),
        downloaded_at=now,
        timezone=deps.timezone,
    )
    slug = _slug(result.request.query) if result.request.query else "filters"
    local_now = now.astimezone(ZoneInfo(deps.timezone))
    base = f"search-{slug}-{local_now.date().isoformat()}"
    return Response(
        text.encode("utf-8"),
        media_type=_DOWNLOAD_TYPES[fmt],
        headers={
            "Content-Disposition": f'attachment; filename="{base}.{fmt}"',
            "X-Content-Type-Options": "nosniff",
        },
    )


def _slug(name: str) -> str:
    out: list[str] = []
    for ch in name.lower():
        if ch in _SLUG_CHARS:
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    return "".join(out).strip("-")[:60] or "case"


def _clear_stale_downloads(folder: Path) -> None:
    now = time.time()
    for old in folder.glob("*.zip"):
        with contextlib.suppress(OSError):
            if now - old.stat().st_mtime > DOWNLOAD_STALE_SECONDS:
                old.unlink()


def _write_zip(
    deps: AppDeps,
    path: Path,
    *,
    case_name: str,
    text: str,
    files: list[case_store.CaseFile],
    stamp: tuple[int, int, int, int, int, int],
) -> None:
    """The case file, each original from the attachment cache, and a
    `SHA256SUMS.txt` of the bytes as written (a file whose bytes no longer
    match its recorded SHA-256 is flagged)."""
    sums: list[str] = []
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        data = text.encode("utf-8")
        zf.writestr(case_name, data)
        sums.append(f"{hashlib.sha256(data).hexdigest()}  {case_name}")
        for f in files:
            arcname = case_store.file_path_in_zip(f)
            source = cache_file(deps.data_root, f.sha256)
            if source is None:
                sums.append(f"# not in the attachment cache: {arcname}")
                continue
            digest = hashlib.sha256()
            entry = zipfile.ZipInfo(arcname, date_time=stamp)
            with source.open("rb") as handle, zf.open(entry, "w", force_zip64=True) as out:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                    out.write(chunk)
            line = f"{digest.hexdigest()}  {arcname}"
            if digest.hexdigest() != f.sha256:
                line += f"  # recorded SHA-256 {f.sha256} differs"
            sums.append(line)
        zf.writestr("SHA256SUMS.txt", "\n".join(sums) + "\n")


_DOWNLOAD_TYPES = {
    "md": "text/markdown; charset=utf-8",
    "csv": "text/csv; charset=utf-8",
    "json": "application/json",
}


def case_download(request: Request) -> Response:
    """The case as Markdown, CSV or JSON; with `files=1`, a zip that adds
    the original files and their SHA-256 list. Called "download" on the
    page: "export" means the Gemini pipeline in this project."""
    session = _session(request)
    if session is None:
        return _login_redirect(request)
    deps = _deps(request)
    ctx = _page_ctx(request, session, title="Case · Messages")
    case_id = _id_param(request, "case_id")
    fmt = request.query_params.get("fmt", "md")
    if fmt not in case_store.DOWNLOAD_FORMATS:
        return PlainTextResponse("unknown format", status_code=400)
    with_files = request.query_params.get("files") == "1"
    with deps.pool.connection() as pg:
        case = case_store.get_case(pg, case_id) if case_id is not None else None
        if case is None:
            return HTMLResponse(views.error_page(ctx, status=404, message="No such case."), status_code=404)
        items = case_store.case_items(
            pg,
            case.case_id,
            index_unsent=deps.settings.index_unsent,
            show_edit_history=deps.page.details.show_edit_history,
            show_raw_handles=deps.page.details.show_raw_handles,
        )
        coverage = _coverage(deps, pg, case.case_id)
    now = datetime.now(UTC)
    text = case_store.render_download(
        fmt,
        case=case,
        items=items,
        coverage=coverage,
        formatter=_ExactTimes(deps.timezone),
        downloaded_at=now,
        timezone=deps.timezone,
        with_files=with_files,
    )
    slug = _slug(case.name)
    slug = slug.removeprefix("case-") if slug != "case" else ""
    local_now = now.astimezone(ZoneInfo(deps.timezone))
    base = f"case-{slug + '-' if slug else ''}{local_now.date().isoformat()}"
    if not with_files:
        return Response(
            text.encode("utf-8"),
            media_type=_DOWNLOAD_TYPES[fmt],
            headers={
                "Content-Disposition": f'attachment; filename="{base}.{fmt}"',
                "X-Content-Type-Options": "nosniff",
            },
        )
    files = case_store.case_files(items)
    total = sum(f.byte_size or 0 for f in files)
    if total > MAX_DOWNLOAD_FILE_BYTES:
        return HTMLResponse(
            views.error_page(
                ctx,
                status=413,
                message=(
                    f"The case's files total {views.human_size(total)}, more than a download may "
                    f"carry ({views.human_size(MAX_DOWNLOAD_FILE_BYTES)}). Download without the "
                    "files, or take the largest files out of the case."
                ),
            ),
            status_code=413,
        )
    folder = deps.data_root / "search-page" / "downloads"
    ensure_private_dir(folder)
    _clear_stale_downloads(folder)
    path = folder / f"{secrets.token_hex(12)}.zip"
    try:
        _write_zip(
            deps,
            path,
            case_name=f"{base}.{fmt}",
            text=text,
            files=files,
            stamp=local_now.timetuple()[:6],
        )
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return FileResponse(
        path,
        media_type="application/zip",
        filename=f"{base}.zip",
        headers={"X-Content-Type-Options": "nosniff"},
        background=BackgroundTask(path.unlink, missing_ok=True),
    )


async def case_item_api(request: Request) -> Response:
    posted = await _json_post(request)
    if isinstance(posted, Response):
        return posted
    _session_row, payload = posted
    message_key = payload.get("message_key")
    attachment_key = payload.get("attachment_key")
    add = payload.get("add")
    if (
        not isinstance(message_key, str)
        or (attachment_key is not None and not isinstance(attachment_key, str))
        or not isinstance(add, bool)
    ):
        return JSONResponse({"error": "invalid request"}, status_code=400)
    deps = _deps(request)

    def toggle() -> case_store.ToggleOutcome:
        with deps.pool.connection() as pg:
            return case_store.toggle_item(
                pg,
                message_key=message_key,
                attachment_key=attachment_key,
                add=add,
                default_name=_default_case_name(deps.timezone),
            )

    try:
        outcome = await run_in_threadpool(toggle)
    except case_store.CaseError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    return JSONResponse(
        {
            "in_case": outcome.in_case,
            "case_id": outcome.case_id,
            "case_name": outcome.case_name,
            "count": outcome.item_count,
        }
    )


async def case_search_api(request: Request) -> Response:
    posted = await _json_post(request)
    if isinstance(posted, Response):
        return posted
    _session_row, payload = posted
    query = payload.get("q")
    if not isinstance(query, str):
        return JSONResponse({"error": "invalid request"}, status_code=400)
    deps = _deps(request)

    def save() -> tuple[int, str]:
        with deps.pool.connection() as pg:
            return case_store.save_search(
                pg, query_text=query, params=payload, default_name=_default_case_name(deps.timezone)
            )

    try:
        search_id, name = await run_in_threadpool(save)
    except case_store.CaseError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse({"search_id": search_id, "case_name": name})


async def case_review_api(request: Request) -> Response:
    posted = await _json_post(request)
    if isinstance(posted, Response):
        return posted
    _session_row, payload = posted
    search_id = payload.get("search_id")
    thread_key = payload.get("thread_key")
    reviewed = payload.get("reviewed")
    if (
        not isinstance(search_id, int)
        or isinstance(search_id, bool)
        or not isinstance(thread_key, str)
        or not isinstance(reviewed, bool)
    ):
        return JSONResponse({"error": "invalid request"}, status_code=400)
    deps = _deps(request)

    def mark() -> int:
        with deps.pool.connection() as pg:
            return case_store.set_reviewed(pg, search_id=search_id, thread_key=thread_key, reviewed=reviewed)

    try:
        count = await run_in_threadpool(mark)
    except case_store.CaseError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    return JSONResponse({"reviewed": count})


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
        marks = case_store.marks(pg, [m.message_key for m in window.messages])
    matcher = QueryMatcher.for_query(analyze_query(query)) if query.strip() else None
    messages_html = views.messages_with_day_breaks(
        window.messages,
        tz=deps.timezone,
        matcher=matcher,
        group=chat.kind == "group" or chat.is_holding,
        anchor_key=window.anchor_key,
        thread_key=chat.thread_key,
        conversation=chat.title,
        marks=marks,
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
        marks = case_store.marks(pg, [m.message_key for m in window.messages])
    matcher = QueryMatcher.for_query(analyze_query(query)) if query.strip() else None
    body = views.messages_with_day_breaks(
        window.messages,
        tz=deps.timezone,
        matcher=matcher,
        group=chat.kind == "group" or chat.is_holding,
        anchor_key=None,
        thread_key=chat.thread_key,
        conversation=chat.title,
        marks=marks,
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
        Route("/timeline", timeline_view, methods=["GET"]),
        Route("/timeline/page", timeline_more, methods=["GET"]),
        Route("/media", media_view, methods=["GET"]),
        Route("/media/page", media_more, methods=["GET"]),
        Route("/message/{message_key}/details", message_details_view, methods=["GET"]),
        Route("/grade", grade_start, methods=["POST"]),
        Route("/grade/{list_id}", grade_view, methods=["GET"]),
        Route("/api/grade", grade_api, methods=["POST"]),
        Route("/case", cases_view, methods=["GET"]),
        Route("/case", case_create, methods=["POST"]),
        Route("/case/{case_id}", case_view, methods=["GET"]),
        Route("/case/{case_id}/download", case_download, methods=["GET"]),
        Route("/case/{case_id}/activate", _case_action("case_id", _activate), methods=["POST"]),
        Route("/case/{case_id}/rename", _case_action("case_id", _rename), methods=["POST"]),
        Route("/case/{case_id}/notes", _case_action("case_id", _notes), methods=["POST"]),
        Route("/case/{case_id}/delete", _case_action("case_id", _delete), methods=["POST"]),
        Route("/case/item/{item_id}/note", _case_action("item_id", _item_note), methods=["POST"]),
        Route("/case/item/{item_id}/remove", _case_action("item_id", _item_remove), methods=["POST"]),
        Route(
            "/case/search/{search_id}/remove", _case_action("search_id", _search_remove), methods=["POST"]
        ),
        Route("/api/case/item", case_item_api, methods=["POST"]),
        Route("/api/case/search", case_search_api, methods=["POST"]),
        Route("/api/case/review", case_review_api, methods=["POST"]),
        Route("/saved", saved_view, methods=["GET"]),
        Route("/api/saved", saved_api, methods=["POST"]),
        Route("/saved/{search_id}/rename", _case_action("search_id", _saved_rename), methods=["POST"]),
        Route("/saved/{search_id}/remove", _case_action("search_id", _saved_remove), methods=["POST"]),
        Route("/search/download", search_download, methods=["GET"]),
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
