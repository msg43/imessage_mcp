"""Server-rendered HTML for the search page.

Every dynamic value goes through `esc` (or through
`imsg.search_page.highlight`, which escapes before it marks), and the page
carries no inline script or style: the Content Security Policy allows
only this origin's `/static/app.js` and `/static/app.css`. Message and
attachment text is untrusted content — somebody else typed it — so it is
only ever text here, never markup; links in it are rendered as plain
anchors with `rel="noopener noreferrer"`.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from imsg.search_page.highlight import QueryMatcher, display_text
from imsg.search_page.labels import NOT_RELEVANT_GRADE, RELEVANT_GRADE, HitLabel, LabelCounts
from imsg.search_page.search import CHANNEL_LABELS, Hit
from imsg.search_page.threads import AttachmentView, ChatView, MessageView

STATIC_VERSION = "1"
_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def human_size(n: int | None) -> str:
    if n is None:
        return ""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"


@dataclass(frozen=True, slots=True)
class PageContext:
    csrf_token: str
    timezone: str
    title: str = "Messages"
    semantic_available: bool = False


def _fmt(dt: datetime, tz: str, pattern: str) -> str:
    return dt.astimezone(ZoneInfo(tz)).strftime(pattern)


def fmt_datetime(dt: datetime, tz: str) -> str:
    return _fmt(dt, tz, "%a %d %b %Y, %H:%M")


def fmt_date(dt: datetime, tz: str) -> str:
    return _fmt(dt, tz, "%a %d %b %Y")


def fmt_time(dt: datetime, tz: str) -> str:
    return _fmt(dt, tz, "%H:%M")


def layout(ctx: PageContext, body: str, *, body_class: str = "", topbar: str = "") -> str:
    return (
        "<!doctype html>\n<html lang=\"en\"><head>"
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="light dark">'
        '<meta name="referrer" content="same-origin">'
        f'<meta name="csrf-token" content="{esc(ctx.csrf_token)}">'
        f"<title>{esc(ctx.title)}</title>"
        f'<link rel="stylesheet" href="/static/app.css?v={STATIC_VERSION}">'
        f'<script src="/static/app.js?v={STATIC_VERSION}" defer></script>'
        f'</head><body class="{esc(body_class)}">{topbar}<main id="main">{body}</main></body></html>'
    )


# --------------------------------------------------------------------------
# login
# --------------------------------------------------------------------------


def login_page(*, login_token: str, next_url: str, error: str | None) -> str:
    ctx = PageContext(csrf_token="", timezone="UTC", title="Sign in · Messages")
    error_html = f'<p class="error" role="alert">{esc(error)}</p>' if error else ""
    body = (
        '<section class="login">'
        "<h1>Message search</h1>"
        f"{error_html}"
        '<form method="post" action="/login" class="login-form">'
        f'<input type="hidden" name="login_token" value="{esc(login_token)}">'
        f'<input type="hidden" name="next" value="{esc(next_url)}">'
        '<label for="password">Password</label>'
        '<input id="password" name="password" type="password" autocomplete="current-password" '
        'required autofocus>'
        '<button type="submit">Sign in</button>'
        "</form></section>"
    )
    return layout(ctx, body, body_class="page-login")


# --------------------------------------------------------------------------
# search form
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FormState:
    query: str = ""
    people: str = ""
    date_from: str = ""
    date_to: str = ""
    attachments: str = "any"
    sort: str = "relevance"

    def params(self, **overrides: str | int) -> dict[str, str]:
        values: dict[str, str] = {
            "q": self.query,
            "people": self.people,
            "from": self.date_from,
            "to": self.date_to,
            "att": self.attachments,
            "sort": self.sort,
        }
        values.update({k: str(v) for k, v in overrides.items()})
        return {k: v for k, v in values.items() if v not in ("", "any") or k == "q"}

    def url(self, path: str = "/search", **overrides: str | int) -> str:
        return f"{path}?{urlencode(self.params(**overrides))}"


def _option(value: str, label: str, current: str) -> str:
    selected = " selected" if value == current else ""
    return f'<option value="{esc(value)}"{selected}>{esc(label)}</option>'


def search_form(form: FormState, *, semantic_available: bool, csrf_token: str) -> str:
    sorts = [("relevance", "Best match"), ("date", "Newest first")]
    if semantic_available:
        sorts.append(("rerank", "Best match, reranked"))
    return (
        '<header class="topbar"><form class="search-form" method="get" action="/search" role="search">'
        '<div class="row main-row">'
        '<a class="home" href="/" title="New search">Messages</a>'
        f'<input class="q" type="search" name="q" value="{esc(form.query)}" '
        'placeholder="Search every message" aria-label="Search" autocomplete="off" '
        f'{"autofocus" if not form.query else ""} required maxlength="1000">'
        '<button type="submit">Search</button>'
        "</div>"
        '<details class="filters"'
        + (" open" if (form.people or form.date_from or form.date_to or form.attachments != "any") else "")
        + "><summary>Filters</summary>"
        '<div class="row filter-row">'
        '<label>People <input type="text" name="people" list="people-list" '
        f'value="{esc(form.people)}" placeholder="name, name" autocomplete="off" '
        'class="people-input"></label>'
        '<datalist id="people-list"></datalist>'
        f'<label>From <input type="date" name="from" value="{esc(form.date_from)}"></label>'
        f'<label>To <input type="date" name="to" value="{esc(form.date_to)}"></label>'
        '<label>Attachments <select name="att">'
        + _option("any", "Any", form.attachments)
        + _option("with", "With attachments", form.attachments)
        + _option("without", "Without", form.attachments)
        + "</select></label>"
        '<label>Order <select name="sort">'
        + "".join(_option(v, label, form.sort) for v, label in sorts)
        + "</select></label>"
        "</div></details></form>"
        '<form class="logout" method="post" action="/logout">'
        f'<input type="hidden" name="csrf_token" value="{esc(csrf_token)}">'
        '<button type="submit" class="link">Sign out</button></form>'
        "</header>"
    )


# --------------------------------------------------------------------------
# messages, attachments
# --------------------------------------------------------------------------


def linkify(escaped_html: str) -> str:
    """Turn plain http(s) URLs in already-escaped text into anchors."""

    def repl(match: re.Match[str]) -> str:
        url = match.group(0)
        trail = ""
        while url and url[-1] in ".,);:!?":
            trail = url[-1] + trail
            url = url[:-1]
        href = html.unescape(url)
        if not href.lower().startswith(("http://", "https://")):
            return match.group(0)
        return (
            f'<a href="{esc(href)}" rel="noopener noreferrer nofollow" target="_blank">{url}</a>{trail}'
        )

    return _URL_RE.sub(repl, escaped_html)


def _attachment_html(att: AttachmentView, *, matched: bool) -> str:
    cls = "att matched" if matched else "att"
    name = att.filename or {"image": "Image", "video": "Video", "audio": "Audio", "pdf": "PDF"}.get(
        att.kind, "Attachment"
    )
    key = esc(att.attachment_key)
    download = (
        f'<a class="att-download" href="/att/{key}?download=1" title="Download">'
        f"Download{(' · ' + esc(human_size(att.byte_size))) if att.byte_size else ''}</a>"
    )
    if not att.available:
        return (
            f'<div class="{cls} att-missing">{esc(name)} '
            f'<span class="muted">(not available: {esc(att.state)})</span></div>'
        )
    caption = att.caption or ""
    if att.kind == "image":
        alt = caption or name
        return (
            f'<figure class="{cls} att-image">'
            f'<a href="/att/{key}/view" target="_blank" rel="noopener">'
            f'<img src="/att/{key}/thumb" alt="{esc(alt)}" loading="lazy" decoding="async"></a>'
            f"<figcaption>{download}</figcaption></figure>"
        )
    if att.kind == "video":
        return (
            f'<figure class="{cls} att-video">'
            f'<video controls preload="none" poster="/att/{key}/poster" src="/att/{key}"></video>'
            f"<figcaption>{esc(name)} · {download}</figcaption></figure>"
        )
    if att.kind == "audio":
        transcript = (
            f"<details><summary>Transcript</summary><p>{esc(att.transcript)}</p></details>"
            if att.transcript
            else ""
        )
        return (
            f'<div class="{cls} att-audio">'
            f'<audio controls preload="none" src="/att/{key}/audio"></audio>'
            f"<div>{esc(name)} · {download}</div>{transcript}</div>"
        )
    if att.kind == "pdf":
        return (
            f'<div class="{cls} att-pdf"><span class="att-icon">PDF</span> {esc(name)} · '
            f'<a href="/att/{key}" target="_blank" rel="noopener">Open</a> · '
            f'<button type="button" class="link pdf-toggle" data-src="/att/{key}">Show here</button>'
            f" · {download}</div>"
        )
    return (
        f'<div class="{cls} att-file"><span class="att-icon">FILE</span> '
        f'<a href="/att/{key}?download=1">{esc(name)}</a>'
        f" <span class=\"muted\">{esc(human_size(att.byte_size))}</span></div>"
    )


def message_html(
    message: MessageView,
    *,
    tz: str,
    matcher: QueryMatcher | None,
    group: bool,
    matched_attachment_ids: frozenset[int] = frozenset(),
    anchor: bool = False,
    thread_key: str | None = None,
    show_date: bool = False,
) -> str:
    classes = ["msg", "me" if message.is_from_me else "them"]
    if message.is_deleted:
        classes.append("deleted")
    if anchor:
        classes.append("anchor")
    text = display_text(message.text) if message.text else ""
    body = matcher.highlight(text) if matcher is not None else esc(text)
    body = linkify(body).replace("\n", "<br>")
    badges: list[str] = []
    if message.is_deleted:
        badges.append('<span class="badge deleted" title="In Recently Deleted (D13)">Deleted</span>')
    if message.is_unsent:
        badges.append('<span class="badge">Unsent</span>')
    if message.is_edited:
        badges.append('<span class="badge">Edited</span>')
    reply = ""
    if message.reply_to_key and thread_key:
        reply = (
            f'<a class="reply-to" href="/thread/{esc(thread_key)}?anchor={esc(message.reply_to_key)}'
            f'#m-{esc(message.reply_to_key)}">Reply to: {esc(display_text(message.reply_to_text or "")[:80])}</a>'
        )
    atts = "".join(
        _attachment_html(att, matched=att.attachment_id in matched_attachment_ids)
        for att in message.attachments
    )
    previews = "".join(
        f'<div class="link-preview"><span class="site">{esc(site or "")}</span> '
        f'<a href="{esc(url)}" rel="noopener noreferrer nofollow" target="_blank">{esc(title or url)}</a></div>'
        for url, title, site in message.link_previews
        if url.lower().startswith(("http://", "https://"))
    )
    reactions = ""
    if message.reactions:
        reactions = '<div class="reactions">' + " ".join(
            f'<span class="reaction" title="{esc(who)}">{esc(symbol)} {esc(who)}</span>'
            for symbol, who in message.reactions
        ) + "</div>"
    sender = "" if message.is_from_me and not group else f'<span class="sender">{esc(message.sender_name)}</span>'
    when = fmt_datetime(message.sent_at, tz) if show_date else fmt_time(message.sent_at, tz)
    meta = (
        f'<div class="meta">{sender}<time datetime="{esc(message.sent_at.isoformat())}" '
        f'title="{esc(fmt_datetime(message.sent_at, tz))}">{esc(when)}</time>{"".join(badges)}</div>'
    )
    text_html = f'<div class="text">{body}</div>' if body else ""
    return (
        f'<article class="{" ".join(classes)}" id="m-{esc(message.message_key)}" '
        f'data-key="{esc(message.message_key)}">{meta}{reply}{text_html}{atts}{previews}{reactions}</article>'
    )


def messages_with_day_breaks(
    messages: Sequence[MessageView],
    *,
    tz: str,
    matcher: QueryMatcher | None,
    group: bool,
    anchor_key: str | None,
    thread_key: str,
    previous_day: str | None = None,
) -> str:
    out: list[str] = []
    day = previous_day
    for message in messages:
        this_day = fmt_date(message.sent_at, tz)
        if this_day != day:
            out.append(f'<div class="day-break" data-day="{esc(this_day)}"><span>{esc(this_day)}</span></div>')
            day = this_day
        out.append(
            message_html(
                message,
                tz=tz,
                matcher=matcher,
                group=group,
                anchor=(message.message_key == anchor_key),
                thread_key=thread_key,
            )
        )
    return "".join(out)


# --------------------------------------------------------------------------
# search results
# --------------------------------------------------------------------------


@dataclass(slots=True)
class HitView:
    hit: Hit
    kind: str
    label_key: str
    messages: list[MessageView]
    anchor_key: str | None
    snippets: list[tuple[str, str]] = field(default_factory=list)
    label: HitLabel | None = None
    span: tuple[datetime, datetime] | None = None
    """When the matching messages shown were sent: the hit's label."""


@dataclass(slots=True)
class ThreadResultView:
    chat: ChatView
    count: int
    hits: list[HitView]
    latest_at: datetime
    shown_all: bool


def _chat_header(chat: ChatView, *, with_participants: bool = True) -> str:
    kind = "Group" if chat.kind == "group" else "Conversation"
    holding = (
        '<span class="badge holding" title="A holding chat: messages that no real conversation '
        'could be named for (D13)">Unfiled</span>'
        if chat.is_holding
        else ""
    )
    people = ""
    if with_participants and chat.kind == "group" and chat.participants:
        people = f'<div class="participants">{esc(", ".join(chat.participants))}</div>'
    return (
        f'<div class="chat-title"><span class="kind">{esc(kind)}</span> '
        f"<strong>{esc(chat.title)}</strong> {holding}</div>{people}"
    )


def _channel_badges(hit: Hit) -> str:
    badges: list[str] = []
    for channel in hit.channels:
        label = CHANNEL_LABELS.get(channel, channel)
        similarity = hit.similarity.get(channel)
        if similarity is not None:
            label = f"{label} {similarity:.2f}"
        badges.append(f'<span class="chan chan-{esc(channel)}">{esc(label)}</span>')
    if hit.rerank_score is not None:
        badges.append(f'<span class="chan chan-rerank">reranked {hit.rerank_score:.2f}</span>')
    return "".join(badges)


def _label_controls(view: HitView) -> str:
    grade = view.label.grade if view.label is not None else None
    source = view.label.source if view.label is not None else ""
    rel_on = grade is not None and grade >= 1
    not_on = grade == NOT_RELEVANT_GRADE
    title_extra = f" (set by {source})" if source and source != "mark_relevant" else ""
    return (
        f'<div class="label-controls" data-kind="{esc(view.kind)}" data-key="{esc(view.label_key)}">'
        f'<button type="button" class="label-btn rel{" on" if rel_on else ""}" '
        f'data-grade="{RELEVANT_GRADE}" aria-pressed="{"true" if rel_on else "false"}" '
        f'title="Relevant to this query{esc(title_extra)}">Relevant</button>'
        f'<button type="button" class="label-btn notrel{" on" if not_on else ""}" '
        f'data-grade="{NOT_RELEVANT_GRADE}" aria-pressed="{"true" if not_on else "false"}" '
        f'title="Not relevant to this query{esc(title_extra)}">Not relevant</button></div>'
    )


def _span_label(start: datetime, end: datetime, tz: str) -> str:
    if start == end:
        return fmt_datetime(start, tz)
    if fmt_date(start, tz) == fmt_date(end, tz):
        return f"{fmt_datetime(start, tz)}\u2013{fmt_time(end, tz)}"
    return f"{fmt_datetime(start, tz)} \u2013 {fmt_datetime(end, tz)}"


def hit_html(view: HitView, *, chat: ChatView, tz: str, matcher: QueryMatcher, query: str) -> str:
    hit = view.hit
    start, end = view.span if view.span is not None else (hit.at, hit.ended_at)
    when = _span_label(start, end, tz)
    matched = frozenset(att_id for _msg, att_id in hit.matched_attachments)
    messages = "".join(
        message_html(
            m,
            tz=tz,
            matcher=matcher,
            group=True,
            matched_attachment_ids=matched,
            thread_key=chat.thread_key,
        )
        for m in view.messages
    )
    snippets = "".join(
        f'<div class="att-snippet"><span class="muted">{esc(name)}:</span> {snippet}</div>'
        for name, snippet in view.snippets
    )
    open_href = f"/thread/{esc(chat.thread_key)}?" + urlencode(
        {k: v for k, v in (("anchor", view.anchor_key or ""), ("q", query)) if v}
    )
    if view.anchor_key:
        open_href += f"#m-{esc(view.anchor_key)}"
    return (
        f'<section class="hit" data-hit="{esc(hit.key)}">'
        f'<div class="hit-head"><a class="when" href="{open_href}">{esc(when)}</a>'
        f'<span class="chans">{_channel_badges(hit)}</span></div>'
        f'<div class="hit-body">{messages}{snippets}</div>'
        f'<div class="hit-foot"><a class="open" href="{open_href}">Open conversation</a>'
        f"{_label_controls(view)}</div></section>"
    )


def thread_result_html(
    view: ThreadResultView, *, tz: str, matcher: QueryMatcher, form: FormState, query: str
) -> str:
    hits = "".join(hit_html(h, chat=view.chat, tz=tz, matcher=matcher, query=query) for h in view.hits)
    more = ""
    remaining = view.count - len(view.hits)
    if remaining > 0 and not view.shown_all:
        more_url = form.url("/search/thread", thread=view.chat.thread_key)
        more = (
            f'<button type="button" class="more-hits link" data-url="{esc(more_url)}">'
            f"Show all {view.count} hits in this conversation</button>"
        )
    return (
        f'<section class="thread-result" data-thread="{esc(view.chat.thread_key)}">'
        f'<div class="thread-head">{_chat_header(view.chat)}'
        f'<div class="thread-meta"><span class="count">{view.count} hit{"s" if view.count != 1 else ""}</span>'
        f' · latest {esc(fmt_date(view.latest_at, tz))} · '
        f'<a href="/thread/{esc(view.chat.thread_key)}">Open</a></div></div>'
        f'<div class="hits">{hits}</div>{more}</section>'
    )


def results_fragment(
    threads: Iterable[ThreadResultView],
    *,
    tz: str,
    matcher: QueryMatcher,
    form: FormState,
    next_page_url: str | None,
    next_full_url: str | None = None,
) -> str:
    body = "".join(
        thread_result_html(t, tz=tz, matcher=matcher, form=form, query=form.query) for t in threads
    )
    if next_page_url:
        body += (
            f'<div class="sentinel" data-next="{esc(next_page_url)}">'
            f'<a href="{esc(next_full_url or next_page_url)}" class="load-more">More conversations</a></div>'
        )
    return body


@dataclass(frozen=True, slots=True)
class StatusView:
    total_hits: int
    total_threads: int
    counts: dict[str, int]
    capped: frozenset[str]
    semantic_state: str
    semantic_note: str | None
    timings_ms: dict[str, float]
    label_counts: LabelCounts
    baseline_line: str
    hidden_non_content: int = 0


def status_html(status: StatusView) -> str:
    parts = [
        f'<strong class="total">{status.total_hits:,} hit{"s" if status.total_hits != 1 else ""}</strong>'
        f" in {status.total_threads:,} conversation{'s' if status.total_threads != 1 else ''}"
    ]
    channel_bits = [
        f"{esc(CHANNEL_LABELS.get(ch, ch))} {n:,}" for ch, n in status.counts.items() if n
    ]
    if channel_bits:
        parts.append(f'<span class="channels">({", ".join(channel_bits)})</span>')
    semantic = {
        "pending": '<span class="semantic pending">semantic search running…</span>',
        "done": '<span class="semantic done">semantic search done</span>',
        "unavailable": '<span class="semantic unavailable">semantic search unavailable</span>',
        "disabled": '<span class="semantic disabled">semantic search off</span>',
    }.get(status.semantic_state, "")
    note = f' <span class="muted">({esc(status.semantic_note)})</span>' if status.semantic_note else ""
    hidden = ""
    if status.hidden_non_content:
        n = status.hidden_non_content
        hidden = (
            f'<p class="hidden-note muted">Not shown: {n:,} match{"es" if n != 1 else ""} '
            f"where the words were only in names, dates, times or labels, not in anything "
            f"anyone wrote.</p>"
        )
    capped = ""
    if status.capped:
        capped = (
            '<p class="warning">Some channels reached their safety cap ('
            + esc(", ".join(CHANNEL_LABELS.get(c, c) for c in sorted(status.capped)))
            + "); narrow the query or add a filter to see every hit.</p>"
        )
    timing = status.timings_ms.get("fulltext_total")
    timing_html = f' · <span class="muted timing">text search {timing:.0f} ms</span>' if timing else ""
    labels = status.label_counts
    label_html = (
        f'<span class="label-count" data-total="{labels.total}">labelled for this query: '
        f'<span class="n-total">{labels.total}</span> (<span class="n-rel">{labels.relevant}</span> relevant, '
        f'<span class="n-notrel">{labels.not_relevant}</span> not)</span>'
    )
    return (
        f'<div class="status" id="status">{" ".join(parts)} · {semantic}{note}{timing_html}'
        f'<div class="status-2">{label_html} · <span class="muted baseline">{esc(status.baseline_line)}</span></div>'
        f"{hidden}{capped}</div>"
    )


def search_page(
    ctx: PageContext,
    *,
    form: FormState,
    status: StatusView | None,
    results: str,
    error: str | None,
    semantic_url: str | None,
    search_key: str | None,
) -> str:
    error_html = f'<p class="error" role="alert">{esc(error)}</p>' if error else ""
    data_attrs = ""
    if semantic_url:
        data_attrs += f' data-semantic-url="{esc(semantic_url)}"'
    if search_key:
        data_attrs += f' data-query="{esc(form.query)}"'
    status_html_text = status_html(status) if status is not None else ""
    empty = ""
    if status is not None and status.total_hits == 0 and status.semantic_state != "pending":
        empty = '<p class="empty">No messages match.</p>'
    intro = ""
    if not form.query and not error:
        intro = (
            '<section class="intro"><p>Every message, every attachment. Words match whole words; '
            '<code>"quoted text"</code> matches exact text anywhere; emoji work too. Results are '
            "grouped by conversation; semantic matches arrive a moment after the text matches.</p></section>"
        )
    body = (
        f"{error_html}{intro}{status_html_text}"
        f'<div id="results" class="results"{data_attrs}>{results}</div>{empty}'
        '<div id="semantic-banner" class="banner" hidden></div>'
    )
    return layout(
        ctx,
        body,
        body_class="page-search",
        topbar=search_form(
            form, semantic_available=ctx.semantic_available, csrf_token=ctx.csrf_token
        ),
    )


# --------------------------------------------------------------------------
# thread view
# --------------------------------------------------------------------------


def thread_page(
    ctx: PageContext,
    *,
    chat: ChatView,
    messages_html: str,
    has_older: bool,
    has_newer: bool,
    first_key: str | None,
    last_key: str | None,
    query: str,
    back_url: str | None,
) -> str:
    participants = ", ".join(chat.participants) if chat.participants else "only you"
    holding = (
        '<p class="notice">This is a holding chat: messages whose real conversation could not be '
        "identified, filed here so they stay searchable (D13).</p>"
        if chat.is_holding
        else ""
    )
    back = f'<a class="back" href="{esc(back_url)}">← Results</a>' if back_url else ""
    q_attr = f' data-q="{esc(query)}"' if query else ""
    older = (
        f'<div class="sentinel top" data-cursor="{esc(first_key)}" data-dir="older">'
        '<button type="button" class="link load-older">Load earlier messages</button></div>'
        if has_older and first_key
        else '<div class="edge">Start of conversation</div>'
    )
    newer = (
        f'<div class="sentinel bottom" data-cursor="{esc(last_key)}" data-dir="newer">'
        '<button type="button" class="link load-newer">Load later messages</button></div>'
        if has_newer and last_key
        else '<div class="edge">Latest message</div>'
    )
    body = (
        f'<header class="thread-header">{back}{_chat_header(chat, with_participants=False)}'
        f'<div class="participants">With: {esc(participants)}</div>{holding}</header>'
        f'<div class="thread" id="thread" data-thread="{esc(chat.thread_key)}" '
        f'data-group="{"1" if chat.kind == "group" else "0"}"{q_attr}>'
        f'{older}<div class="messages">{messages_html}</div>{newer}</div>'
    )
    topbar = search_form(
        FormState(query=query), semantic_available=ctx.semantic_available, csrf_token=ctx.csrf_token
    )
    return layout(ctx, body, body_class="page-thread", topbar=topbar)


def error_page(ctx: PageContext, *, status: int, message: str) -> str:
    body = f'<section class="error-page"><h1>{status}</h1><p>{esc(message)}</p><p><a href="/">Search</a></p></section>'
    return layout(ctx, body, body_class="page-error")


__all__ = [
    "FormState",
    "HitView",
    "PageContext",
    "StatusView",
    "ThreadResultView",
    "error_page",
    "esc",
    "fmt_date",
    "fmt_datetime",
    "human_size",
    "linkify",
    "login_page",
    "message_html",
    "messages_with_day_breaks",
    "results_fragment",
    "search_page",
    "status_html",
    "thread_page",
    "thread_result_html",
]
