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
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from imsg.search_page.browse import MediaItem
from imsg.search_page.cases import (
    DOWNLOAD_FORMATS,
    SEARCH_PARAMS,
    CaseItem,
    CaseMarks,
    CaseSummary,
    SavedSearch,
    SearchCoverage,
)
from imsg.search_page.details import FILED_BY, MessageDetails, service_name
from imsg.search_page.grading import (
    GRADE_LABELS,
    GRADED_POSITIONS,
    MAX_POSITIONS,
    GradedCandidate,
    GradedList,
    describe_filters,
)
from imsg.search_page.highlight import QueryMatcher, display_text
from imsg.search_page.labels import (
    NOT_RELEVANT_GRADE,
    RELEVANT_GRADE,
    HitLabel,
    LabelledQuery,
)
from imsg.search_page.search import CHANNEL_LABELS, Hit
from imsg.search_page.threads import AttachmentView, ChatView, MessageView

STATIC_VERSION = "7"
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


def fmt_seconds(dt: datetime, tz: str) -> str:
    """To the second, with the zone's abbreviation: `Mon 24 Apr 2023,
    14:28:07 EDT`."""
    return _fmt(dt, tz, "%a %d %b %Y, %H:%M:%S %Z")


def utc_offset(dt: datetime, tz: str) -> str:
    """`UTC\u221204:00` for a time four hours behind UTC."""
    offset = dt.astimezone(ZoneInfo(tz)).utcoffset()
    minutes = int(offset.total_seconds() // 60) if offset is not None else 0
    sign = "+" if minutes >= 0 else "\u2212"
    minutes = abs(minutes)
    return f"UTC{sign}{minutes // 60:02d}:{minutes % 60:02d}"


def citation_line(message: MessageView, *, conversation: str, tz: str) -> str:
    """One line that cites a message exactly: time to the second, sender,
    conversation, text, message ID. What "Copy citation" copies."""
    parts = [fmt_seconds(message.sent_at, tz), message.sender_name, conversation]
    text = " ".join(display_text(message.text).split()) if message.text else ""
    if text:
        parts.append(f"\u201c{text}\u201d")
    names = [a.filename or a.kind for a in message.attachments]
    if names:
        parts.append(("attachment: " if len(names) == 1 else "attachments: ") + ", ".join(names))
    if message.is_edited:
        parts.append("edited")
    if message.is_unsent:
        parts.append("unsent")
    if message.is_deleted:
        parts.append("deleted in Messages")
    parts.append(f"ID {message.message_key}")
    return " \u00b7 ".join(parts)


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


FILTER_KEYS = SEARCH_PARAMS
"""The search's filters as URL parameters: what a saved search keeps."""


def _choice(value: object, allowed: tuple[str, ...]) -> str:
    return value if isinstance(value, str) and value in allowed else allowed[0]


@dataclass(frozen=True, slots=True)
class FormState:
    query: str = ""
    people: str = ""
    date_from: str = ""
    date_to: str = ""
    attachments: str = "any"
    sort: str = "relevance"
    sender: str = ""
    """"Sent by": a person's name, or `me`."""
    direction: str = "any"
    """`sent` (by the owner) or `received`."""
    chat_kind: str = "any"
    """`dm` (one-to-one) or `group`."""
    thread: str = ""
    """Only this conversation (its `thread_key`)."""

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> FormState:
        """From URL parameters, a posted form or a saved search's filters,
        clipped and with unknown choices set back to their defaults."""

        def text(key: str, limit: int) -> str:
            value = values.get(key, "")
            return value[:limit] if isinstance(value, str) else ""

        return cls(
            query=text("q", 2000),
            people=text("people", 500),
            date_from=text("from", 10),
            date_to=text("to", 10),
            attachments=_choice(values.get("att"), ("any", "with", "without")),
            sort=_choice(values.get("sort"), ("relevance", "date", "rerank")),
            sender=text("sender", 200),
            direction=_choice(values.get("dir"), ("any", "sent", "received")),
            chat_kind=_choice(values.get("kind"), ("any", "dm", "group")),
            thread=text("in", 100),
        )

    @property
    def has_filters(self) -> bool:
        return any(k != "sort" for k in self.params() if k != "q")

    def filter_params(self) -> dict[str, str]:
        return {k: v for k, v in self.params().items() if k in FILTER_KEYS}

    def params(self, **overrides: str | int) -> dict[str, str]:
        values: dict[str, str] = {
            "q": self.query,
            "people": self.people,
            "from": self.date_from,
            "to": self.date_to,
            "att": self.attachments,
            "sender": self.sender,
            "dir": self.direction,
            "kind": self.chat_kind,
            "in": self.thread,
            "sort": self.sort,
        }
        values.update({k: str(v) for k, v in overrides.items()})
        return {k: v for k, v in values.items() if v not in ("", "any") or k == "q"}

    def url(self, path: str = "/search", **overrides: str | int) -> str:
        return f"{path}?{urlencode(self.params(**overrides))}"


def _option(value: str, label: str, current: str) -> str:
    selected = " selected" if value == current else ""
    return f'<option value="{esc(value)}"{selected}>{esc(label)}</option>'


def search_form(
    form: FormState,
    *,
    semantic_available: bool,
    csrf_token: str,
    thread_title: str | None = None,
) -> str:
    """`thread_title`: the conversation's name when the search is limited
    to one (`form.thread`)."""
    sorts = [("relevance", "Best match"), ("date", "Newest first")]
    if semantic_available:
        sorts.append(("rerank", "Best match, reranked"))
    only_in = ""
    if form.thread:
        without = FormState.from_mapping({**form.params(), "in": ""}).url()
        only_in = (
            f'<input type="hidden" name="in" value="{esc(form.thread)}">'
            f'<span class="only-in">Only in \u201c{esc(thread_title or "this conversation")}\u201d '
            f'<a href="{esc(without)}" title="Search every conversation">\u00d7</a></span>'
        )
    return (
        '<header class="topbar"><form class="search-form" method="get" action="/search" role="search">'
        '<div class="row main-row">'
        '<a class="home" href="/" title="New search">Messages</a>'
        f'<input class="q" type="search" name="q" value="{esc(form.query)}" '
        'placeholder="Search every message" aria-label="Search" autocomplete="off" '
        f'{"autofocus" if not form.query else ""} maxlength="1000">'
        '<button type="submit">Search</button>'
        "</div>"
        f"{only_in}"
        '<div class="row second-row"><details class="filters"'
        + (" open" if form.has_filters and not (form.thread and len(form.filter_params()) == 1) else "")
        + "><summary>Filters</summary>"
        '<div class="row filter-row">'
        '<label title="Conversations every listed person is in">People in the conversation '
        '<input type="text" name="people" list="people-list" '
        f'value="{esc(form.people)}" placeholder="name, name" autocomplete="off" '
        'class="people-input"></label>'
        '<label title="Who wrote the message: a name, or me">Sent by '
        '<input type="text" name="sender" list="people-list" '
        f'value="{esc(form.sender)}" placeholder="name or me" autocomplete="off" '
        'class="people-input sender-input"></label>'
        '<label>Messages <select name="dir">'
        + _option("any", "Sent and received", form.direction)
        + _option("sent", "Sent by me", form.direction)
        + _option("received", "Received", form.direction)
        + "</select></label>"
        '<label>Conversations <select name="kind">'
        + _option("any", "All", form.chat_kind)
        + _option("dm", "One-to-one", form.chat_kind)
        + _option("group", "Groups", form.chat_kind)
        + "</select></label>"
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
        "</div></details>"
        '<nav class="views" aria-label="Views"><a href="/timeline">Timeline</a>'
        '<a href="/media">Media</a><a href="/saved">Saved</a><a href="/case">Case</a>'
        '<a href="/labels">Labels</a></nav>'
        "</div></form>"
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


def _attachment_html(
    att: AttachmentView,
    *,
    matched: bool,
    message_key: str | None = None,
    marks: CaseMarks | None = None,
) -> str:
    cls = "att matched" if matched else "att"
    add = f" \u00b7 {case_button(message_key, att.attachment_key, marks)}" if message_key else ""
    name = att.filename or {"image": "Image", "video": "Video", "audio": "Audio", "pdf": "PDF"}.get(
        att.kind, "Attachment"
    )
    key = esc(att.attachment_key)
    download = (
        f'<a class="att-download" href="/att/{key}?download=1" title="Download">'
        f"Download{(' · ' + esc(human_size(att.byte_size))) if att.byte_size else ''}</a>{add}"
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
        f" <span class=\"muted\">{esc(human_size(att.byte_size))}</span>{add}</div>"
    )


def case_button(message_key: str, attachment_key: str | None, marks: CaseMarks | None) -> str:
    """"Add to case" for a message, or for one of its files."""
    on = marks is not None and (message_key, attachment_key) in marks
    what = "file " if attachment_key else ""
    label = "In case \u2713" if on else f"Add {what}to case"
    data = f' data-attachment="{esc(attachment_key)}"' if attachment_key else ""
    return (
        f'<button type="button" class="link case-btn{" on" if on else ""}" '
        f'data-message="{esc(message_key)}"{data} aria-pressed="{"true" if on else "false"}" '
        f'data-add-label="Add {what}to case">{label}</button>'
    )


def message_actions(
    message: MessageView, *, conversation: str, tz: str, marks: CaseMarks | None = None
) -> str:
    """The row under a message: add to the case, its Details panel, its
    citation."""
    cite = citation_line(message, conversation=conversation, tz=tz)
    key = esc(message.message_key)
    return (
        '<div class="msg-actions">'
        + case_button(message.message_key, None, marks)
        + f'<button type="button" class="link details-btn" data-url="/message/{key}/details" '
        'aria-expanded="false">Details</button>'
        f'<button type="button" class="link cite-btn" data-cite="{esc(cite)}">Copy citation</button>'
        "</div>"
    )


def _duration(seconds: float) -> str:
    whole = round(seconds)
    if whole < 60:
        return f"{whole} second{'s' if whole != 1 else ''}"
    minutes, rest = divmod(whole, 60)
    if minutes < 60:
        text = f"{minutes} minute{'s' if minutes != 1 else ''}"
        return text + (f" {rest} second{'s' if rest != 1 else ''}" if rest else "")
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours} hour{'s' if hours != 1 else ''}" + (
            f" {minutes} minute{'s' if minutes != 1 else ''}" if minutes else ""
        )
    return f"{hours // 24} days"


def exact_time(dt: datetime, tz: str) -> str:
    """`Mon 24 Apr 2023, 14:28:07 EDT (UTC\u221204:00)`."""
    return f"{fmt_seconds(dt, tz)} ({utc_offset(dt, tz)})"


HIDDEN_BY_SETTING = "hidden by setting"


def details_html(details: MessageDetails, *, tz: str) -> str:
    """The Details panel for one message (an HTML fragment)."""
    rows: list[tuple[str, str]] = []
    rows.append(("Sent", esc(exact_time(details.sent_at, tz))))
    sender = [esc(details.sender_name)]
    if not details.is_from_me:
        if details.raw_handle is not None:
            handle = esc(details.raw_handle)
            if details.raw_handle_service and details.raw_handle_service != details.service:
                handle += f" ({esc(service_name(details.raw_handle_service))})"
            sender.append(f'<span class="handle">{handle}</span>')
        elif details.handles_hidden:
            sender.append(
                f'<span class="muted">number or email {HIDDEN_BY_SETTING} '
                "(search_page.details.show_raw_handles)</span>"
            )
        else:
            sender.append('<span class="muted">no number or email recorded</span>')
    sender.append(esc(service_name(details.service)))
    rows.append(("From", " \u00b7 ".join(sender)))
    if details.is_edited:
        if details.date_edited is not None:
            after = _duration((details.date_edited - details.sent_at).total_seconds())
            edited = (
                f"Yes, last edited {esc(exact_time(details.date_edited, tz))}, "
                f"{esc(after)} after sending"
            )
        else:
            edited = "Yes; the edit time was not recorded"
        if details.versions is None:
            edited += (
                f'<div class="muted">Earlier text {HIDDEN_BY_SETTING} '
                "(search_page.details.show_edit_history)</div>"
            )
        elif details.versions:
            items = "".join(
                "<li>\u201c"
                + esc(v.text)
                + "\u201d"
                + (f' <span class="muted">dated {esc(exact_time(v.edited_at, tz))}</span>' if v.edited_at else "")
                + "</li>"
                for v in details.versions
            )
            edited += f'<div>Earlier text, oldest first:</div><ol class="versions">{items}</ol>'
        else:
            edited += '<div class="muted">No earlier text was recorded.</div>'
    else:
        edited = "No"
    rows.append(("Edited", edited))
    if details.deleted_at is not None:
        rows.append(
            (
                "Deleted",
                f"Deleted in Messages on {esc(exact_time(details.deleted_at, tz))}; "
                "kept from Recently Deleted",
            )
        )
    else:
        rows.append(("Deleted", "No"))
    rows.append(("Unsent", "Yes" if details.is_unsent else "No"))
    kind = "group" if details.chat_kind == "group" else "one-to-one"
    if details.is_holding:
        kind = "unfiled"
    filed = FILED_BY.get(details.chat_evidence, details.chat_evidence)
    rows.append(("Conversation", f"{esc(details.chat_title)} ({esc(kind)}) \u00b7 filed here because {esc(filed)}"))
    if details.sources:
        lines = []
        for source in details.sources:
            what = {
                "live": "the live Messages database",
                "seed": "an older or recovered copy, used only to fill gaps",
            }.get(source.merge_mode or "", "a Messages database")
            when = f", read {esc(exact_time(source.read_at, tz))}" if source.read_at else ""
            lines.append(
                f"<li>{esc(source.source_name)}: {esc(what)}, row {source.source_rowid}{when}</li>"
            )
        rows.append(("Found in", f'<ul class="sources">{"".join(lines)}</ul>'))
    else:
        rows.append(("Found in", '<span class="muted">no source recorded</span>'))
    if details.attachments:
        lines = []
        for att in details.attachments:
            parts = [esc(att.filename or "unnamed file")]
            if att.byte_size is not None:
                parts.append(esc(human_size(att.byte_size)))
            if att.sha256:
                parts.append(f'SHA-256 <code class="sha">{esc(att.sha256)}</code>')
            if att.state != "materialized":
                parts.append(f"file {esc(att.state)}")
            if att.sources:
                parts.append("from " + esc(", ".join(att.sources)))
            lines.append("<li>" + " \u00b7 ".join(parts) + "</li>")
        rows.append(("Attachments", f'<ul class="attachments">{"".join(lines)}</ul>'))
    rows.append(("Message ID", f'<code class="key">{esc(details.message_key)}</code>'))
    rows.append(("Messages GUID", f'<code class="key">{esc(details.source_guid)}</code>'))
    body = "".join(f"<dt>{label}</dt><dd>{value}</dd>" for label, value in rows)
    return (
        '<div class="details" role="region" aria-label="Message details">'
        f'<div class="details-title">Message details</div><dl>{body}</dl></div>'
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
    conversation: str | None = None,
    marks: CaseMarks | None = None,
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
        deleted_title = "Deleted in Messages"
        if message.deleted_at is not None:
            deleted_title += f" on {fmt_datetime(message.deleted_at, tz)}"
        deleted_title += "; kept from Recently Deleted"
        badges.append(f'<span class="badge deleted" title="{esc(deleted_title)}">Deleted</span>')
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
        _attachment_html(
            att,
            matched=att.attachment_id in matched_attachment_ids,
            message_key=message.message_key if conversation is not None else None,
            marks=marks,
        )
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
    actions = (
        message_actions(message, conversation=conversation, tz=tz, marks=marks)
        if conversation is not None
        else ""
    )
    return (
        f'<article class="{" ".join(classes)}" id="m-{esc(message.message_key)}" '
        f'data-key="{esc(message.message_key)}">{meta}{reply}{text_html}{atts}{previews}{reactions}'
        f"{actions}</article>"
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
    conversation: str | None = None,
    marks: CaseMarks | None = None,
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
                conversation=conversation,
                marks=marks,
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
    next_offset: int | None = None
    """Where the next 50 of this conversation's hits start, when the owner
    asked for all of them and more remain."""
    shown_from: int = 0


def _chat_header(chat: ChatView, *, with_participants: bool = True) -> str:
    kind = "Group" if chat.kind == "group" else "Conversation"
    holding = (
        '<span class="badge holding" title="No conversation could be named for these '
        'messages">Unfiled</span>'
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


@dataclass(frozen=True, slots=True)
class ReviewState:
    """The saved search for the shown search, if the owner saved it (to
    the active case, or on its own): which conversations are marked
    reviewed."""

    search_id: int
    reviewed: frozenset[str]
    case_id: int | None = None
    case_name: str | None = None
    name: str = ""


def hit_html(
    view: HitView,
    *,
    chat: ChatView,
    tz: str,
    matcher: QueryMatcher,
    query: str,
    marks: CaseMarks | None = None,
) -> str:
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
            conversation=chat.title,
            marks=marks,
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


HITS_PAGE = 50
"""Hits per request when the owner asks for all of a conversation's hits."""


def _next_hits_button(view: ThreadResultView, form: FormState) -> str:
    if view.next_offset is None:
        return ""
    shown = view.next_offset
    step = min(HITS_PAGE, view.count - shown)
    more_url = form.url("/search/thread", thread=view.chat.thread_key, offset=shown)
    return (
        f'<button type="button" class="more-hits link" data-url="{esc(more_url)}" data-append="1">'
        f"Show {step} more ({shown:,} of {view.count:,} shown)</button>"
    )


def thread_hits_html(
    view: ThreadResultView,
    *,
    tz: str,
    matcher: QueryMatcher,
    form: FormState,
    query: str,
    marks: CaseMarks | None = None,
    review: ReviewState | None = None,
) -> str:
    """The next hits of one conversation, appended by the page script."""
    hits = "".join(
        hit_html(h, chat=view.chat, tz=tz, matcher=matcher, query=query, marks=marks) for h in view.hits
    )
    return hits + _next_hits_button(view, form)


def _review_box(chat: ChatView, review: ReviewState | None) -> str:
    if review is None:
        return ""
    checked = " checked" if chat.thread_key in review.reviewed else ""
    return (
        '<label class="review-label"><input type="checkbox" class="review-box" '
        f'data-search="{review.search_id}" data-thread="{esc(chat.thread_key)}"{checked}> Reviewed</label>'
    )


def thread_result_html(
    view: ThreadResultView,
    *,
    tz: str,
    matcher: QueryMatcher,
    form: FormState,
    query: str,
    marks: CaseMarks | None = None,
    review: ReviewState | None = None,
) -> str:
    hits = "".join(
        hit_html(h, chat=view.chat, tz=tz, matcher=matcher, query=query, marks=marks) for h in view.hits
    )
    more = ""
    remaining = view.count - len(view.hits)
    if view.shown_all:
        more = _next_hits_button(view, form)
    elif remaining > 0:
        more_url = form.url("/search/thread", thread=view.chat.thread_key, offset=0)
        more = (
            f'<button type="button" class="more-hits link" data-url="{esc(more_url)}">'
            f"Show all {view.count:,} hits in this conversation</button>"
        )
    return (
        f'<section class="thread-result" data-thread="{esc(view.chat.thread_key)}">'
        f'<div class="thread-head">{_review_box(view.chat, review)}{_chat_header(view.chat)}'
        f'<div class="thread-meta"><span class="count">{view.count} hit{"s" if view.count != 1 else ""}</span>'
        f' · latest {esc(fmt_date(view.latest_at, tz))} · '
        f'<a href="/thread/{esc(view.chat.thread_key)}">Open</a>'
        + (
            ""
            if form.thread
            else f' · <a class="only-here" href="{esc(form.url(**{"in": view.chat.thread_key}))}">'
            "Search only here</a>"
        )
        + "</div></div>"
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
    marks: CaseMarks | None = None,
    review: ReviewState | None = None,
) -> str:
    body = "".join(
        thread_result_html(
            t, tz=tz, matcher=matcher, form=form, query=form.query, marks=marks, review=review
        )
        for t in threads
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
    labels_link = ' · <a class="labels-link" href="/labels">Labels</a>'
    return (
        f'<div class="status" id="status">{" ".join(parts)} · {semantic}{note}{timing_html}'
        f"{labels_link}{hidden}{capped}</div>"
    )


def grade_form(form: FormState, *, csrf_token: str) -> str:
    """"Grade the top 20": a form, because starting a graded list stores
    the search (nothing is stored until the owner asks)."""
    fields = "".join(
        f'<input type="hidden" name="{esc(k)}" value="{esc(v)}">'
        for k, v in form.params().items()
        if k != "sort"
    )
    return (
        '<form class="grade-form" method="post" action="/grade">'
        f'<input type="hidden" name="csrf_token" value="{esc(csrf_token)}">{fields}'
        '<button type="submit" class="link" title="Grade every one of the best 20 results, '
        'for measuring search quality">Grade the top 20</button></form>'
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
    case_box: str = "",
    thread_title: str | None = None,
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
    if not form.query and not form.has_filters and not error:
        intro = (
            '<section class="intro"><p>Every message, every attachment. Words match whole words; '
            '<code>"quoted text"</code> matches exact text anywhere; emoji work too. Results are '
            "grouped by conversation; semantic matches arrive a moment after the text matches. "
            "Leave the words empty and set a filter to list every message it keeps.</p></section>"
        )
    elif not form.query and status is not None and (form.date_from or form.date_to):
        timeline = BrowseForm(
            date_from=form.date_from, date_to=form.date_to, people=form.people, sender=form.sender
        ).url("/timeline")
        intro = (
            '<p class="muted filters-only">Every message these filters keep, newest first. '
            f'<a href="{esc(timeline)}">The same days on the Timeline</a></p>'
        )
    elif not form.query and status is not None:
        intro = '<p class="muted filters-only">Every message these filters keep, newest first.</p>'
    grading = (
        grade_form(form, csrf_token=ctx.csrf_token)
        if status is not None and status.total_hits > 0 and form.query.strip()
        else ""
    )
    body = (
        f'{error_html}{intro}{status_html_text}<div class="result-tools">{grading}{case_box}</div>'
        f'<div id="results" class="results"{data_attrs}>{results}</div>{empty}'
        '<div id="semantic-banner" class="banner" hidden></div>'
    )
    return layout(
        ctx,
        body,
        body_class="page-search",
        topbar=search_form(
            form,
            semantic_available=ctx.semantic_available,
            csrf_token=ctx.csrf_token,
            thread_title=thread_title,
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
        '<p class="notice">No conversation could be named for these messages. They are kept '
        "together here so they stay searchable.</p>"
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
        f'<div class="participants">With: {esc(participants)}</div>{holding}'
        '<form class="search-here" method="get" action="/search" role="search">'
        f'<input type="hidden" name="in" value="{esc(chat.thread_key)}">'
        f'<input type="search" name="q" value="{esc(query)}" placeholder="Search this conversation" '
        'aria-label="Search this conversation" maxlength="1000">'
        '<button type="submit" class="small">Search here</button></form></header>'
        f'<div class="thread" id="thread" data-thread="{esc(chat.thread_key)}" '
        f'data-group="{"1" if chat.kind == "group" else "0"}"{q_attr}>'
        f'{older}<div class="messages">{messages_html}</div>{newer}</div>'
    )
    topbar = search_form(
        FormState(query=query), semantic_available=ctx.semantic_available, csrf_token=ctx.csrf_token
    )
    return layout(ctx, body, body_class="page-thread", topbar=topbar)


@dataclass(slots=True)
class CandidateView:
    candidate: GradedCandidate
    chat: ChatView | None
    messages: list[MessageView]
    attachment_text: dict[int, str] = field(default_factory=dict)
    """Extracted text of the candidate's attachments, by attachment id."""


def _grade_controls(list_id: int, candidate: GradedCandidate) -> str:
    buttons = []
    for grade in (2, 1, 0):
        on = candidate.grade == grade
        buttons.append(
            f'<button type="button" class="grade-btn g{grade}{" on" if on else ""}" '
            f'data-grade="{grade}" aria-pressed="{"true" if on else "false"}">'
            f"{grade} \u00b7 {esc(GRADE_LABELS[grade])}</button>"
        )
    return (
        f'<div class="grade-controls" data-list="{list_id}" data-anchor="{esc(candidate.anchor_guid)}">'
        + "".join(buttons)
        + "</div>"
    )


def grading_page(
    ctx: PageContext,
    *,
    graded: GradedList,
    views: Sequence[CandidateView],
    positions: int,
    tz: str,
    matcher: QueryMatcher,
) -> str:
    """Every candidate of one search, in random order and without scores,
    each with the three grades."""
    described = describe_filters(graded.filters)
    shown_total = min(positions, len(graded.candidates))
    graded_count = graded.graded(positions)
    notes: list[str] = []
    if described:
        notes.append(
            "This search has filters. Its grades stay with this list for comparing ways of "
            "ranking; they do not count toward the first measured evaluation, which runs "
            "searches without filters."
        )
    semantic = str(graded.ranking.get("semantic", ""))
    if semantic and semantic != "done":
        reason = str(graded.ranking.get("semantic_note") or semantic)
        notes.append(
            f"The meaning search did not run ({reason}), so this list holds word matches only."
        )
    cards: list[str] = []
    for view in views:
        candidate = view.candidate
        if view.chat is not None and view.messages:
            title = esc(view.chat.title)
            start, end = view.messages[0].sent_at, view.messages[-1].sent_at
            when = esc(_span_label(start, end, tz))
            body = "".join(
                message_html(m, tz=tz, matcher=matcher, group=True, thread_key=view.chat.thread_key)
                for m in view.messages
            )
            names = {a.attachment_id: a.filename or "attachment" for m in view.messages for a in m.attachments}
            body += "".join(
                f'<div class="att-snippet"><span class="muted">{esc(names.get(att_id, "attachment"))}:</span> '
                f"{matcher.snippet(text)}</div>"
                for att_id, text in sorted(view.attachment_text.items())
                if matcher.matched_terms(text)
            )
        else:
            title = "Conversation"
            when = ""
            body = f'<pre class="candidate-text">{esc(candidate.segment_text)}</pre>'
        cards.append(
            f'<section class="candidate" data-anchor="{esc(candidate.anchor_guid)}">'
            f'<div class="candidate-head"><span class="place">{candidate.shown_order}</span> '
            f"<strong>{title}</strong> <span class=\"muted\">{when}</span></div>"
            f'<div class="candidate-body">{body}</div>{_grade_controls(graded.list_id, candidate)}'
            "</section>"
        )
    more = ""
    if positions < MAX_POSITIONS and len(graded.candidates) > positions:
        extra = min(len(graded.candidates), MAX_POSITIONS) - positions
        more = (
            f'<p><a href="/grade/{graded.list_id}?more=1">Grade {extra} more '
            f"(places {positions + 1} to {positions + extra})</a></p>"
        )
    notes_html = "".join(f'<p class="notice-plain">{esc(n)}</p>' for n in notes)
    body_html = (
        f'<section class="grading" data-list="{graded.list_id}" data-positions="{positions}">'
        f"<h1>Grade: \u201c{esc(graded.query_text)}\u201d</h1>"
        f'<p class="muted">{esc("Filters: " + described) if described else "No filters."} '
        f'<a href="{esc(FormState(query=graded.query_text).url())}">Back to the search</a></p>'
        "<p>Grade each result for this search: <strong>2</strong> exactly what you wanted, "
        "<strong>1</strong> relevant, <strong>0</strong> not relevant. The results are in random "
        "order and show no scores, so a grade does not depend on where a result stood in the "
        "list.</p>"
        f"{notes_html}"
        f'<p class="grade-progress"><span class="n-graded">{graded_count}</span> of '
        f"{shown_total} graded.</p>"
        + "".join(cards)
        + more
        + "</section>"
    )
    topbar = search_form(
        FormState(query=graded.query_text),
        semantic_available=ctx.semantic_available,
        csrf_token=ctx.csrf_token,
    )
    return layout(ctx, body_html, body_class="page-grading", topbar=topbar)


def _progress_row(label: str, have: int, need: int) -> str:
    met = have >= need
    return (
        f'<li class="{"met" if met else "short"}">{have:,} of the {need:,} {esc(label)}'
        f'{" (met)" if met else ""}</li>'
    )


def labels_page(
    ctx: PageContext,
    *,
    query_count: int,
    judgment_count: int,
    queries_with_relevant: int,
    passed: bool,
    minimums: tuple[int, int, int],
    queries: Sequence[LabelledQuery],
    graded_lists: Sequence[GradedList] = (),
) -> str:
    """Progress toward the first measured evaluation of search quality,
    and every query that has labels."""
    need_queries, need_judgments, need_relevant = minimums
    verdict = (
        "Enough labels for the first measured evaluation."
        if passed
        else "Not enough labels yet for the first measured evaluation."
    )
    rows = "".join(
        "<tr>"
        f'<td><a href="{esc(FormState(query=q.query_text).url())}">{esc(q.query_text)}</a></td>'
        f"<td>{q.relevant:,} relevant</td><td>{q.not_relevant:,} not relevant</td>"
        f"<td class=\"muted\">{esc(q.query_id if not q.query_id.startswith('adhoc:') else '')}</td>"
        "</tr>"
        for q in queries
    )
    table = (
        f'<table class="labels-table"><tbody>{rows}</tbody></table>'
        if rows
        else '<p class="muted">No query has labels yet.</p>'
    )
    list_rows = "".join(
        "<tr>"
        f'<td><a href="/grade/{g.list_id}">{esc(g.query_text)}</a>'
        + (f' <span class="muted">({esc(describe_filters(g.filters))})</span>' if describe_filters(g.filters) else "")
        + "</td>"
        f"<td>{g.graded(GRADED_POSITIONS)} of {min(GRADED_POSITIONS, len(g.candidates))} graded</td>"
        f'<td class="muted">{esc(fmt_date(g.created_at, ctx.timezone))}</td>'
        "</tr>"
        for g in graded_lists
    )
    lists_html = (
        f'<table class="labels-table"><tbody>{list_rows}</tbody></table>'
        if list_rows
        else '<p class="muted">No search has been graded yet. Use "Grade the top 20" on a '
        "results page.</p>"
    )
    body = (
        '<section class="labels"><h1>Labels</h1>'
        "<p>The Relevant and Not relevant buttons on each hit, and the grades given in grading "
        "mode, record whether a result answers the search. Measuring search quality needs a "
        "minimum number of them:</p>"
        "<ul class=\"progress\">"
        + _progress_row("queries with labels", query_count, need_queries)
        + _progress_row("graded results", judgment_count, need_judgments)
        + _progress_row("queries with a relevant result", queries_with_relevant, need_relevant)
        + f"</ul><p><strong>{esc(verdict)}</strong></p>"
        f"<h2>Graded searches</h2>{lists_html}"
        f"<h2>Labelled queries</h2>{table}</section>"
    )
    topbar = search_form(FormState(), semantic_available=ctx.semantic_available, csrf_token=ctx.csrf_token)
    return layout(ctx, body, body_class="page-labels", topbar=topbar)


# --------------------------------------------------------------------------
# browse views: timeline and media
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BrowseForm:
    """The Timeline and Media filters, as the owner typed them."""

    date_from: str = ""
    date_to: str = ""
    people: str = ""
    sender: str = ""
    query: str = ""
    media_type: str = "all"

    def params(self, **overrides: str) -> dict[str, str]:
        values = {
            "from": self.date_from,
            "to": self.date_to,
            "people": self.people,
            "sender": self.sender,
            "q": self.query,
            "type": self.media_type,
        }
        values.update(overrides)
        return {k: v for k, v in values.items() if v and not (k == "type" and v == "all")}

    def url(self, path: str, **overrides: str) -> str:
        params = self.params(**overrides)
        return f"{path}?{urlencode(params)}" if params else path


def _browse_filters(form: BrowseForm, *, action: str, media: bool) -> str:
    words = (
        ""
        if media
        else f'<label>Highlight <input type="search" name="q" value="{esc(form.query)}" '
        'placeholder="words" maxlength="1000"></label>'
    )
    type_field = (
        f'<input type="hidden" name="type" value="{esc(form.media_type)}">' if media and form.media_type != "all" else ""
    )
    return (
        f'<form class="browse-form" method="get" action="{esc(action)}"><div class="row filter-row">'
        f'<label>From <input type="date" name="from" value="{esc(form.date_from)}"></label>'
        f'<label>To <input type="date" name="to" value="{esc(form.date_to)}"></label>'
        f'<label>With <input type="text" name="people" list="people-list" class="people-input" '
        f'value="{esc(form.people)}" placeholder="name, name" autocomplete="off"></label>'
        f'<label>Sent by <input type="text" name="sender" value="{esc(form.sender)}" '
        'placeholder="name, or me" autocomplete="off"></label>'
        f"{words}{type_field}"
        '<button type="submit">Show</button></div></form>'
    )


def timeline_rows_html(
    messages: Sequence[MessageView],
    *,
    chats: dict[int, ChatView],
    tz: str,
    matcher: QueryMatcher | None,
    previous_day: str | None,
    marks: CaseMarks | None = None,
) -> str:
    """Timeline rows with a break at each new day."""
    out: list[str] = []
    day = previous_day
    for message in messages:
        this_day = fmt_date(message.sent_at, tz)
        if this_day != day:
            out.append(f'<div class="day-break" data-day="{esc(this_day)}"><span>{esc(this_day)}</span></div>')
            day = this_day
        chat = chats.get(message.chat_id)
        title = chat.title if chat is not None else "Conversation"
        key = esc(message.message_key)
        href = (
            f"/thread/{esc(chat.thread_key)}?anchor={key}#m-{key}" if chat is not None else "#"
        )
        text = display_text(message.text) if message.text else ""
        body = matcher.highlight(text) if matcher is not None else esc(text)
        body = linkify(body).replace("\n", "<br>")
        badges = ""
        if message.is_deleted:
            badges += '<span class="badge deleted">Deleted</span>'
        if message.is_unsent:
            badges += '<span class="badge">Unsent</span>'
        if message.is_edited:
            badges += '<span class="badge">Edited</span>'
        atts = "".join(
            _attachment_html(att, matched=False, message_key=message.message_key, marks=marks)
            for att in message.attachments
        )
        out.append(
            f'<article class="tl-row{" me" if message.is_from_me else ""}" id="m-{key}" data-key="{key}">'
            f'<time class="tl-when" datetime="{esc(message.sent_at.isoformat())}">'
            f"{esc(_fmt(message.sent_at, tz, '%H:%M:%S'))}</time>"
            f'<div class="tl-main"><div class="tl-head"><a class="tl-chat" href="{href}">{esc(title)}</a>'
            f' \u00b7 <span class="sender">{esc(message.sender_name)}</span>{badges}</div>'
            + (f'<div class="tl-text">{body}</div>' if body else "")
            + atts
            + message_actions(message, conversation=title, tz=tz, marks=marks)
            + "</div></article>"
        )
    return "".join(out)


def timeline_page(
    ctx: PageContext,
    *,
    form: BrowseForm,
    heading: str,
    day_counts: Sequence[tuple[str, str, int]],
    rows_html: str,
    next_url: str | None,
    error: str | None,
) -> str:
    """Every message across conversations in time order, for a day or a
    range. `day_counts` is `(label, url, count)` per day."""
    error_html = f'<p class="error" role="alert">{esc(error)}</p>' if error else ""
    days = "".join(
        f'<li><a href="{esc(url)}">{esc(label)}</a> <span class="muted">{n:,} message{"s" if n != 1 else ""}</span></li>'
        for label, url, n in day_counts
    )
    days_html = f'<ul class="day-counts">{days}</ul>' if len(day_counts) > 1 else ""
    sentinel = (
        f'<div class="sentinel" data-next="{esc(next_url)}"><a href="{esc(next_url)}" class="load-more">Later messages</a></div>'
        if next_url
        else ""
    )
    body = (
        f'<section class="browse"><h1>Timeline</h1>{_browse_filters(form, action="/timeline", media=False)}'
        f"{error_html}<p class=\"browse-heading\">{esc(heading)}</p>{days_html}"
        f'<div id="results" class="results timeline">{rows_html}{sentinel}</div></section>'
    )
    topbar = search_form(FormState(), semantic_available=ctx.semantic_available, csrf_token=ctx.csrf_token)
    return layout(ctx, body, body_class="page-timeline", topbar=topbar)


_TILE_LABELS = {"image": "Photo", "video": "Video", "audio": "Voice note", "pdf": "PDF", "other": "File"}


def media_tiles_html(
    items: Sequence[MediaItem],
    *,
    chats: dict[int, ChatView],
    tz: str,
    marks: CaseMarks | None = None,
) -> str:
    out: list[str] = []
    for item in items:
        att = item.attachment
        chat = chats.get(item.chat_id)
        mkey = esc(item.message_key)
        key = esc(att.attachment_key)
        href = f"/thread/{esc(chat.thread_key)}?anchor={mkey}#m-{mkey}" if chat is not None else "#"
        label = _TILE_LABELS.get(att.kind, "File")
        thumb = ""
        if att.available and att.kind in ("image", "video"):
            thumb = f'<img src="/att/{key}/thumb" alt="" loading="lazy" decoding="async">'
        player = ""
        if att.available and att.kind == "audio":
            player = f'<audio controls preload="none" src="/att/{key}/audio"></audio>'
        missing = "" if att.available else f' <span class="muted">(not available: {esc(att.state)})</span>'
        name = att.filename or label
        download = f' \u00b7 <a href="/att/{key}?download=1">Download</a>' if att.available else ""
        out.append(
            f'<figure class="tile tile-{esc(att.kind)}" data-key="{key}">'
            f'<a class="tile-open" href="{href}" title="Open the conversation at this message">'
            f'<span class="tile-thumb"><span class="tile-kind">{esc(label)}</span>{thumb}</span></a>'
            f"{player}"
            f'<figcaption><span class="tile-when">{esc(fmt_datetime(item.sent_at, tz))}</span>'
            f' \u00b7 {esc(item.sender_name)}<br><span class="tile-name">{esc(name)}</span>'
            f"{missing}{download}<br>{case_button(item.message_key, att.attachment_key, marks)}"
            "</figcaption></figure>"
        )
    return "".join(out)


def media_page(
    ctx: PageContext,
    *,
    form: BrowseForm,
    counts: dict[str, int],
    labels: dict[str, str],
    tiles_html: str,
    next_url: str | None,
    error: str | None,
) -> str:
    error_html = f'<p class="error" role="alert">{esc(error)}</p>' if error else ""
    total = sum(counts.values())
    chips = [
        f'<a class="chip{" on" if form.media_type == "all" else ""}" href="{esc(form.url("/media", type="all"))}">All {total:,}</a>'
    ]
    for kind, label in labels.items():
        chips.append(
            f'<a class="chip{" on" if form.media_type == kind else ""}" '
            f'href="{esc(form.url("/media", type=kind))}">{esc(label)} {counts.get(kind, 0):,}</a>'
        )
    sentinel = (
        f'<div class="sentinel" data-next="{esc(next_url)}"><a href="{esc(next_url)}" class="load-more">More</a></div>'
        if next_url
        else ""
    )
    empty = "" if tiles_html else '<p class="empty">No files match.</p>'
    body = (
        f'<section class="browse"><h1>Media</h1>{_browse_filters(form, action="/media", media=True)}'
        f'{error_html}<nav class="chips" aria-label="Type">{"".join(chips)}</nav>{empty}'
        f'<div id="results" class="results media-grid">{tiles_html}{sentinel}</div></section>'
    )
    topbar = search_form(FormState(), semantic_available=ctx.semantic_available, csrf_token=ctx.csrf_token)
    return layout(ctx, body, body_class="page-media", topbar=topbar)


# --------------------------------------------------------------------------
# evidence cases
# --------------------------------------------------------------------------


def case_box(
    form: FormState,
    *,
    active: CaseSummary | None,
    review: ReviewState | None,
    total_threads: int,
) -> str:
    """Under the result count: save this search (on its own, or to the open
    case), or, once it is saved, where and how many of its conversations
    are marked reviewed; and download every result."""
    download = download_links(form)
    if review is not None:
        if review.case_id is not None:
            where = (
                f'Saved in \u201c<a href="/case/{review.case_id}">{esc(review.case_name or "")}</a>\u201d'
            )
        else:
            label = f" as \u201c{esc(review.name)}\u201d" if review.name else ""
            where = f'<a href="/saved">Saved</a>{label}'
        return (
            f'<div class="case-box">{where} '
            f'\u00b7 reviewed <span class="n-reviewed">{len(review.reviewed)}</span> of '
            f"{total_threads:,} conversation{'s' if total_threads != 1 else ''}{download}</div>"
        )
    params = esc(json.dumps({"q": form.query, **form.filter_params()}, ensure_ascii=False))
    target = f" \u201c{esc(active.name)}\u201d" if active is not None else ""
    return (
        '<div class="case-box">'
        f'<button type="button" class="link save-search-btn" data-endpoint="/api/saved" data-params="{params}">'
        "Save this search</button> \u00b7 "
        f'<button type="button" class="link save-search-btn" data-endpoint="/api/case/search" data-params="{params}">'
        f"Save to case{target}</button>{download}</div>"
    )


def download_links(form: FormState) -> str:
    """"Download results" as Markdown, CSV or JSON: every result of the
    search, not only the page shown."""
    links = " ".join(
        f'<a href="{esc(form.url("/search/download", fmt=fmt))}" download>{label}</a>'
        for fmt, label in (("md", "Markdown"), ("csv", "CSV"), ("json", "JSON"))
    )
    return (
        ' \u00b7 <span class="download-results" title="A download is a copy outside the encrypted '
        f'volume">Download results: {links}</span>'
    )


def _post_form(action: str, csrf_token: str, body: str, *, cls: str = "inline-form") -> str:
    return (
        f'<form class="{esc(cls)}" method="post" action="{esc(action)}">'
        f'<input type="hidden" name="csrf_token" value="{esc(csrf_token)}">{body}</form>'
    )


def cases_page(ctx: PageContext, *, cases: Sequence[CaseSummary], error: str | None) -> str:
    error_html = f'<p class="error" role="alert">{esc(error)}</p>' if error else ""
    rows = "".join(
        "<tr>"
        f'<td><a href="/case/{c.case_id}">{esc(c.name)}</a>'
        + (' <span class="badge on">open</span>' if c.is_active else "")
        + f"</td><td>{c.item_count:,} item{'s' if c.item_count != 1 else ''}</td>"
        f"<td>{c.search_count:,} saved search{'es' if c.search_count != 1 else ''}</td>"
        f'<td class="muted">{esc(fmt_date(c.updated_at, ctx.timezone))}</td></tr>'
        for c in cases
    )
    table = (
        f'<table class="labels-table"><tbody>{rows}</tbody></table>'
        if rows
        else '<p class="muted">No case yet. "Add to case" on any message starts one.</p>'
    )
    create = _post_form(
        "/case",
        ctx.csrf_token,
        '<label>New case <input type="text" name="name" maxlength="200" required '
        'placeholder="a name for what you are looking into"></label> <button type="submit">Create</button>',
    )
    body = (
        f'<section class="cases"><h1>Cases</h1>{error_html}'
        "<p>A case collects messages and files you want to keep together, with notes and the "
        'searches that found them. "Add to case" adds to the open case.</p>'
        f"{table}{create}</section>"
    )
    topbar = search_form(FormState(), semantic_available=ctx.semantic_available, csrf_token=ctx.csrf_token)
    return layout(ctx, body, body_class="page-cases", topbar=topbar)


def _case_item_html(item: CaseItem, n: int, *, ctx: PageContext, tz: str, show_raw_handles: bool) -> str:
    if not item.present:
        body = (
            '<p class="muted">This message is no longer shown: it left the index, or it is unsent '
            f'and unsent messages are hidden by setting. Message ID <code class="key">{esc(item.message_key)}</code></p>'
        )
    else:
        assert item.sent_at is not None
        who = esc(item.sender_name)
        if item.raw_handle:
            who += f' <span class="handle">({esc(item.raw_handle)})</span>'
        kind = f" ({esc(item.conversation_kind)})" if item.conversation_kind else ""
        head = (
            f'<div class="case-item-head"><strong>{esc(exact_time(item.sent_at, tz))}</strong> \u00b7 {who} '
            f"\u00b7 {esc(service_name(item.service))} \u00b7 in "
            f'<a href="/thread/{esc(item.thread_key)}?anchor={esc(item.message_key)}#m-{esc(item.message_key)}">'
            f"{esc(item.conversation)}</a>{kind}</div>"
        )
        text = ""
        if item.text:
            text = f'<div class="case-text">{linkify(esc(display_text(item.text))).replace(chr(10), "<br>")}</div>'
        files = "".join(
            f'<li><a href="/att/{esc(f.attachment_key)}?download=1">{esc(f.filename or "unnamed file")}</a>'
            + (f" \u00b7 {esc(human_size(f.byte_size))}" if f.byte_size is not None else "")
            + (f' \u00b7 SHA-256 <code class="sha">{esc(f.sha256)}</code>' if f.sha256 else "")
            + ("" if f.available else f" \u00b7 file {esc(f.state)}")
            + "</li>"
            for f in item.files
        )
        files_html = f'<ul class="case-files">{files}</ul>' if files else ""
        facts: list[str] = []
        if item.is_edited:
            facts.append(
                "Edited" + (f" {esc(exact_time(item.date_edited, tz))}" if item.date_edited else "")
            )
            for v in item.versions or ():
                facts.append(f"Earlier text: \u201c{esc(v.text)}\u201d")
        if item.deleted_at is not None:
            facts.append(f"Deleted in Messages {esc(exact_time(item.deleted_at, tz))}; kept from Recently Deleted")
        if item.is_unsent:
            facts.append("Unsent")
        if item.sources:
            facts.append(
                "Found in " + "; ".join(f"{esc(s.source_name)} row {s.source_rowid}" for s in item.sources)
            )
        facts.append(f'Message ID <code class="key">{esc(item.message_key)}</code>')
        body = head + text + files_html + "".join(f'<div class="case-fact muted">{f}</div>' for f in facts)
    note = _post_form(
        f"/case/item/{item.item_id}/note",
        ctx.csrf_token,
        f'<label class="note-label">Note <textarea name="note" rows="2" maxlength="5000">{esc(item.note)}</textarea></label>'
        '<button type="submit" class="small">Save note</button>',
        cls="note-form",
    )
    remove = _post_form(
        f"/case/item/{item.item_id}/remove",
        ctx.csrf_token,
        '<button type="submit" class="link danger">Remove from case</button>',
    )
    return (
        f'<li class="case-item" id="item-{item.item_id}"><div class="case-n">{n}.</div>'
        f'<div class="case-body">{body}{note}{remove}</div></li>'
    )


def case_page(
    ctx: PageContext,
    *,
    case: CaseSummary,
    items: Sequence[CaseItem],
    coverage: Sequence[SearchCoverage],
    tz: str,
    show_raw_handles: bool,
    confirm_delete: bool,
    error: str | None,
) -> str:
    error_html = f'<p class="error" role="alert">{esc(error)}</p>' if error else ""
    header = (
        f'<h1>{esc(case.name)}{" <span class=\"badge on\">open</span>" if case.is_active else ""}</h1>'
        f'<p class="muted"><a href="/case">All cases</a> \u00b7 {len(items):,} item{"s" if len(items) != 1 else ""}</p>'
    )
    if not case.is_active:
        header += _post_form(
            f"/case/{case.case_id}/activate",
            ctx.csrf_token,
            '<button type="submit" class="link">Make this the open case</button> '
            '<span class="muted">("Add to case" adds to the open case)</span>',
        )
    rename = _post_form(
        f"/case/{case.case_id}/rename",
        ctx.csrf_token,
        f'<label>Name <input type="text" name="name" value="{esc(case.name)}" maxlength="200" required></label> '
        '<button type="submit" class="small">Rename</button>',
    )
    notes = _post_form(
        f"/case/{case.case_id}/notes",
        ctx.csrf_token,
        f'<textarea name="notes" rows="4" maxlength="20000" aria-label="Case notes">{esc(case.notes)}</textarea>'
        '<button type="submit" class="small">Save notes</button>',
        cls="notes-form",
    )
    item_list = "".join(
        _case_item_html(item, n, ctx=ctx, tz=tz, show_raw_handles=show_raw_handles)
        for n, item in enumerate(items, start=1)
    )
    items_html = (
        f'<ol class="case-items">{item_list}</ol>'
        if item_list
        else '<p class="muted">No items yet. Use "Add to case" on a message or a file.</p>'
    )
    searches: list[str] = []
    for c in coverage:
        s = c.search
        filters = esc(describe_search_filters(s.params))
        url = FormState.from_mapping({**s.params, "q": s.query_text}).url()
        count = (
            f"reviewed {len(s.reviewed):,} of {c.conversations:,} conversation{'s' if c.conversations != 1 else ''}"
            if c.conversations is not None
            else f"{len(s.reviewed):,} conversations reviewed; the search could not run now"
        )
        remove = _post_form(
            f"/case/search/{s.search_id}/remove",
            ctx.csrf_token,
            '<button type="submit" class="link danger">Remove</button>',
        )
        searches.append(
            f'<li><a href="{esc(url)}">{esc(search_title(s))}</a>'
            + (f' <span class="muted">({filters})</span>' if filters else "")
            + f" \u00b7 {count} {remove}</li>"
        )
    searches_html = (
        f'<ul class="case-searches">{"".join(searches)}</ul>'
        if searches
        else '<p class="muted">No saved searches. "Save this search" on a results page adds one.</p>'
    )
    formats = "".join(
        f'<label><input type="radio" name="fmt" value="{f}"{" checked" if f == "md" else ""}> {label}</label>'
        for f, label in zip(DOWNLOAD_FORMATS, ("Markdown", "CSV", "JSON"), strict=True)
    )
    download = (
        f'<form class="download-form" method="get" action="/case/{case.case_id}/download">'
        f"<div>{formats}</div>"
        '<label><input type="checkbox" name="files" value="1"> Add the original files and a '
        "SHA-256 list (zip)</label>"
        '<button type="submit">Download</button>'
        '<p class="muted">A download is a copy outside the encrypted volume. It goes only to this '
        "browser; keep it somewhere safe.</p></form>"
    )
    if confirm_delete:
        delete = _post_form(
            f"/case/{case.case_id}/delete",
            ctx.csrf_token,
            f'<p>Delete \u201c{esc(case.name)}\u201d with its {len(items):,} items, notes and saved '
            'searches? The messages themselves stay in the index.</p><input type="hidden" name="confirm" value="1">'
            f'<button type="submit" class="danger-btn">Delete this case</button> <a href="/case/{case.case_id}">Keep it</a>',
            cls="delete-form",
        )
    else:
        delete = f'<p><a class="danger" href="/case/{case.case_id}?delete=1">Delete this case\u2026</a></p>'
    body = (
        f'<section class="case">{header}{error_html}{rename}'
        f"<h2>Notes</h2>{notes}"
        f"<h2>Items, in the order they were sent</h2>{items_html}"
        f"<h2>Saved searches</h2>{searches_html}"
        f"<h2>Download</h2>{download}{delete}</section>"
    )
    topbar = search_form(FormState(), semantic_available=ctx.semantic_available, csrf_token=ctx.csrf_token)
    return layout(ctx, body, body_class="page-case", topbar=topbar)


# --------------------------------------------------------------------------
# saved searches
# --------------------------------------------------------------------------


def describe_search_filters(params: Mapping[str, str], *, conversation: str | None = None) -> str:
    """A saved search's filters in plain words: `people Alice; sent by
    Bob; from 2023-04-01; to 2023-04-30; with attachments`."""
    parts: list[str] = []
    if params.get("people"):
        parts.append(f"with {params['people']}")
    if params.get("sender"):
        parts.append(f"sent by {params['sender']}")
    direction = params.get("dir")
    if direction == "sent":
        parts.append("sent by me")
    elif direction == "received":
        parts.append("received")
    kind = params.get("kind")
    if kind == "dm":
        parts.append("one-to-one conversations")
    elif kind == "group":
        parts.append("group conversations")
    if params.get("in"):
        parts.append(f"only in \u201c{conversation}\u201d" if conversation else "in one conversation")
    if params.get("from"):
        parts.append(f"from {params['from']}")
    if params.get("to"):
        parts.append(f"to {params['to']}")
    att = params.get("att")
    if att == "with":
        parts.append("with attachments")
    elif att == "without":
        parts.append("without attachments")
    return "; ".join(parts)


def search_title(search: SavedSearch) -> str:
    """The saved search's name, else its words in quotes, else "(filters only)"."""
    if search.name:
        return search.name
    return f"\u201c{search.query_text}\u201d" if search.query_text else "(filters only)"


def saved_page(
    ctx: PageContext,
    *,
    searches: Sequence[SavedSearch],
    conversations: Mapping[str, str],
    tz: str,
    error: str | None,
) -> str:
    """Every saved search: run it, rename it, download its results, or
    remove it. `conversations` names the conversation of each "only in"
    filter, by thread key."""
    error_html = f'<p class="error" role="alert">{esc(error)}</p>' if error else ""
    rows: list[str] = []
    for s in searches:
        form = FormState.from_mapping({**s.params, "q": s.query_text})
        filters = describe_search_filters(s.params, conversation=conversations.get(s.params.get("in", "")))
        words = (
            f"\u201c{esc(s.query_text)}\u201d" if s.query_text else '<span class="muted">no words</span>'
        )
        where = (
            f'case <a href="/case/{s.case_id}">{esc(s.case_name or "")}</a>'
            if s.case_id is not None
            else "on its own"
        )
        rename = _post_form(
            f"/saved/{s.search_id}/rename",
            ctx.csrf_token,
            f'<input type="text" name="name" value="{esc(s.name)}" maxlength="200" '
            'placeholder="Name" aria-label="Name">'
            '<button type="submit" class="small">Rename</button>',
        )
        remove = _post_form(
            f"/saved/{s.search_id}/remove",
            ctx.csrf_token,
            '<button type="submit" class="link danger">Remove</button>',
        )
        rows.append(
            f'<li class="saved-search" id="saved-{s.search_id}">'
            f'<div><a class="run" href="{esc(form.url())}"><strong>{esc(search_title(s))}</strong></a>'
            + (f" \u00b7 {words}" if s.name else "")
            + (f' <span class="muted">({esc(filters)})</span>' if filters else "")
            + "</div>"
            f'<div class="muted">Saved {esc(fmt_date(s.created_at, tz))}, {where} \u00b7 '
            f"{len(s.reviewed):,} conversation{'s' if len(s.reviewed) != 1 else ''} marked reviewed"
            f"{download_links(form)}</div>"
            f'<div class="saved-actions">{rename} {remove}</div></li>'
        )
    body = (
        '<section class="saved"><h1>Saved searches</h1>'
        f"{error_html}"
        + (
            f'<ul class="saved-searches">{"".join(rows)}</ul>'
            if rows
            else '<p class="muted">No saved searches yet. "Save this search" on a results page '
            "keeps one here.</p>"
        )
        + '<p class="muted">A download is a copy outside the encrypted volume. It goes only to '
        "this browser; keep it somewhere safe.</p></section>"
    )
    topbar = search_form(FormState(), semantic_available=ctx.semantic_available, csrf_token=ctx.csrf_token)
    return layout(ctx, body, body_class="page-saved", topbar=topbar)


def error_page(ctx: PageContext, *, status: int, message: str) -> str:
    body = f'<section class="error-page"><h1>{status}</h1><p>{esc(message)}</p><p><a href="/">Search</a></p></section>'
    return layout(ctx, body, body_class="page-error")


__all__ = [
    "BrowseForm",
    "FormState",
    "HitView",
    "PageContext",
    "ReviewState",
    "StatusView",
    "ThreadResultView",
    "case_box",
    "case_button",
    "case_page",
    "cases_page",
    "citation_line",
    "details_html",
    "error_page",
    "esc",
    "fmt_date",
    "fmt_datetime",
    "grade_form",
    "grading_page",
    "human_size",
    "labels_page",
    "linkify",
    "login_page",
    "media_page",
    "media_tiles_html",
    "message_html",
    "messages_with_day_breaks",
    "results_fragment",
    "search_page",
    "status_html",
    "thread_page",
    "thread_result_html",
    "timeline_page",
    "timeline_rows_html",
]
