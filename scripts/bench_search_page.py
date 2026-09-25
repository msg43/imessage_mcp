#!/usr/bin/env python3
"""Build a synthetic corpus in a scratch database and time the search
page against it; or build a small fictional demo corpus to look at.

Everything here is synthetic: invented words, invented names, random
vectors with topic structure. It never touches a real index. Point it at
a scratch Postgres 17 with pgvector (never the production cluster):

    export IMSG_BENCH_DSN=postgresql://postgres@127.0.0.1:55491/postgres
    python scripts/bench_search_page.py build --root /tmp/bench-root      # ~production scale
    python scripts/bench_search_page.py measure --root /tmp/bench-root
    python scripts/bench_search_page.py demo --root /tmp/demo-root        # small, with media
    python scripts/bench_search_page.py serve --root /tmp/demo-root --db imsg_search_page_demo

`build` defaults to the production corpus's shape as recorded on
2026-09-17 (673,113 messages, 123,267 segments, 100,926 attachments,
63,977 image vectors, 10,796 people): 675,000 messages in 8,000 chats,
split into segments of about five messages, 100,000 attachments, 60,000
attachment text chunks, 2,048-dimension text vectors and 1,280-dimension
image vectors drawn around 200 topic centres so that a query near one
centre has a realistic number of neighbours above the similarity floor.
Word frequencies follow a Zipf curve over a synthetic vocabulary, so there
are rare, medium and near-stopword queries to time.

`measure` times, per query: the full-text first paint (`run_fulltext`),
the whole first page over HTTP (search, grouping, the page's messages and
attachments, HTML), the next page, and the semantic threshold scans with a
topic-centred query vector. Warm numbers are medians and 95th percentiles
over repeated runs with the result cache cleared each time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import apsw
import psycopg

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from imsg.db.migrations import PostgresMigrationRunner  # noqa: E402
from imsg.embed.fts.schema import create_schema  # noqa: E402
from imsg.embed.vector_codec import vector_literal  # noqa: E402
from imsg.keys import attachment_key, message_key, thread_key  # noqa: E402
from imsg.textnorm import normalize_text  # noqa: E402

BENCH_DB = "imsg_search_page_bench"
DEMO_DB = "imsg_search_page_demo"
TOPICS = 200
TEXT_DIM = 2048
IMAGE_DIM = 1280
EPOCH = datetime(2010, 1, 1, tzinfo=UTC)
SPAN_DAYS = 16 * 365


def admin_dsn() -> str:
    value = os.environ.get("IMSG_BENCH_DSN")
    if not value:
        sys.exit("set IMSG_BENCH_DSN to a scratch Postgres (never the production cluster)")
    return value


def db_dsn(name: str) -> str:
    base = admin_dsn()
    prefix, _, _ = base.rpartition("/")
    return f"{prefix}/{name}"


def recreate(name: str) -> psycopg.Connection:
    with psycopg.connect(admin_dsn(), autocommit=True) as admin:
        admin.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")
        admin.execute(f"CREATE DATABASE {name}")
    conn = psycopg.connect(db_dsn(name), autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    PostgresMigrationRunner(conn, REPO / "migrations").apply_pending()
    return conn


# --------------------------------------------------------------------------
# synthetic words and names
# --------------------------------------------------------------------------

COMMON = ["the", "to", "and", "you", "a", "i", "it", "is", "that", "of", "in", "for", "on", "have", "be", "we", "me", "my", "so", "do", "this", "just", "can", "what", "with", "at", "are", "not", "your", "get", "was", "like", "will", "if", "about", "know", "but", "all", "up", "out", "go", "how", "one", "when", "see", "time", "there", "good", "now", "then", "they", "going", "think", "tomorrow", "tonight", "today", "dinner", "lunch", "call", "text", "home", "work", "need", "want", "thanks", "love", "yes", "no", "okay", "sure", "great", "sounds", "meet", "later", "week", "weekend", "morning", "night", "house", "car", "kids", "school", "game", "party", "trip", "flight", "hotel", "beach", "coffee", "drink", "house", "deck", "kitchen", "permit", "budget", "invoice", "quote", "contractor", "lumber", "paint", "roof", "fence", "garden", "gazebo", "patio", "concrete", "estimate", "schedule", "delivery", "photo"]
SYLLABLES = ["ka", "ri", "to", "ne", "lo", "ma", "su", "vi", "da", "re", "no", "sa", "te", "li", "ko", "ba", "mi", "ra", "pe", "lu", "zo", "fa", "ne", "gri", "tal", "ven", "dor", "mek", "sil", "pra", "bon", "cur", "fen", "hal", "ish", "wex", "yor", "zan", "quo"]


def pseudo_words(n: int, rng: random.Random) -> list[str]:
    seen: set[str] = set(COMMON)
    out: list[str] = []
    while len(out) < n:
        word = "".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 4)))
        if word not in seen:
            seen.add(word)
            out.append(word)
    return out


FIRST = ["Avery", "Blake", "Casey", "Devon", "Emery", "Finley", "Harper", "Jordan", "Kendall", "Logan", "Morgan", "Parker", "Quinn", "Reese", "Riley", "Sawyer", "Skyler", "Taylor"]
LAST = ["Ashdown", "Brightwater", "Coldbrook", "Dunmore", "Elmstead", "Fairhaven", "Glenrock", "Hollis", "Ironwood", "Juniper", "Kestrel", "Larkspur", "Marlow", "Northcott"]


def fake_name(i: int, rng: random.Random) -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)} {i}"


# --------------------------------------------------------------------------
# build: ~production-scale synthetic corpus
# --------------------------------------------------------------------------


def copy_rows(conn: psycopg.Connection, sql: str, rows: Iterator[Sequence[Any]]) -> int:
    count = 0
    with conn.cursor() as cur, cur.copy(sql) as copy:
        for row in rows:
            copy.write_row(row)
            count += 1
    return count


def build(args: argparse.Namespace) -> None:
    import numpy as np

    rng = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)
    root = Path(args.root)
    (root / "fts").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    conn = recreate(args.db)
    log = lambda msg: print(f"[{time.perf_counter() - started:7.1f}s] {msg}", flush=True)  # noqa: E731

    vocab = COMMON + pseudo_words(args.vocab, rng)
    ranks = np.arange(1, len(vocab) + 1, dtype=np.float64)
    weights = 1.0 / ranks**1.05
    weights /= weights.sum()
    (root / "vocab.json").write_text(json.dumps(vocab))

    # people and chats
    people = [(i, fake_name(i, rng)) for i in range(1, args.people + 1)]
    copy_rows(
        conn,
        "COPY person (display_name, short_name, is_owner, needs_review) FROM STDIN",
        iter([("Owner", "owner", True, False)] + [(n, f"p{i}", False, False) for i, n in people]),
    )
    person_ids = [int(r[0]) for r in conn.execute("SELECT person_id FROM person WHERE NOT is_owner ORDER BY person_id").fetchall()]
    owner_id = int(conn.execute("SELECT person_id FROM person WHERE is_owner").fetchone()[0])  # type: ignore[index]
    chats: list[tuple[str, str, str, str | None]] = []
    members: list[list[int]] = []
    for c in range(args.chats):
        guid = f"chat-{c}"
        if c % 8 == 0:
            group = rng.sample(person_ids, rng.randint(2, 7))
            chats.append((guid, thread_key(guid), "group", f"Group {c}"))
            members.append(group)
        else:
            chats.append((guid, thread_key(guid), "dm", None))
            members.append([rng.choice(person_ids)])
    copy_rows(conn, "COPY chat (source_guid, thread_key, kind, display_name) FROM STDIN", iter(chats))
    chat_ids = [int(r[0]) for r in conn.execute("SELECT chat_id FROM chat ORDER BY chat_id").fetchall()]
    copy_rows(
        conn,
        "COPY chat_participant (chat_id, person_id) FROM STDIN",
        ((cid, pid) for cid, group in zip(chat_ids, members, strict=True) for pid in [owner_id, *group]),
    )
    log(f"{len(people)} people, {len(chat_ids)} chats")

    # messages per chat: Zipf over chats
    chat_weights = 1.0 / np.arange(1, len(chat_ids) + 1) ** 0.9
    chat_weights /= chat_weights.sum()
    per_chat = nrng.multinomial(args.messages, chat_weights)
    lengths = np.clip(nrng.geometric(0.12, size=args.messages), 1, 40)
    word_ids = nrng.choice(len(vocab), size=int(lengths.sum()), p=weights)
    texts: list[str] = []
    offset = 0
    for n in lengths:
        texts.append(" ".join(vocab[w] for w in word_ids[offset : offset + int(n)]))
        offset += int(n)
    for i in range(0, len(texts), 997):
        texts[i] += " \U0001f389"  # a few emoji messages
    log(f"{len(texts)} message texts")

    msg_rows: list[tuple[Any, ...]] = []
    seg_plan: list[tuple[int, list[int]]] = []  # (chat index, message indexes)
    m = 0
    for ci, count in enumerate(per_chat):
        count = int(count)
        if count == 0:
            continue
        start = EPOCH + timedelta(days=rng.random() * SPAN_DAYS * 0.5)
        at = start
        current: list[int] = []
        target = rng.randint(1, 10)
        for _ in range(count):
            at += timedelta(minutes=rng.expovariate(1 / 180))
            sender = None if rng.random() < 0.45 else rng.choice(members[ci])
            guid = f"m{m}"
            msg_rows.append(
                (guid, message_key(guid), chat_ids[ci], sender, sender is None, at, "imessage", texts[m], normalize_text(texts[m]))
            )
            current.append(m)
            if len(current) >= target:
                seg_plan.append((ci, current))
                current, target = [], rng.randint(1, 10)
            m += 1
        if current:
            seg_plan.append((ci, current))
    copy_rows(
        conn,
        "COPY message (source_guid, message_key, chat_id, sender_person_id, is_from_me, sent_at, service, text_original, text_normalized) FROM STDIN",
        iter(msg_rows),
    )
    message_ids = [int(r[0]) for r in conn.execute("SELECT message_id FROM message ORDER BY message_id").fetchall()]
    log(f"{len(message_ids)} messages")

    names = dict(zip(person_ids, (n for _, n in people), strict=True))
    sessions = []
    for ci, idxs in seg_plan:
        sessions.append((chat_ids[ci], msg_rows[idxs[0]][5], msg_rows[idxs[-1]][5], 3.0))
    copy_rows(conn, "COPY session (chat_id, started_at, ended_at, gap_hours) FROM STDIN", iter(sessions))
    session_ids = [int(r[0]) for r in conn.execute("SELECT session_id FROM session ORDER BY session_id").fetchall()]
    seg_rows = []
    rendered_texts = []
    for (ci, idxs), sid in zip(seg_plan, session_ids, strict=True):
        header = "Chat: " + ", ".join(names[p] for p in members[ci]) + (f' (group "{chats[ci][3]}")' if chats[ci][3] else "")
        lines = [header, "---"]
        for i in idxs:
            sender = msg_rows[i][3]
            lines.append(f"[{msg_rows[i][5]:%H:%M}] {'owner' if sender is None else f'p{sender}'}: {msg_rows[i][7]}")
        text = "\n".join(lines)
        rendered_texts.append(text)
        stable = hashlib.sha256(f"seg:{sid}".encode()).hexdigest()
        seg_rows.append((stable, chat_ids[ci], sid, 0, msg_rows[idxs[0]][5], msg_rows[idxs[-1]][5], len(idxs), 10, text, "x", "cfg"))
    copy_rows(
        conn,
        "COPY segment (stable_key, chat_id, session_id, seq_in_session, started_at, ended_at, message_count, token_count, rendered_text, rendered_sha256, seg_config_hash) FROM STDIN",
        iter(seg_rows),
    )
    segment_ids = [int(r[0]) for r in conn.execute("SELECT segment_id FROM segment ORDER BY segment_id").fetchall()]
    copy_rows(
        conn,
        "COPY segment_message (segment_id, message_id) FROM STDIN",
        ((sid, message_ids[i]) for sid, (_ci, idxs) in zip(segment_ids, seg_plan, strict=True) for i in idxs),
    )
    log(f"{len(segment_ids)} segments")

    # attachments and their text
    kinds = [("image/jpeg", "IMG.jpg"), ("image/heic", "IMG.HEIC"), ("video/quicktime", "clip.mov"), ("audio/x-caf", "Audio.caf"), ("application/pdf", "doc.pdf"), ("text/vcard", "card.vcf")]
    kind_weights = [0.45, 0.25, 0.1, 0.05, 0.1, 0.05]
    att_msgs = rng.sample(range(len(message_ids)), args.attachments)
    att_rows = []
    for a, _mi in enumerate(att_msgs):
        mime, fname = rng.choices(kinds, kind_weights)[0]
        guid = f"a{a}"
        att_rows.append((guid, attachment_key(guid), fname, mime, rng.randint(10_000, 5_000_000), hashlib.sha256(guid.encode()).hexdigest(), "materialized"))
    copy_rows(conn, "COPY attachment (source_guid, attachment_key, filename, mime_type, byte_size, sha256, state) FROM STDIN", iter(att_rows))
    att_ids = [int(r[0]) for r in conn.execute("SELECT attachment_id FROM attachment ORDER BY attachment_id").fetchall()]
    copy_rows(conn, "COPY message_attachment (message_id, attachment_id, ordinal) FROM STDIN", ((message_ids[mi], aid, 0) for mi, aid in zip(att_msgs, att_ids, strict=True)))
    conn.execute("UPDATE message SET has_attachments = true WHERE message_id IN (SELECT message_id FROM message_attachment)")
    chunk_atts = rng.sample(att_ids, min(args.chunks, len(att_ids)))
    chunk_words = nrng.choice(len(vocab), size=len(chunk_atts) * 60, p=weights)
    chunk_rows = [(aid, "ocr", 0, " ".join(vocab[w] for w in chunk_words[i * 60 : (i + 1) * 60])) for i, aid in enumerate(chunk_atts)]
    copy_rows(conn, "COPY attachment_chunk (attachment_id, kind, seq, text) FROM STDIN", iter(chunk_rows))
    log(f"{len(att_ids)} attachments, {len(chunk_rows)} chunks")

    # FTS sidecar
    fts_path = root / "fts" / "fts.db"
    fts_path.unlink(missing_ok=True)
    fts = apsw.Connection(str(fts_path))
    create_schema(fts)
    with fts:
        cur = fts.cursor()
        for sid, (stable, *_rest), text in zip(segment_ids, seg_rows, rendered_texts, strict=True):
            normalized = normalize_text(text)
            cur.execute("INSERT INTO seg_map (fts_rowid, segment_id, stable_key) VALUES (?, ?, ?)", (sid, sid, stable))
            cur.execute("INSERT INTO seg_fts (rowid, text) VALUES (?, ?)", (sid, normalized))
            cur.execute("INSERT INTO seg_fts_tri (rowid, text) VALUES (?, ?)", (sid, normalized))
        for chunk_id, aid, text in conn.execute("SELECT chunk_id, attachment_id, text FROM attachment_chunk").fetchall():
            normalized = normalize_text(text)
            cur.execute("INSERT INTO att_map (fts_rowid, chunk_id, attachment_id) VALUES (?, ?, ?)", (chunk_id, chunk_id, aid))
            cur.execute("INSERT INTO att_fts (rowid, text) VALUES (?, ?)", (chunk_id, normalized))
            cur.execute("INSERT INTO att_fts_tri (rowid, text) VALUES (?, ?)", (chunk_id, normalized))
    fts.close()
    log(f"FTS sidecar {fts_path.stat().st_size / 2**20:.0f} MiB")

    # vectors with topic structure, generated inside Postgres
    conn.execute(f"CREATE TABLE bench_topic (topic int PRIMARY KEY, c vector({TEXT_DIM}), ci vector({IMAGE_DIM}))")
    conn.execute(
        f"""INSERT INTO bench_topic
            SELECT t,
                   l2_normalize(array(SELECT random() - 0.5 + 0 * t FROM generate_series(1, {TEXT_DIM}))::vector),
                   l2_normalize(array(SELECT random() - 0.5 + 0 * t FROM generate_series(1, {IMAGE_DIM}))::vector)
            FROM generate_series(0, {TOPICS - 1}) t"""
    )
    # HNSW indexes build far faster in bulk (and in parallel) than row by
    # row, so drop them for the load and recreate them with the migrations'
    # own definitions.
    hnsw = conn.execute(
        "SELECT indexname, indexdef FROM pg_indexes WHERE indexdef LIKE '%USING hnsw%'"
    ).fetchall()
    for name, _definition in hnsw:
        conn.execute(f"DROP INDEX {name}")
    # Noise scale k spreads each text item's cosine similarity to its centre
    # over about 0.35-0.85 (1 / sqrt(1 + dim * k^2 / 12)); images run higher.
    for table, key, dim, col, id_sql in (
        ("segment_embedding", "segment_id", TEXT_DIM, "c", "SELECT segment_id AS id FROM segment"),
        ("attachment_chunk_embedding", "chunk_id", TEXT_DIM, "c", "SELECT chunk_id AS id FROM attachment_chunk"),
        ("attachment_mm_embedding", "attachment_id", IMAGE_DIM, "ci", f"SELECT attachment_id AS id FROM attachment WHERE mime_type LIKE 'image/%' LIMIT {args.image_vectors}"),
    ):
        extra = "text_sha256" if table != "attachment_mm_embedding" else "media_sha256"
        conn.execute(
            f"""INSERT INTO {table} ({key}, model, dim, {extra}, vec)
                SELECT x.id, 'synthetic', {dim}, 'x',
                       l2_normalize(t.{col} + array(
                           SELECT (random() - 0.5) * (0.047 + (x.id % 7) * 0.026)
                           FROM generate_series(1, {dim}))::vector)::halfvec
                FROM ({id_sql}) x JOIN bench_topic t ON t.topic = x.id % {TOPICS}"""
        )
        log(f"{table} filled")
    conn.execute("SET maintenance_work_mem = '4GB'")
    conn.execute("SET max_parallel_maintenance_workers = 8")
    for name, definition in hnsw:
        conn.execute(definition)
        log(f"rebuilt {name}")
    conn.execute("ANALYZE")
    log("done")
    conn.close()


# --------------------------------------------------------------------------
# measure
# --------------------------------------------------------------------------


def percentile(values: Sequence[float], p: float) -> float:
    ordered = sorted(values)
    rank = max(1, round(p / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def timed(fn: Callable[[], Any], runs: int) -> tuple[list[float], Any]:
    times, last = [], None
    for _ in range(runs):
        start = time.perf_counter()
        last = fn()
        times.append((time.perf_counter() - start) * 1000)
    return times, last


def measure(args: argparse.Namespace) -> None:
    from starlette.testclient import TestClient

    from imsg.search_page import auth as auth_module
    from imsg.search_page.app import AppDeps, ConnectionPool, FtsReaders, build_app
    from imsg.search_page.auth import LoginGuard, PasswordFile, SessionStore, set_password
    from imsg.search_page.config import SearchPageConfig
    from imsg.search_page.media import MediaConverter
    from imsg.search_page.search import (
        QueryVectors,
        SearchRequest,
        SearchSettings,
        run_fulltext,
        run_semantic,
    )
    from imsg.search_page.server import open_fts_reader

    root = Path(args.root)
    vocab = json.loads((root / "vocab.json").read_text())
    settings = SearchSettings(
        timezone="America/New_York", index_unsent=False, rrf_k=60, fts_max_hits=20000,
        unindexed_window_days=60, semantic_enabled=True, multimodal_enabled=True,
        text_min_similarity=args.text_floor, multimodal_min_similarity=args.image_floor,
        max_hits_per_channel=2000, ef_search=200, max_scan_tuples=50000,
    )
    pg = psycopg.connect(db_dsn(args.db), autocommit=True)
    fts = open_fts_reader(root / "fts" / "fts.db")
    top_person = pg.execute(
        "SELECT p.short_name FROM person p JOIN message m ON m.sender_person_id = p.person_id "
        "GROUP BY p.short_name ORDER BY count(*) DESC LIMIT 1"
    ).fetchone()[0]  # type: ignore[index]
    n = len(vocab)

    def word(rank_in_40k: int) -> str:
        """The word at the same relative frequency rank as `rank_in_40k` in
        the default 40,000-word vocabulary."""
        return str(vocab[min(n - 1, rank_in_40k * n // 40_000)])

    rare, uncommon, medium = word(20000), word(3000), word(400)
    queries: list[tuple[str, SearchRequest]] = [
        ("rare word", SearchRequest(query=rare)),
        ("uncommon word", SearchRequest(query=uncommon)),
        ("medium word", SearchRequest(query=medium)),
        ("common word (hits the cap)", SearchRequest(query="the")),
        ("two words", SearchRequest(query=f"{word(150)} {word(250)}")),
        ("quoted phrase", SearchRequest(query=f'"{word(120)} {vocab[1]}"')),
        ("emoji", SearchRequest(query="\U0001f389")),
        ("medium word, one person", SearchRequest(query=medium, people=(top_person,))),
        ("medium word, one year", SearchRequest(query=medium, after="2012-01-01", before="2013-01-01")),
    ]
    report: dict[str, Any] = {"host": os.uname().nodename, "when": datetime.now(UTC).isoformat(), "queries": []}
    for label, request in queries:
        first_ms = timed(lambda r=request: run_fulltext(pg, fts, r, settings), 1)[0][0]
        times, result = timed(lambda r=request: run_fulltext(pg, fts, r, settings), args.runs)
        report["queries"].append(
            {
                "label": label,
                "query": request.query,
                "hits": result.total_hits,
                "threads": len(result.threads("relevance")),
                "capped": sorted(result.capped),
                "first_ms": round(first_ms, 1),
                "p50_ms": round(statistics.median(times), 1),
                "p95_ms": round(percentile(times, 95), 1),
                "stages_ms": {k: round(v, 1) for k, v in result.timings_ms.items()},
            }
        )
        print(f"{label:34s} hits={result.total_hits:6d} first={first_ms:7.1f} p50={statistics.median(times):7.1f} p95={percentile(times, 95):7.1f} ms", flush=True)

    # The whole first page over HTTP, and the next page.
    auth_module.SCRYPT_N = 2**14
    pw = root / "private" / "pw"
    set_password(pw, "benchmark password only")
    passwords = PasswordFile(pw)
    deps = AppDeps(
        page=SearchPageConfig(enabled=True, allowed_hosts=["testserver"]),
        settings=settings,
        data_root=root,
        pool=ConnectionPool(lambda: psycopg.connect(db_dsn(args.db), autocommit=True), 4),
        fts=FtsReaders(lambda: open_fts_reader(root / "fts" / "fts.db"), root / "fts" / "fts.db", 4),
        passwords=passwords,
        sessions=SessionStore(root / "private" / "sessions.json", lifetime_seconds=3600),
        login_guard=LoginGuard(passwords, max_failures_per_client=5, max_failures_global=30, window_seconds=900),
        media=MediaConverter(root / "thumbs"),
        model_api=None,
    )
    client = TestClient(build_app(deps))
    import re as _re

    token = _re.search(r'name="login_token" value="([^"]+)"', client.get("/login").text).group(1)  # type: ignore[union-attr]
    client.post("/login", data={"login_token": token, "password": "benchmark password only", "next": "/"})
    report["pages"] = []
    for label, request in queries:
        params = {"q": request.query}
        if request.people:
            params["people"] = ",".join(request.people)
        page_times, next_times = [], []
        for _ in range(args.runs):
            deps.cache.clear()
            start = time.perf_counter()
            response = client.get("/search", params=params)
            page_times.append((time.perf_counter() - start) * 1000)
            assert response.status_code == 200, response.text[:200]
            nxt = _re.search(r'data-next="([^"]+)"', response.text)
            if nxt:
                start = time.perf_counter()
                client.get(nxt.group(1).replace("&amp;", "&"))
                next_times.append((time.perf_counter() - start) * 1000)
        report["pages"].append(
            {
                "label": label,
                "page_p50_ms": round(statistics.median(page_times), 1),
                "page_p95_ms": round(percentile(page_times, 95), 1),
                "next_page_p50_ms": round(statistics.median(next_times), 1) if next_times else None,
                "bytes": len(response.content),
            }
        )
        print(f"page {label:29s} p50={statistics.median(page_times):7.1f} p95={percentile(page_times, 95):7.1f} next={statistics.median(next_times) if next_times else 0:7.1f} ms", flush=True)

    # Semantic threshold scans with topic-centred query vectors.
    report["semantic"] = []
    for topic in (3, 57, 141):
        text_q = pg.execute(
            f"SELECT l2_normalize(c + array(SELECT (random() - 0.5) * 0.01 + 0 * {topic} FROM generate_series(1, {TEXT_DIM}))::vector)::text, "
            f"l2_normalize(ci + array(SELECT (random() - 0.5) * 0.01 + 0 * {topic} FROM generate_series(1, {IMAGE_DIM}))::vector)::text "
            "FROM bench_topic WHERE topic = %s",
            (topic,),
        ).fetchone()
        assert text_q is not None
        vectors = QueryVectors(text=json.loads(text_q[0]), multimodal=json.loads(text_q[1]))
        times = []
        for _ in range(args.runs):
            result = run_fulltext(pg, fts, SearchRequest(query=rare), settings)
            start = time.perf_counter()
            status = run_semantic(pg, result, vectors, settings)
            times.append((time.perf_counter() - start) * 1000)
        report["semantic"].append(
            {
                "topic": topic,
                "hits_added": status.added_hits,
                "counts": dict(result.counts),
                "capped": sorted(result.capped),
                "p50_ms": round(statistics.median(times), 1),
                "p95_ms": round(percentile(times, 95), 1),
                "stages_ms": {k: round(v, 1) for k, v in result.timings_ms.items() if k.startswith("pg_") or k == "semantic_total"},
            }
        )
        print(f"semantic topic {topic:3d}: added={status.added_hits} counts={dict(result.counts)} p50={statistics.median(times):.1f} p95={percentile(times, 95):.1f} ms", flush=True)

    sizes = pg.execute(
        """SELECT relname, pg_total_relation_size(c.oid) FROM pg_class c
           JOIN pg_namespace n ON n.oid = c.relnamespace
           WHERE n.nspname = 'public' AND c.relkind IN ('r','i')
           ORDER BY 2 DESC LIMIT 12"""
    ).fetchall()
    report["relation_sizes_mib"] = {name: round(size / 2**20, 1) for name, size in sizes}
    report["fts_sidecar_mib"] = round((root / "fts" / "fts.db").stat().st_size / 2**20, 1)
    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# demo: a small fictional corpus with media, for looking at the page
# --------------------------------------------------------------------------


def demo(args: argparse.Namespace) -> None:
    from imsg.embed.fts.sync import upsert_chunk_row, upsert_segment_row

    root = Path(args.root)
    (root / "fts").mkdir(parents=True, exist_ok=True)
    (root / "attachments").mkdir(parents=True, exist_ok=True)
    (root / "fts" / "fts.db").unlink(missing_ok=True)
    conn = recreate(args.db)
    fts = apsw.Connection(str(root / "fts" / "fts.db"))
    create_schema(fts)
    media = Path(args.media) if args.media else None

    def one(sql: str, params: Sequence[Any]) -> int:
        row = conn.execute(sql, params).fetchone()
        assert row is not None
        return int(row[0])

    def person(name: str, owner: bool = False) -> int:
        short = name.split()[0].lower() if not owner else "owner"
        return one("INSERT INTO person (display_name, short_name, is_owner, needs_review) VALUES (%s,%s,%s,false) RETURNING person_id", (name, short, owner))

    owner = person("Owner", True)
    alice, bob, carol, dan = person("Alice Example"), person("Bob Builder"), person("Carol Example"), person("Dana Driver")
    names = {alice: "alice", bob: "bob", carol: "carol", dan: "dan"}

    def chat(guid: str, people: list[int], kind: str = "dm", display: str | None = None, unfiled: str | None = None) -> int:
        cid = one("INSERT INTO chat (source_guid, thread_key, kind, display_name, unfiled_key) VALUES (%s,%s,%s,%s,%s) RETURNING chat_id", (guid, thread_key(guid), kind, display, unfiled))
        for p in [owner, *people]:
            conn.execute("INSERT INTO chat_participant (chat_id, person_id) VALUES (%s,%s) ON CONFLICT DO NOTHING", (cid, p))
        return cid

    counter = [0]

    def segment(chat_id: int, lines: list[tuple[datetime, int | None, str]], attach: dict[int, list[tuple[str, str, bytes, str | None]]] | None = None) -> list[int]:
        sid = one("INSERT INTO session (chat_id, started_at, ended_at, gap_hours) VALUES (%s,%s,%s,3.0) RETURNING session_id", (chat_id, lines[0][0], lines[-1][0]))
        body = "\n".join(f"[{at:%H:%M}] {names.get(p, 'owner') if p else 'owner'}: {t}" for at, p, t in lines)
        extra = []
        for idx, items in (attach or {}).items():
            for fname, _mime, _content, text in items:
                extra.append(f"[{lines[idx][0]:%H:%M}] attachment {fname}: {text or ''}")
        rendered = "---\n" + body + ("\n" + "\n".join(extra) if extra else "")
        counter[0] += 1
        stable = hashlib.sha256(f"demo-seg-{counter[0]}".encode()).hexdigest()
        seg = one(
            "INSERT INTO segment (stable_key, chat_id, session_id, seq_in_session, started_at, ended_at, message_count, token_count, rendered_text, rendered_sha256, seg_config_hash) VALUES (%s,%s,%s,0,%s,%s,%s,10,%s,'x','cfg') RETURNING segment_id",
            (stable, chat_id, sid, lines[0][0], lines[-1][0], len(lines), rendered),
        )
        upsert_segment_row(fts, seg, stable, rendered)
        ids = []
        for idx, (at, p, text) in enumerate(lines):
            counter[0] += 1
            guid = f"demo-msg-{counter[0]}"
            mid = one(
                "INSERT INTO message (source_guid, message_key, chat_id, sender_person_id, is_from_me, sent_at, service, text_original, text_normalized, is_edited) VALUES (%s,%s,%s,%s,%s,%s,'imessage',%s,%s,%s) RETURNING message_id",
                (guid, message_key(guid), chat_id, p, p is None, at, text, normalize_text(text), "(edited)" in text),
            )
            conn.execute("INSERT INTO segment_message (segment_id, message_id) VALUES (%s,%s)", (seg, mid))
            for fname, mime, content, att_text in (attach or {}).get(idx, []):
                counter[0] += 1
                aguid = f"demo-att-{counter[0]}"
                sha = hashlib.sha256(content).hexdigest()
                path = root / "attachments" / sha[:2] / sha
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                aid = one(
                    "INSERT INTO attachment (source_guid, attachment_key, filename, mime_type, byte_size, sha256, state) VALUES (%s,%s,%s,%s,%s,%s,'materialized') RETURNING attachment_id",
                    (aguid, attachment_key(aguid), fname, mime, len(content), sha),
                )
                conn.execute("INSERT INTO message_attachment (message_id, attachment_id, ordinal) VALUES (%s,%s,0)", (mid, aid))
                conn.execute("UPDATE message SET has_attachments = true WHERE message_id = %s", (mid,))
                if att_text:
                    kind = "pdf_text" if mime == "application/pdf" else ("transcript" if mime.startswith("audio") else "caption")
                    conn.execute("INSERT INTO enrichment (attachment_id, kind, state, text) VALUES (%s,%s,'done',%s)", (aid, kind, att_text))
                    chunk = one("INSERT INTO attachment_chunk (attachment_id, kind, seq, text) VALUES (%s,%s,0,%s) RETURNING chunk_id", (aid, kind, att_text))
                    upsert_chunk_row(fts, chunk, aid, att_text)
            ids.append(mid)
        return ids

    def blob(name: str, fallback: bytes) -> bytes:
        if media and (media / name).exists():
            return (media / name).read_bytes()
        return fallback

    photo = blob("deck.jpg", tiny_png(320, 240, (120, 90, 60)))
    photo2 = blob("gazebo.heic", tiny_png(320, 240, (60, 120, 90)))
    clip = blob("clip.mov", b"")
    voice = blob("voice.caf", b"")
    pdf = blob("bid.pdf", b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n")
    t = datetime(2023, 4, 12, 17, 58, tzinfo=UTC)
    deck = chat("demo-deck", [alice, bob], "group", "Deck project")
    ids = segment(deck, [
        (t, None, "did the revised bid for the deck come through?"),
        (t + timedelta(minutes=5), alice, "yes, see attached. materials came in under budget"),
        (t + timedelta(minutes=7), bob, "footings go in Monday if the permit clears"),
        (t + timedelta(minutes=9), None, "great, let's lock it in"),
    ], {1: [("bid-rev3.pdf", "application/pdf", pdf, "Deck rebuild estimate: footings, joists, composite decking, materials 14,200")]})
    conn.execute("INSERT INTO tapback (source_guid, target_source_guid, target_message_id, sender_person_id, is_from_me, kind, acted_at) SELECT 'demo-tb-1', source_guid, message_id, %s, false, 'loved', now() FROM message WHERE message_id = %s", (bob, ids[3]))
    segment(deck, [
        (t + timedelta(days=12), bob, "footings are poured, photo attached"),
        (t + timedelta(days=12, minutes=2), None, "looks solid. is the deck stain ordered?"),
        (t + timedelta(days=12, minutes=30), alice, "stain arrives Thursday (edited)"),
    ], {0: [("IMG_2041.jpg", "image/jpeg", photo, "concrete deck footings with rebar")]})
    dm = chat("demo-alice", [alice])
    segment(dm, [
        (t - timedelta(days=30), alice, "coffee tomorrow? want to talk about the deck and the garden"),
        (t - timedelta(days=30, minutes=-3), None, "yes! 9am at the usual place"),
    ])
    segment(dm, [
        (t + timedelta(days=40), alice, "the gazebo kit arrived, it's huge"),
        (t + timedelta(days=40, minutes=1), None, "send a photo"),
        (t + timedelta(days=40, minutes=2), alice, "here it is"),
    ], {2: [("IMG_0007.HEIC", "image/heic", photo2, "wooden gazebo frame in a backyard")]}
       | ({1: [("clip.mov", "video/quicktime", clip, None)]} if clip else {}))
    dm2 = chat("demo-carol", [carol])
    segment(dm2, [
        (t + timedelta(days=3), carol, "can you send me the contractor's number for the deck?"),
        (t + timedelta(days=3, minutes=4), None, "sure, it's in the card I just sent"),
    ], ({1: [("Audio Message.caf", "audio/x-caf", voice, "the contractor number is in the card")]} if voice else None))
    holding = chat("demo-unfiled", [dan], "dm", "Unfiled: Dana Driver", "sender:dana")
    segment(holding, [(t + timedelta(days=5), dan, "deck boards are on sale this week")])
    conn.execute("UPDATE message SET deleted_at = now() WHERE text_original LIKE 'deck boards%'")
    for i in range(1, 60):
        segment(dm, [(t + timedelta(days=60 + i), alice if i % 2 else None, f"garden update {i}: tomatoes, basil and the deck planters")])

    # Semantic neighbours for the demo query "backyard projects", using the
    # fake providers `serve` runs: a few segments and one image are placed
    # at chosen similarities to that query's fake vector.
    from imsg.embed.provider import FakeMultimodalEmbeddingProvider, FakeTextEmbeddingProvider

    def near(q: list[float], similarity: float, seed: int) -> list[float]:
        rng = random.Random(seed)
        r = [rng.uniform(-1, 1) for _ in q]
        dot = sum(a * b for a, b in zip(r, q, strict=True))
        r = [a - dot * b for a, b in zip(r, q, strict=True)]
        norm = sum(a * a for a in r) ** 0.5
        other = (1 - similarity**2) ** 0.5
        return [similarity * a + other * b / norm for a, b in zip(q, r, strict=True)]

    q_text = FakeTextEmbeddingProvider(dim=TEXT_DIM).embed_query("backyard projects", instruction="retrieve")
    q_image = FakeMultimodalEmbeddingProvider(dim=IMAGE_DIM).embed_text("backyard projects")
    for similarity, needle in ((0.78, "the gazebo kit arrived%"), (0.66, "footings are poured%"), (0.52, "coffee tomorrow%")):
        seg_id = one(
            "SELECT sm.segment_id FROM message m JOIN segment_message sm ON sm.message_id = m.message_id "
            "WHERE m.text_original LIKE %s LIMIT 1",
            (needle,),
        )
        conn.execute(
            "INSERT INTO segment_embedding (segment_id, model, dim, text_sha256, vec) VALUES (%s,'fake',%s,'x',%s::halfvec)",
            (seg_id, TEXT_DIM, vector_literal(near(q_text, similarity, seg_id))),
        )
    image_id = one("SELECT attachment_id FROM attachment WHERE filename = 'IMG_0007.HEIC'", ())
    conn.execute(
        "INSERT INTO attachment_mm_embedding (attachment_id, model, dim, media_sha256, vec) VALUES (%s,'fake',%s,'x',%s::halfvec)",
        (image_id, IMAGE_DIM, vector_literal(near(q_image, 0.31, image_id))),
    )
    fts.close()
    conn.close()
    print(f"demo corpus in {args.db}, data root {root}")


def tiny_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """A solid-colour PNG written by hand (no imaging library)."""
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def serve(args: argparse.Namespace) -> None:
    """Serve the page on 127.0.0.1 against a scratch corpus (password from
    IMSG_BENCH_PASSWORD), with the fake-model internal API if asked."""
    import uvicorn

    from imsg.embed.provider import FakeMultimodalEmbeddingProvider, FakeTextEmbeddingProvider
    from imsg.retrieval.reranker import FakeRerankerProvider
    from imsg.search_page.app import AppDeps, ConnectionPool, FtsReaders, build_app
    from imsg.search_page.auth import LoginGuard, PasswordFile, SessionStore, set_password
    from imsg.search_page.config import SearchPageConfig
    from imsg.search_page.media import MediaConverter
    from imsg.search_page.model_api_client import ModelApiClient
    from imsg.search_page.model_api_server import ModelAccess, ModelApiServer, ModelApiState
    from imsg.search_page.search import SearchSettings
    from imsg.search_page.secret_files import write_private_file
    from imsg.search_page.server import open_fts_reader

    root = Path(args.root)
    password = os.environ.get("IMSG_BENCH_PASSWORD", "demo password only")
    pw = root / "private" / "pw"
    set_password(pw, password)
    secret = "d" * 48
    write_private_file(root / "private" / "model.secret", secret.encode())
    state = ModelApiState(
        models=ModelAccess(
            text_provider=FakeTextEmbeddingProvider(dim=TEXT_DIM),
            reranker=FakeRerankerProvider(),
            multimodal_provider=FakeMultimodalEmbeddingProvider(dim=IMAGE_DIM),
            query_instruction="retrieve",
            multimodal_enabled=True,
        ),
        secret=secret,
        port=0,
        warm_up=None,
        idle_unloader=None,
        log=print,
    )
    api = ModelApiServer(0, state).start()
    passwords = PasswordFile(pw)
    host = f"127.0.0.1:{args.port}"
    deps = AppDeps(
        page=SearchPageConfig(enabled=True, allowed_hosts=[host, f"localhost:{args.port}"]),
        settings=SearchSettings(
            timezone="America/New_York", index_unsent=False, rrf_k=60, fts_max_hits=20000,
            unindexed_window_days=60, semantic_enabled=True, multimodal_enabled=True,
            text_min_similarity=0.45, multimodal_min_similarity=0.2, max_hits_per_channel=2000,
            ef_search=200, max_scan_tuples=50000,
        ),
        data_root=root,
        pool=ConnectionPool(lambda: psycopg.connect(db_dsn(args.db), autocommit=True), 4),
        fts=FtsReaders(lambda: open_fts_reader(root / "fts" / "fts.db"), root / "fts" / "fts.db", 4),
        passwords=passwords,
        sessions=SessionStore(root / "private" / "sessions.json", lifetime_seconds=86400),
        login_guard=LoginGuard(passwords, max_failures_per_client=5, max_failures_global=30, window_seconds=900),
        media=MediaConverter(root / "thumbs"),
        model_api=ModelApiClient(port=api.port, secret_file=root / "private" / "model.secret", timeout_seconds=5),
    )
    print(f"serving http://{host}/ (fake models on 127.0.0.1:{api.port})", flush=True)
    uvicorn.run(build_app(deps), host="127.0.0.1", port=args.port, log_level="warning", access_log=False, server_header=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--root", required=True)
    b.add_argument("--db", default=BENCH_DB)
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--people", type=int, default=10_800)
    b.add_argument("--chats", type=int, default=8_000)
    b.add_argument("--messages", type=int, default=675_000)
    b.add_argument("--attachments", type=int, default=100_000)
    b.add_argument("--chunks", type=int, default=60_000)
    b.add_argument("--image-vectors", type=int, default=64_000)
    b.add_argument("--vocab", type=int, default=40_000)
    m = sub.add_parser("measure")
    m.add_argument("--root", required=True)
    m.add_argument("--db", default=BENCH_DB)
    m.add_argument("--runs", type=int, default=7)
    m.add_argument("--text-floor", type=float, default=0.45)
    m.add_argument("--image-floor", type=float, default=0.2)
    m.add_argument("--out", default="search-page-bench.json")
    d = sub.add_parser("demo")
    d.add_argument("--root", required=True)
    d.add_argument("--db", default=DEMO_DB)
    d.add_argument("--media", default=None, help="directory with deck.jpg, gazebo.heic, clip.mov, voice.caf, bid.pdf")
    s = sub.add_parser("serve")
    s.add_argument("--root", required=True)
    s.add_argument("--db", default=DEMO_DB)
    s.add_argument("--port", type=int, default=8719)
    args = parser.parse_args(argv)
    {"build": build, "measure": measure, "demo": demo, "serve": serve}[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
