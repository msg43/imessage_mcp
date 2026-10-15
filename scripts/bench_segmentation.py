"""Non-mutating paired chunking experiment on a deterministic corpus sample.

Both variants use identical source messages, text rendering and real text
embeddings. Queries are automatically constructed excerpts with known source
message anchors: this measures source rediscovery, NOT real-question relevance.
All text stays in memory; only aggregate counts/timings are saved under data_root.
Production segments, vectors, and relevance judgments are never modified.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import time
from pathlib import Path

from imsg.background_gate import BackgroundGate
from imsg.config.loader import load_config
from imsg.db.connection import connect
from imsg.db.fingerprint import verify_data_directory
from imsg.heavy_lock import HeavyModelLock
from imsg.memory_admission import MemoryAdmission, ModelRole
from imsg.providers.factory import build_boundary_provider, build_text_provider
from imsg.segment.boundaries import segment_session
from imsg.segment.models import MessageForSegmentation, Session
from imsg.segment.render import render_message_line


def simple_chunks(messages, *, gap_hours=3.0, max_messages=50, max_tokens=2000, overlap=3):
    """Time/size chunks with overlap only across size boundaries; no model."""
    chunks, current, tokens = [], [], 0
    for message in messages:
        cost = max(1, len(message.text or "") // 4)
        gap = bool(
            current and (message.sent_at - current[-1].sent_at).total_seconds() > gap_hours * 3600
        )
        size = bool(current and (len(current) >= max_messages or tokens + cost > max_tokens))
        if gap or size:
            chunks.append(tuple(current))
            current = (
                [] if gap else current[-min(overlap, max(0, len(current) - 1)) :] if overlap else []
            )
            tokens = sum(max(1, len(m.text or "") // 4) for m in current)
            while current and (len(current) >= max_messages or tokens + cost > max_tokens):
                current.pop(0)
                tokens = sum(max(1, len(m.text or "") // 4) for m in current)
        current.append(message)
        tokens += cost
    if current:
        chunks.append(tuple(current))
    return chunks


def main():
    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument(
        "--queries",
        type=Path,
        help="Private JSON questions with anchor_ids; all anchors must be in the sample.",
    )
    ap.add_argument("--sessions", type=int, default=12)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg.models.backend != "real":
        raise SystemExit("This experiment requires real providers, never fakes.")
    output = (
        cfg.paths.data_root
        / "eval"
        / (
            "segmentation-paraphrase-comparison.json"
            if args.queries
            else "segmentation-comparison.json"
        )
    )
    with HeavyModelLock(cfg.paths.data_root, command="segmentation comparison", wait=False):
        admission = MemoryAdmission.for_role(
            cfg, ModelRole.SEGMENT, command="segmentation comparison"
        )
        reason = BackgroundGate.from_config(cfg, log=print).admit(admission)
        if reason:
            raise SystemExit(reason.line())
        with connect(cfg.database) as db:
            verify_data_directory(db, cfg.paths.data_root)
            db.execute("SET TRANSACTION READ ONLY")
            rows = db.execute(
                """SELECT se.chat_id,se.started_at,se.ended_at
              FROM session se JOIN segment s USING(session_id)
              GROUP BY se.session_id
              HAVING sum(s.message_count) BETWEEN 15 AND 60
              ORDER BY md5(se.session_id::text) LIMIT %s""",
                (args.sessions,),
            ).fetchall()
            samples = []
            for chat, start, end in rows:
                records = db.execute(
                    """SELECT m.message_id,m.source_guid,m.sent_at,m.is_from_me,
                    COALESCE(p.short_name,'unknown'),m.text_original,m.is_unsent,m.is_edited
                    FROM message m LEFT JOIN person p ON p.person_id=m.sender_person_id
                    WHERE m.chat_id=%s AND m.sent_at >= %s AND m.sent_at <= %s AND NOT m.is_unsent
                    ORDER BY m.sent_at,m.message_id""",
                    (chat, start, end),
                ).fetchall()
                samples.append(
                    [
                        MessageForSegmentation(
                            message_id=x[0],
                            source_guid=x[1],
                            chat_id=chat,
                            sent_at=x[2],
                            is_from_me=x[3],
                            sender_short_name="owner" if x[3] else x[4],
                            text=x[5],
                            is_unsent=x[6],
                            is_edited=x[7],
                            has_attachments=False,
                        )
                        for x in records
                    ]
                )
        prompt = args.prompt.read_text()
        provider = build_boundary_provider(cfg, prompt)
        variants = {"topical": [], "simple": []}
        times = {"topical": 0.0, "simple": 0.0}
        fallback = 0
        topical_session_seconds = []
        for i, messages in enumerate(samples):
            if not messages:
                continue
            started = time.perf_counter()
            drafts, failed = segment_session(
                Session(
                    messages[0].chat_id,
                    messages[0].sent_at,
                    messages[-1].sent_at,
                    tuple(messages),
                    cfg.segmentation.session_gap_hours,
                ),
                topical_min_messages=cfg.segmentation.topical_min_messages,
                max_messages=cfg.segmentation.max_messages,
                max_tokens=cfg.segmentation.max_tokens,
                boundary_provider=provider,
            )
            elapsed = time.perf_counter() - started
            times["topical"] += elapsed
            topical_session_seconds.append(elapsed)
            fallback += failed
            variants["topical"].extend(d.messages for d in drafts)
            started = time.perf_counter()
            variants["simple"].extend(
                simple_chunks(
                    messages,
                    gap_hours=cfg.segmentation.session_gap_hours,
                    max_messages=cfg.segmentation.max_messages,
                    max_tokens=cfg.segmentation.max_tokens,
                )
            )
            times["simple"] += time.perf_counter() - started
            print(f"session {i + 1}/{len(samples)} compared", flush=True)
        unload = getattr(provider, "unload", None)
        if unload:
            unload()
        del provider
        gc.collect()
        queries = []
        for messages in samples:
            eligible = [m for m in messages if len((m.text or "").split()) >= 8]
            if eligible:
                m = eligible[len(eligible) // 2]
                queries.append((" ".join(m.text.split()[:8]), {m.message_id}))
        if args.queries:
            queries = [
                (entry["query"], set(entry["anchor_ids"]))
                for entry in json.loads(args.queries.read_text())
            ]
        sample_ids = {m.message_id for messages in samples for m in messages}
        if not queries or any(not anchors or not anchors <= sample_ids for _, anchors in queries):
            raise SystemExit(
                "Questions must have nonempty known anchors within the sampled messages."
            )
        admission = MemoryAdmission.for_role(
            cfg, ModelRole.EMBED, command="segmentation comparison embeddings"
        )
        reason = BackgroundGate.from_config(cfg, log=print).admit(admission)
        if reason:
            raise SystemExit(reason.line())
        embedder = build_text_provider(cfg)
        query_vectors = np.array(
            [
                embedder.embed_query(q, instruction=cfg.embedding.query_instruction)
                for q, _ in queries
            ]
        )
        report = {
            "method": (
                "source-grounded analyst-authored paraphrases"
                if args.queries
                else "constructed 8-word source excerpts"
            )
            + "; exact cosine + keyword rank fusion; no reranking",
            "timing_note": "First topical session includes model loading (and any download); subsequent sessions measure warm inference. Embedding/search timings exclude query embedding and are order-sensitive.",
            "topical_session_seconds": topical_session_seconds,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "sample_sha256": hashlib.sha256(
                json.dumps(
                    [[m.source_guid, m.text] for messages in samples for m in messages]
                ).encode()
            ).hexdigest(),
            "embedding": {
                "model": cfg.embedding.model,
                "revision": cfg.embedding.revision,
                "dim": cfg.embedding.dim,
            },
            "scope": "text-only sample; attachments excluded equally from both variants",
            "quality_equivalence_established": False,
            "sessions": len(samples),
            "messages": sum(map(len, samples)),
            "queries": len(queries),
            "topical_fallback_sessions": fallback,
            "parameters": cfg.segmentation.model_dump(mode="json"),
            "variants": {},
        }
        retrieved = {}
        for name, chunks in variants.items():
            texts = [
                "\n".join(
                    render_message_line(
                        m,
                        timezone=cfg.render.timezone,
                        attachment_snippet_chars=cfg.render.attachment_snippet_chars,
                    )
                    for m in chunk
                )
                for chunk in chunks
            ]
            started = time.perf_counter()
            vectors = []
            for text in texts:
                vectors.extend(embedder.embed_documents([text]))
            matrix = np.array(vectors)
            scores = query_vectors @ matrix.T
            import apsw

            fts = apsw.Connection(":memory:")
            fts.execute("CREATE VIRTUAL TABLE chunks USING fts5(text)")
            for idx, text in enumerate(texts):
                fts.execute("INSERT INTO chunks(rowid,text) VALUES(?,?)", (idx + 1, text))
            hits = 0
            recall = []
            returned_words = []
            retrieved[name] = []
            for (query, anchor), row in zip(queries, scores, strict=True):
                expression = " ".join(
                    '"' + w.replace('"', '""') + '"' for w in re.findall(r"\w+", query)
                )
                keywords = [
                    r[0] - 1
                    for r in fts.execute(
                        "SELECT rowid FROM chunks WHERE chunks MATCH ? ORDER BY rank", (expression,)
                    )
                ]
                semantic = np.argsort(-row).tolist()
                fused = {}
                for ranking in [keywords, semantic]:
                    for rank, idx in enumerate(ranking):
                        fused[idx] = fused.get(idx, 0) + 1 / (61 + rank)
                best = sorted(fused, key=lambda idx: (-fused[idx], idx))[:3]
                ids = {m.message_id for idx in best for m in chunks[idx]}
                hits += anchor <= ids
                recall.append(len(anchor & ids) / len(anchor))
                returned_words.append(sum(len(texts[idx].split()) for idx in best))
                retrieved[name].append(ids)
            report["variants"][name] = {
                "chunks": len(chunks),
                "segmentation_seconds": times[name],
                "embedding_and_search_seconds": time.perf_counter() - started,
                "source_rediscovery_at_3": hits / len(queries) if queries else None,
                "indexed_message_appearances": sum(map(len, chunks)),
                "mean_support_message_recall_at_3": sum(recall) / len(recall),
                "mean_returned_messages_at_3": sum(map(len, retrieved[name])) / len(queries),
                "mean_returned_words_at_3": sum(returned_words) / len(returned_words),
            }
            print(name, report["variants"][name], flush=True)
        overlaps = [
            len(a & b) / len(a | b) if a | b else 1
            for a, b in zip(retrieved["topical"], retrieved["simple"], strict=True)
        ]
        report["mean_top3_message_jaccard"] = sum(overlaps) / len(overlaps) if overlaps else None
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print("Aggregate report:", output)


if __name__ == "__main__":
    main()
