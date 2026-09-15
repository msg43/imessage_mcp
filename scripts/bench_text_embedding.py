#!/usr/bin/env python3
"""Benchmark the MLX text embedder's batching strategies on SYNTHETIC text.

Times `MlxTextEmbeddingProvider.embed_documents` over batches of
word-salad texts whose token lengths follow a length distribution —
the real corpus's (`--dsn`: `SELECT token_count FROM segment`, a
statistic held in memory and never written anywhere) or a log-normal
stand-in — so the numbers reflect the padding the real corpus induces
without a single byte of real message text leaving Postgres.

Usage (the project venv with the `models` extra):

    python scripts/bench_text_embedding.py --dsn postgresql://user@host:port/db \\
        --sample 256 --configs all --report-json /path/to/report.json

Each configuration is a way of grouping the same sampled texts into
batches; the provider is built to pass every batch through as one
forward pass, so what is measured is exactly the padding each strategy
pays:

- `fixed<n>-random`  — batches of `n` in random order (the pipeline's
  behaviour before 2026-09-15: `segment_id` order is random with
  respect to length);
- `fixed<n>-sorted`  — batches of `n`, longest rows first;
- `budget<k>k-sorted` — length-sorted batches capped at `k x 1024`
  padded tokens (`rows x longest row`), any row count;
- `maxlen2048:<config>` — the same with the provider's `max_length` at
  2048 instead of 8192 (a second weight load), to show whether the
  provider pads to `max_length` or to the longest row in the batch.

Reported per configuration: batches, real and padded tokens, seconds,
real and padded tokens per second, texts per minute, MLX peak memory
and the buffer cache left behind (`mx.get_cache_memory`). `--cache-
limit-gb` applies `mx.set_cache_limit` first, to compare against the
runtime default (equal to the memory limit, 121.6 GiB on a 128 GB M2
Ultra). `--include-longest K` adds the K longest lengths of the
distribution to the sample, which is how a single 8k-token row lands in
a random batch.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from imsg import constants
from imsg.embed.batching import padded_tokens, plan_batches
from imsg.embed.mlx_text import MlxTextEmbeddingProvider

WORDS = (  # noqa: SIM905 — a word list reads better as prose than as 200 quoted items
    "the of and to in a is that for it as was with be by on not he this are or his from at "
    "which but have an had they you were their one all we can her has there been if more "
    "when will would who so no she what up out about into than them time only some could "
    "these two may then do first any my now such like our over man me even most made after "
    "also did many before must through back years where much your way well down should "
    "because each just those people how too little state good very make world still own see "
    "men work long get here between both life being under never day same another know while "
    "last might us great old year off come since against go came right used take three "
    "harbor kite festival gate opens nine blue drifted quiet dawn thanks sounds good okay "
    "tomorrow tonight later maybe dinner running late see you soon call me haha lol yes no"
).split()
PUNCTUATION = (".", ",", "!", "?", "\n", " -", "...")

DEFAULT_CONFIGS = (
    "fixed32-random",
    "fixed8-random",
    "fixed16-random",
    "fixed64-random",
    "fixed8-sorted",
    "fixed16-sorted",
    "fixed32-sorted",
    "fixed64-sorted",
    "budget4k-sorted",
    "budget8k-sorted",
    "budget16k-sorted",
    "budget32k-sorted",
    "maxlen2048:fixed32-random",
)

Planner = Callable[[Sequence[int], random.Random], list[list[int]]]


# --------------------------------------------------------------------------
# length distributions
# --------------------------------------------------------------------------


def lengths_from_postgres(dsn: str) -> list[int]:
    """Every `segment.token_count` plus every non-null
    `attachment_chunk.token_count` — a statistic, kept in memory only."""
    import psycopg

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT token_count FROM segment "
            "UNION ALL SELECT token_count FROM attachment_chunk WHERE token_count IS NOT NULL"
        )
        return [int(n) for (n,) in cur.fetchall() if n is not None and n > 0]


def synthetic_lengths(count: int, rng: random.Random, *, median: float, sigma: float) -> list[int]:
    """A log-normal stand-in for a chat corpus: most segments are a few
    dozen tokens, with a long tail."""
    return [max(1, round(rng.lognormvariate(math.log(median), sigma))) for _ in range(count)]


def summarize(values: Sequence[int]) -> dict[str, float]:
    ordered = sorted(values)

    def pct(p: float) -> float:
        return float(ordered[min(len(ordered) - 1, int(p * len(ordered)))])

    return {
        "count": len(ordered),
        "min": float(ordered[0]),
        "median": float(statistics.median(ordered)),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "max": float(ordered[-1]),
        "mean": float(statistics.fmean(ordered)),
        "sum": float(sum(ordered)),
    }


# --------------------------------------------------------------------------
# synthetic text of a given token length
# --------------------------------------------------------------------------


def word_salad(rng: random.Random, words: int) -> str:
    out: list[str] = []
    for _ in range(words):
        out.append(rng.choice(WORDS))
        if rng.random() < 0.12:
            out.append(rng.choice(PUNCTUATION))
    return " ".join(out).replace(" \n ", "\n").replace(" .", ".").replace(" ,", ",")


class SyntheticTexts:
    """Texts whose provider-measured token length (EOS included) hits a
    target: slices of one long tokenized word-salad stream, decoded back
    to text, then nudged by whole words until the count matches."""

    def __init__(self, provider: MlxTextEmbeddingProvider, rng: random.Random) -> None:
        self.provider = provider
        self.rng = rng
        tokenizer = provider.tokenizer
        self._tokenizer = tokenizer
        stream = word_salad(rng, 40_000)
        self._ids = [int(t) for t in tokenizer.encode(stream, add_special_tokens=False)]
        if len(self._ids) < provider.max_length + 64:
            raise RuntimeError("synthetic stream is shorter than max_length; enlarge word_salad")

    def make(self, target: int) -> tuple[str, int]:
        target = max(2, min(target, self.provider.max_length))
        body = target - 1
        start = self.rng.randrange(0, len(self._ids) - body)
        text = str(self._tokenizer.decode(self._ids[start : start + body]))
        measured = self.provider.token_length(text)
        for _ in range(8):
            if measured == target:
                break
            if measured < target:
                text = f"{text} {self.rng.choice(WORDS)}"
            else:
                text = text.rsplit(" ", 1)[0] if " " in text else text[:-1]
            measured = self.provider.token_length(text)
        return text, measured


# --------------------------------------------------------------------------
# batch planners
# --------------------------------------------------------------------------


def fixed_random(size: int) -> Planner:
    def plan(lengths: Sequence[int], rng: random.Random) -> list[list[int]]:
        order = list(range(len(lengths)))
        rng.shuffle(order)
        return [order[i : i + size] for i in range(0, len(order), size)]

    return plan


def fixed_sorted(size: int) -> Planner:
    def plan(lengths: Sequence[int], rng: random.Random) -> list[list[int]]:
        return plan_batches(lengths, max_batch_size=size, max_batch_tokens=1 << 40)

    return plan


def budget_sorted(tokens: int) -> Planner:
    def plan(lengths: Sequence[int], rng: random.Random) -> list[list[int]]:
        return plan_batches(lengths, max_batch_size=4096, max_batch_tokens=tokens)

    return plan


def parse_config(name: str) -> tuple[str, int, Planner]:
    """`(base name, max_length, planner)` for a configuration name."""
    max_length = 8192
    base = name
    if name.startswith("maxlen"):
        prefix, _, base = name.partition(":")
        max_length = int(prefix[len("maxlen") :])
    if base.startswith("fixed") and base.endswith("-random"):
        return base, max_length, fixed_random(int(base[len("fixed") : -len("-random")]))
    if base.startswith("fixed") and base.endswith("-sorted"):
        return base, max_length, fixed_sorted(int(base[len("fixed") : -len("-sorted")]))
    if base.startswith("budget") and base.endswith("k-sorted"):
        return base, max_length, budget_sorted(int(base[len("budget") : -len("k-sorted")]) * 1024)
    raise SystemExit(f"unknown configuration {name!r}")


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------


def gib(n: float) -> float:
    return round(n / 2**30, 2)


def build_provider(args: argparse.Namespace, max_length: int) -> MlxTextEmbeddingProvider:
    provider = MlxTextEmbeddingProvider(
        args.model,
        args.revision,
        constants.PRIMARY_EMBEDDING_DIM,
        batch_size=1 << 20,  # the benchmark decides the batches; pass each through whole
        max_length=max_length,
        max_batch_tokens=1 << 40,
        cache_limit_bytes=None,
    )
    t0 = time.perf_counter()
    provider.load()
    print(
        f"loaded {provider.model_id} (max_length={max_length}) in {time.perf_counter() - t0:.1f}s"
    )
    return provider


def run_config(
    mx: Any,
    provider: MlxTextEmbeddingProvider,
    name: str,
    planner: Planner,
    texts: Sequence[str],
    lengths: Sequence[int],
    rng: random.Random,
    *,
    verbose: bool,
) -> dict[str, Any]:
    batches = planner(lengths, rng)
    real = sum(lengths)
    padded = padded_tokens(lengths, batches)
    mx.clear_cache()
    mx.reset_peak_memory()
    per_batch: list[dict[str, float]] = []
    t0 = time.perf_counter()
    for batch in batches:
        width = max(lengths[i] for i in batch)
        b0 = time.perf_counter()
        vectors = provider.embed_documents([texts[i] for i in batch])
        seconds = time.perf_counter() - b0
        if len(vectors) != len(batch):
            raise RuntimeError(f"{name}: {len(vectors)} vectors for {len(batch)} texts")
        row = {
            "rows": len(batch),
            "width": width,
            "padded": len(batch) * width,
            "seconds": round(seconds, 3),
            "cache_gib": gib(mx.get_cache_memory()),
        }
        per_batch.append(row)
        if verbose:
            print(
                f"  {name}: rows={len(batch):4d} width={width:5d} padded={len(batch) * width:7d} "
                f"{seconds:7.2f}s {len(batch) * width / seconds:8.0f} padded tok/s "
                f"cache={row['cache_gib']:.1f}GiB"
            )
    total = time.perf_counter() - t0
    result = {
        "config": name,
        "max_length": provider.max_length,
        "batches": len(batches),
        "texts": len(texts),
        "real_tokens": real,
        "padded_tokens": padded,
        "padding_ratio": round(padded / real, 2),
        "seconds": round(total, 2),
        "real_tok_per_s": round(real / total, 1),
        "padded_tok_per_s": round(padded / total, 1),
        "texts_per_min": round(60 * len(texts) / total, 1),
        "peak_gib": gib(mx.get_peak_memory()),
        "active_gib_after": gib(mx.get_active_memory()),
        "cache_gib_after": gib(mx.get_cache_memory()),
        "per_batch": per_batch,
    }
    print(
        f"{name:28s} batches={len(batches):3d} real={real:7d} padded={padded:8d} "
        f"({padded / real:4.1f}x) {total:7.1f}s  real {real / total:7.0f} tok/s  "
        f"padded {padded / total:7.0f} tok/s  {60 * len(texts) / total:6.0f} texts/min  "
        f"peak {result['peak_gib']:.1f} GiB  cache {result['cache_gib_after']:.1f} GiB"
    )
    return result


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def compare_embeddings(
    args: argparse.Namespace, reference: tuple[list[str], list[list[float]]]
) -> dict[str, Any]:
    """Embed the reference texts with the second checkpoint and report
    the text-by-text cosine similarity against the first's vectors."""
    texts, expected = reference
    other = MlxTextEmbeddingProvider(
        args.similarity_against,
        args.similarity_revision,
        constants.PRIMARY_EMBEDDING_DIM,
        batch_size=1 << 20,
        max_length=args.max_length,
        max_batch_tokens=1 << 40,
        cache_limit_bytes=None,
    )
    t0 = time.perf_counter()
    other.load()
    load_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    actual = other.embed_documents(texts)
    embed_seconds = time.perf_counter() - t0
    sims = [cosine(a, b) for a, b in zip(expected, actual, strict=True)]
    return {
        "reference_model": f"{args.model}@{args.revision or 'main'}",
        "other_model": f"{args.similarity_against}@{args.similarity_revision or 'main'}",
        "texts": len(sims),
        "min": min(sims),
        "mean": statistics.fmean(sims),
        "max": max(sims),
        "other_load_seconds": round(load_seconds, 2),
        "other_embed_seconds": round(embed_seconds, 2),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dsn", help="Postgres DSN to read token_count statistics from")
    parser.add_argument("--sample", type=int, default=256, help="texts per configuration")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=constants.TEXT_EMBEDDING_MODEL_REPO)
    parser.add_argument(
        "--revision",
        default=constants.TEXT_EMBEDDING_MODEL_REVISION,
        help="commit sha for a Hub repo; 'none' for a local directory",
    )
    parser.add_argument(
        "--similarity-against",
        metavar="MODEL",
        help="a second checkpoint (repo id or local directory) to embed the first "
        "--similarity-texts synthetic texts with; reports the cosine similarity of "
        "its vectors against --model's, text by text",
    )
    parser.add_argument("--similarity-revision", default="none")
    parser.add_argument("--similarity-texts", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument(
        "--configs", default="all", help="comma-separated configuration names, or 'all'"
    )
    parser.add_argument(
        "--cache-limit-gb", type=float, default=None, help="mx.set_cache_limit before measuring"
    )
    parser.add_argument(
        "--include-longest", type=int, default=0, help="add the K longest lengths to the sample"
    )
    parser.add_argument("--synthetic-median", type=float, default=96.0)
    parser.add_argument("--synthetic-sigma", type=float, default=1.1)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--quiet", action="store_true", help="no per-batch lines")
    args = parser.parse_args(argv)
    if args.revision == "none":
        args.revision = None
    if args.similarity_revision == "none":
        args.similarity_revision = None

    import mlx.core as mx

    rng = random.Random(args.seed)
    if args.dsn:
        population = lengths_from_postgres(args.dsn)
        source = "postgres token_count"
    else:
        population = synthetic_lengths(
            100_000, rng, median=args.synthetic_median, sigma=args.synthetic_sigma
        )
        source = "log-normal stand-in"
    targets = rng.choices(population, k=args.sample)
    if args.include_longest:
        targets.extend(sorted(population, reverse=True)[: args.include_longest])
    print(f"length source: {source}; population {summarize(population)}")
    print(f"sample targets: {summarize(targets)}")

    if args.cache_limit_gb is not None:
        previous = mx.set_cache_limit(int(args.cache_limit_gb * 2**30))
        print(f"mx.set_cache_limit({args.cache_limit_gb} GiB); previous {gib(previous)} GiB")

    names = list(DEFAULT_CONFIGS) if args.configs == "all" else args.configs.split(",")
    parsed = [(name, *parse_config(name)) for name in names]
    max_lengths = sorted({max_length for _, _, max_length, _ in parsed}, reverse=True)

    results: list[dict[str, Any]] = []
    reference: tuple[list[str], list[list[float]]] | None = None
    for max_length in max_lengths:
        provider = build_provider(args, max_length)
        synth = SyntheticTexts(provider, random.Random(args.seed + 1))
        texts: list[str] = []
        lengths: list[int] = []
        for target in targets:
            text, measured = synth.make(target)
            texts.append(text)
            lengths.append(measured)
        print(f"synthetic texts (max_length={max_length}): measured {summarize(lengths)}")
        # warm-up: compile kernels / first-touch allocations outside the timings
        provider.embed_documents(texts[:4])
        if args.similarity_against and reference is None:
            subset = texts[: args.similarity_texts]
            reference = (subset, provider.embed_documents(subset))
        for name, _base, config_max_length, planner in parsed:
            if config_max_length != max_length:
                continue
            results.append(
                run_config(
                    mx,
                    provider,
                    name,
                    planner,
                    texts,
                    lengths,
                    random.Random(args.seed + 2),
                    verbose=not args.quiet,
                )
            )
        del provider, synth
        gc.collect()
        mx.clear_cache()

    similarity: dict[str, Any] | None = None
    if args.similarity_against and reference is not None:
        similarity = compare_embeddings(args, reference)
        print(
            f"cosine({args.model} vs {args.similarity_against}) over {similarity['texts']} texts: "
            f"min={similarity['min']:.5f} mean={similarity['mean']:.5f} max={similarity['max']:.5f}"
        )

    if args.report_json:
        report = {
            "similarity": similarity,
            "length_source": source,
            "population": summarize(population),
            "sample": summarize(targets),
            "cache_limit_gb": args.cache_limit_gb,
            "model": args.model,
            "revision": args.revision,
            "results": results,
        }
        args.report_json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"report written to {args.report_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
