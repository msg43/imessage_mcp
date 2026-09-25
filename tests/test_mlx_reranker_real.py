"""The reranker's shared-prefix path on real MLX kernels, against the
whole-row computation it replaces (`imsg.retrieval.mlx_reranker`, module
docstring "Shared prefix").

`test_mlx_reranker.py` checks what the provider asks the runtime for;
this module checks that real attention, rotary positions and key/value
caches give the same scores. It runs only where the `models` extra is
installed (it skips cleanly otherwise, as in CI):

- a tiny, randomly initialised Qwen3 built from `mlx_lm`'s own model
  class, in float32, always;
- the real Qwen3-Reranker-0.6B conversion when `IMSG_TEST_RERANKER_DIR`
  names its directory (the lock's `qwen3-reranker-0.6b-bf16` output_dir
  under a data root), in bf16 as loaded and cast to float32.

The reference is every row scored alone and whole
(`MlxRerankerProvider.score_token_rows([row])`), the model card's
computation; "before" is whole rows under 1,024 padded tokens, the path
before 2026-09-25. Every document is fictional.
"""

from __future__ import annotations

import os
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

mx = pytest.importorskip("mlx.core")
qwen3 = pytest.importorskip("mlx_lm.models.qwen3")

from imsg.embed.batching import plan_batches  # noqa: E402
from imsg.retrieval import mlx_reranker  # noqa: E402
from imsg.retrieval.mlx_reranker import MlxRerankerProvider, common_prefix_length  # noqa: E402

RERANKER_DIR_ENV = "IMSG_TEST_RERANKER_DIR"

QUERY = "when is the kite festival at the harbor this weekend"


def _segment(people: str, day: str, lines: Sequence[tuple[str, str, str]]) -> str:
    """A segment rendered the way `imsg.segment.render` renders one."""
    first, last = lines[0][0], lines[-1][0]
    head = [f"Chat: {people}", f"Time: {day} {first} \u2013 {last} America/Los_Angeles", "---"]
    return "\n".join([*head, *(f"[{t}] {who}: {text}" for t, who, text in lines)])


DOCUMENTS: tuple[str, ...] = (
    _segment(
        "Alice, Bob",
        "2026-03-12",
        [
            ("18:02", "Alice", "the kite festival moved to saturday at the harbor"),
            ("18:03", "Bob", "what time does it start?"),
            ("18:05", "Alice", "gates open at nine, the big kites go up around ten"),
        ],
    ),
    _segment(
        "Bob, Carol",
        "2026-03-10",
        [
            ("07:40", "Carol", "the ferry schedule changed, first boat is 7:15 now"),
            ("07:41", "Bob", "ugh ok, I'll leave earlier"),
        ],
    ),
    _segment("Alice", "2026-02-28", [("20:11", "Alice", "lemon cake: two lemons, one cup sugar")]),
    _segment(
        'Alice, Bob, Dave (group "Harbor crew")',
        "2026-03-13",
        [
            ("12:30", "Dave", "parking for the harbor kite festival fills up by ten"),
            ("12:31", "Bob", "let's meet at the ferry lot at 8:45 saturday"),
            ("12:33", "Alice", "I'll bring the blue kite and the spare string"),
            ("12:34", "Dave", "forecast says steady wind all morning"),
        ],
    ),
    _segment(
        "Dave",
        "2026-03-01",
        [("09:00", "Dave", "the bike shop says the new chain comes in thursday")],
    ),
    _segment(
        "Carol, Erin",
        "2026-03-05",
        [
            ("19:00", "Erin", f"choir rehearsal notes, part {n}: " + "hold the long note, " * 6)
            for n in range(1, 9)
        ],
    ),
    _segment(
        "Alice, Erin",
        "2026-03-08",
        [
            ("15:20", "Erin", "the library book sale is sunday, bring bags"),
            ("15:24", "Alice", "can we go after lunch?"),
        ],
    ),
    _segment(
        "Bob",
        "2026-03-11",
        [("10:10", "Bob", "picked up kite string and a reel at the hobby store")],
    ),
    _segment(
        "Alice, Bob",
        "2026-03-09",
        [
            ("21:00", "Bob", "weekend weather looks clear and windy"),
            ("21:02", "Alice", "perfect, bring jackets, it gets cold by the water"),
        ],
    ),
    _segment("Erin", "2026-02-20", [("11:45", "Erin", "pottery class moved to room 4")]),
    _segment(
        "Bob, Dave",
        "2026-03-02",
        [
            ("08:15", "Dave", "canoe rental is booked for the 21st"),
            ("08:20", "Bob", "nice, how many paddles?"),
            ("08:21", "Dave", "four, plus two life jackets for the kids"),
        ],
    ),
    _segment(
        "Alice, Carol",
        "2026-03-07",
        [
            ("13:00", "Carol", "the orchard trip is next weekend, not this one"),
            ("13:05", "Alice", "got it, I'll tell the others"),
        ],
    ),
    _segment(
        "Alice, Bob",
        "2026-03-14",
        [
            ("09:10", "Alice", "we're at the harbor, festival is packed already"),
            ("09:11", "Bob", "on the ferry, ten minutes out"),
        ],
    ),
    _segment(
        "Carol",
        "2026-03-03",
        [("17:30", "Carol", "board game night at my place friday, bring snacks")],
    ),
)


def _reference(provider: MlxRerankerProvider, rows: list[list[int]]) -> list[float]:
    """Every row alone and whole: no batch, no padding, no cache."""
    return [provider.score_token_rows([row])[0] for row in rows]


def _before(provider: MlxRerankerProvider, rows: list[list[int]]) -> list[float]:
    """Whole rows planned under 1,024 padded tokens: the path before
    2026-09-25."""
    scores = [0.0] * len(rows)
    plan = plan_batches([len(r) for r in rows], max_batch_size=32, max_batch_tokens=1024)
    for group in plan:
        for index, value in zip(
            group, provider.score_token_rows([rows[i] for i in group]), strict=True
        ):
            scores[index] = value
    return scores


def _common_start(rows: list[list[int]]) -> int:
    """The tokens every row starts with, leaving each row its last token —
    worked out here independently of the provider."""
    shared = min(common_prefix_length(rows[0], row) for row in rows)
    return min(shared, min(len(row) for row in rows) - 1)


def _misordered_pairs(
    reference: list[float], scores: list[float], gap: float
) -> list[tuple[int, int]]:
    """Pairs the reference separates by more than `gap` that `scores`
    orders the other way."""
    n = len(reference)
    return [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if abs(reference[i] - reference[j]) > gap
        and (reference[i] - reference[j]) * (scores[i] - scores[j]) < 0
    ]


class _RecordingModel:
    """A loaded `mlx_lm` model whose base transformer records every call as
    (input shape, tokens its caches already held or None without caches)."""

    def __init__(self, model: Any) -> None:
        self.args = model.args
        self.layers = model.layers
        self._inner = model.model
        self.calls: list[tuple[tuple[int, ...], int | None]] = []

    @property
    def model(self) -> _RecordingModel:
        return self

    @property
    def embed_tokens(self) -> Any:
        return self._inner.embed_tokens

    def __call__(self, inputs: Any, cache: list[Any] | None = None) -> Any:
        offset = None if cache is None else int(cache[0].offset)
        self.calls.append((tuple(int(d) for d in inputs.shape), offset))
        return self._inner(inputs, cache=cache)


# --------------------------------------------------------------------------
# a tiny random Qwen3, float32: the mechanics, exactly
# --------------------------------------------------------------------------

TINY_VOCAB = 512


class _WordTokenizer:
    """Whitespace words hashed into the tiny vocabulary; "yes" and "no" are
    single tokens, as in the Qwen3 tokenizer."""

    pad_token_id = 0
    unk_token_id = None

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [self._id(word) for word in text.split()]

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"yes": 1, "no": 2}.get(token, self._id(token))

    @staticmethod
    def _id(word: str) -> int:
        return 3 + zlib.crc32(word.encode("utf-8")) % (TINY_VOCAB - 3)


def _tiny_provider(
    monkeypatch: pytest.MonkeyPatch, **options: Any
) -> tuple[MlxRerankerProvider, _RecordingModel]:
    mx.random.seed(20260925)
    args = qwen3.ModelArgs(
        model_type="qwen3",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=TINY_VOCAB,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        rope_theta=1_000_000.0,
        head_dim=16,
        tie_word_embeddings=True,
    )
    model = qwen3.Model(args)
    mx.eval(model.parameters())
    recording = _RecordingModel(model)
    monkeypatch.setattr(
        mlx_reranker,
        "load_model_and_tokenizer",
        lambda repo, revision: (recording, _WordTokenizer()),
    )
    provider = MlxRerankerProvider("tiny-qwen3", None, cache_limit_bytes=None, **options)
    return provider, recording


def test_tiny_qwen3_shared_prefix_path_equals_scoring_each_row_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, recording = _tiny_provider(monkeypatch, doc_max_tokens=48, max_batch_tokens=400)
    rows = provider.token_rows(QUERY, list(DOCUMENTS))
    reference = _reference(provider, rows)
    recording.calls.clear()

    scores = provider.score(QUERY, list(DOCUMENTS))

    shared = _common_start(rows)
    assert shared > 40  # the chat prefix, the instruction and the query
    # One pass over the prefix into fresh caches, then the rests, which
    # continue after it; under 400 tokens that takes several passes.
    assert recording.calls[0] == ((1, shared), 0)
    assert len(recording.calls) > 2
    assert all(offset == shared for _, offset in recording.calls[1:])
    assert sum(shape[0] for shape, _ in recording.calls[1:]) == len(rows)
    # float32 rounding only: measured at most 1.2e-7 here (2026-09-25).
    assert max(abs(a - b) for a, b in zip(scores, reference, strict=True)) <= 1e-5
    assert _misordered_pairs(reference, scores, gap=2e-5) == []
    # The scores are spread out, so a wrong ordering would show.
    assert max(reference) - min(reference) > 0.1


def test_tiny_qwen3_without_reuse_prefix_reads_whole_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, recording = _tiny_provider(monkeypatch, doc_max_tokens=48, reuse_prefix=False)
    rows = provider.token_rows(QUERY, list(DOCUMENTS))
    reference = _reference(provider, rows)
    recording.calls.clear()
    scores = provider.score(QUERY, list(DOCUMENTS))
    assert all(offset is None for _, offset in recording.calls)
    assert max(abs(a - b) for a, b in zip(scores, reference, strict=True)) <= 1e-5


# --------------------------------------------------------------------------
# the real Qwen3-Reranker-0.6B conversion, when its directory is given
# --------------------------------------------------------------------------


def _loaded(directory: str) -> MlxRerankerProvider:
    provider = MlxRerankerProvider(
        directory, None, doc_max_tokens=256, cache_limit_bytes=None, model_id="test-reranker"
    )
    provider.load()
    return provider


@pytest.fixture
def real_provider() -> MlxRerankerProvider:
    """A fresh load for each test (under a second from a warm page cache),
    so one test casting the weights cannot change another's."""
    directory = os.environ.get(RERANKER_DIR_ENV)
    if not directory or not Path(directory).is_dir():
        pytest.skip(f"set {RERANKER_DIR_ENV} to a Qwen3-Reranker-0.6B MLX conversion to run")
    return _loaded(directory)


def test_real_reranker_bf16_new_path_stays_within_the_old_paths_rounding(
    real_provider: MlxRerankerProvider,
) -> None:
    """bf16 as loaded, the production setting. Batch shapes change bf16
    rounding: on the 45 benchmark pools of 20 (2026-09-24 pools, measured
    2026-09-25) the path before 2026-09-25 differed from scoring each row
    alone by up to 0.070 in P(yes), and the new path by up to 0.061; on
    this set by 0.0061 and 0.0125. Both are held to 0.1, and the ranking to
    the reference's wherever the reference separates two documents by more
    than twice that."""
    rows = real_provider.token_rows(QUERY, list(DOCUMENTS))
    reference = _reference(real_provider, rows)
    before = _before(real_provider, rows)
    recording = _RecordingModel(real_provider._model)
    real_provider._model = recording  # the provider has no public handle on its weights
    scores = real_provider.score(QUERY, list(DOCUMENTS))

    # The shared prefix, read once: the chat prefix, the instruction, the
    # query, "<Document>:" and the segment header's opening "Chat:".
    shared = _common_start(rows)
    assert shared > 60
    assert recording.calls[0] == ((1, shared), 0)
    assert [offset for _, offset in recording.calls[1:]] == [shared] * (len(recording.calls) - 1)
    assert max(abs(a - b) for a, b in zip(before, reference, strict=True)) <= 0.1
    assert max(abs(a - b) for a, b in zip(scores, reference, strict=True)) <= 0.1
    assert _misordered_pairs(reference, scores, gap=0.2) == []
    # The three segments about the festival, at 0.88-0.996 against at
    # most 0.002 for the rest (measured).
    top = sorted(range(len(scores)), key=lambda i: -scores[i])[:3]
    assert set(top) == {0, 3, 12}


def test_real_reranker_float32_new_path_equals_scoring_each_row_whole(
    real_provider: MlxRerankerProvider,
) -> None:
    """The same weights cast to float32, where rounding is small enough to
    see that the two computations are the same: the new path differed from
    scoring each row alone by at most 1e-5 on the 45 benchmark pools and
    3.3e-6 on this set (measured 2026-09-25), with no pair ordered
    differently."""
    provider = real_provider
    provider._model.set_dtype(mx.float32)  # the provider has no public handle on its weights
    rows = provider.token_rows(QUERY, list(DOCUMENTS))
    reference = _reference(provider, rows)
    scores = provider.score(QUERY, list(DOCUMENTS))
    assert max(abs(a - b) for a, b in zip(scores, reference, strict=True)) <= 1e-4
    assert _misordered_pairs(reference, scores, gap=2e-4) == []
