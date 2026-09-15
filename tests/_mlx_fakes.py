"""Dependency-free stand-ins for the MLX runtime (``mlx.core`` and
``mlx_lm``) used by the ``test_mlx_*`` suites.

The real runtime is never installed in this repository's test
environment (and never will be in CI), so the providers are exercised
against tiny fakes installed into ``sys.modules`` via ``monkeypatch``.
The fakes model only the surface the providers touch — ``mx.array`` /
``take_along_axis`` / ``take`` / ``eval`` / ``float32``, ``mlx_lm.load``
/ ``stream_generate`` / ``sample_utils.make_sampler`` / ``utils.
get_model_path``, and ``huggingface_hub.snapshot_download`` — and record
every call so tests can assert on what the providers asked for.

Fake numerics are deliberately hand-computable:

- :class:`FakeBaseTransformer` returns hidden states
  ``h[row][pos][k] = prefix_sum(row, pos) + k * pos`` — every position
  depends on all earlier tokens (causal) and on its own position, so
  reading the wrong position (e.g. a padding slot) is detectable.
- :class:`FakeLmHead` returns ``logits[v] = weight[v] * sum(state)``.
"""

from __future__ import annotations

import copy
import sys
import types
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

RUNTIME_MODULE_NAMES = (
    "mlx",
    "mlx.core",
    "mlx_lm",
    "mlx_lm.sample_utils",
    "mlx_lm.utils",
    "huggingface_hub",
)


class FakeArray:
    """A nested-list array with just ``shape`` / ``astype`` / ``tolist``."""

    def __init__(self, data: Any) -> None:
        self._data = copy.deepcopy(data)

    @property
    def shape(self) -> tuple[int, ...]:
        shape: list[int] = []
        cursor: Any = self._data
        while isinstance(cursor, list):
            shape.append(len(cursor))
            cursor = cursor[0] if cursor else None
        return tuple(shape)

    def astype(self, dtype: Any) -> FakeArray:
        return self

    def tolist(self) -> Any:
        return copy.deepcopy(self._data)


def _array(data: Any, dtype: Any = None) -> FakeArray:
    return FakeArray(data.tolist() if isinstance(data, FakeArray) else data)


def _take_along_axis(a: FakeArray, indices: FakeArray, axis: int) -> FakeArray:
    assert axis == 1, "the providers only gather along the sequence axis"
    rows = a.tolist()
    index = indices.tolist()
    return FakeArray([[rows[b][index[b][0][0]]] for b in range(len(rows))])


def _take(a: FakeArray, indices: FakeArray, axis: int) -> FakeArray:
    assert axis in (-1, 2), "the providers only take along the vocabulary axis"
    ids = indices.tolist()
    return FakeArray([[[row[0][i] for i in ids]] for row in a.tolist()])


def make_mx_module() -> types.ModuleType:
    module = types.ModuleType("mlx.core")
    module.array = _array  # type: ignore[attr-defined]
    module.take_along_axis = _take_along_axis  # type: ignore[attr-defined]
    module.take = _take  # type: ignore[attr-defined]
    module.eval = lambda *args: None  # type: ignore[attr-defined]
    module.float32 = "float32"  # type: ignore[attr-defined]
    module.int32 = "int32"  # type: ignore[attr-defined]
    return module


def prefix_sums(row: Sequence[int]) -> list[int]:
    total = 0
    out: list[int] = []
    for token in row:
        total += token
        out.append(total)
    return out


def fake_hidden_state(prefix_sum: int, position: int, hidden_size: int) -> list[float]:
    """What :class:`FakeBaseTransformer` emits at one position."""
    return [float(prefix_sum + k * position) for k in range(hidden_size)]


class FakeBaseTransformer:
    """Stands in for ``model.model`` (the base transformer)."""

    def __init__(self, hidden_size: int) -> None:
        self.hidden_size = hidden_size
        self.calls: list[list[list[int]]] = []
        self.embed_tokens = SimpleNamespace(as_linear=None)

    def __call__(self, ids: FakeArray) -> FakeArray:
        rows = ids.tolist()
        self.calls.append(rows)
        out = []
        for row in rows:
            sums = prefix_sums(row)
            out.append([fake_hidden_state(sums[p], p, self.hidden_size) for p in range(len(row))])
        return FakeArray(out)


class FakeLmHead:
    """Stands in for ``model.lm_head`` / ``embed_tokens.as_linear``:
    ``logits[v] = weights.get(v, 0.0) * sum(state)``."""

    def __init__(self, vocab_size: int, weights: dict[int, float]) -> None:
        self.vocab_size = vocab_size
        self.weights = dict(weights)
        self.calls = 0

    def __call__(self, states: FakeArray) -> FakeArray:
        self.calls += 1
        out = []
        for row in states.tolist():  # (batch, 1, hidden)
            total = sum(row[0])
            out.append([[self.weights.get(v, 0.0) * total for v in range(self.vocab_size)]])
        return FakeArray(out)


class FakeModel:
    def __init__(
        self,
        *,
        hidden_size: int = 3,
        tie_word_embeddings: bool = False,
        vocab_size: int = 8,
        head_weights: dict[int, float] | None = None,
    ) -> None:
        self.args = SimpleNamespace(
            hidden_size=hidden_size, tie_word_embeddings=tie_word_embeddings
        )
        self.model = FakeBaseTransformer(hidden_size)
        self.head = FakeLmHead(vocab_size, head_weights or {})
        if tie_word_embeddings:
            self.model.embed_tokens = SimpleNamespace(as_linear=self.head)
        else:
            self.lm_head = self.head


class FakeTokenizer:
    """Whitespace tokenizer: a word maps to ``vocab[word]`` when listed,
    else to ``len(word)``. ``add_special_tokens=True`` appends
    ``eos_suffix`` (the post-processor template's special tokens); the
    default ids mirror Qwen3-Embedding's (``<|endoftext|>`` 151643 as the
    appended EOS and pad, ``<|im_end|>`` 151645 as ``eos_token_id``)."""

    YES_ID = 5
    NO_ID = 6

    def __init__(
        self,
        *,
        vocab: dict[str, int] | None = None,
        eos_suffix: Sequence[int] = (151643,),
        eos_token_id: int | None = 151645,
        pad_token_id: int | None = 151643,
        unk_token_id: int | None = None,
        chat_template: str | None = "{{ messages }}",
        special_ids: dict[str, int | None] | None = None,
    ) -> None:
        self.vocab = dict(vocab or {})
        self.eos_suffix = list(eos_suffix)
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.chat_template = chat_template
        self.special_ids: dict[str, int | None] = {"yes": self.YES_ID, "no": self.NO_ID}
        if special_ids:
            self.special_ids.update(special_ids)
        self.encode_calls: list[tuple[str, bool]] = []
        self.chat_calls: list[dict[str, Any]] = []
        self.chat_error: Exception | None = None

    def token_id(self, word: str) -> int:
        return self.vocab.get(word, len(word))

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        self.encode_calls.append((text, add_special_tokens))
        ids = [self.token_id(word) for word in text.split()]
        if add_special_tokens:
            ids.extend(self.eos_suffix)
        return ids

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self.special_ids.get(token)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        **kwargs: Any,
    ) -> str | list[int]:
        self.chat_calls.append(
            {
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
                "tokenize": tokenize,
                **kwargs,
            }
        )
        if self.chat_error is not None:
            raise self.chat_error
        rendered = f"<|user|>{messages[-1]['content']}<|assistant|>"
        return self.encode(rendered, add_special_tokens=False) if tokenize else rendered


class FakeRuntime:
    """Builds and installs fake ``mlx`` / ``mlx_lm`` / ``huggingface_hub``
    modules, recording every call the providers make."""

    def __init__(
        self,
        *,
        model: FakeModel | None = None,
        tokenizer: FakeTokenizer | None = None,
        load_supports_revision: bool = True,
        expose_get_model_path: bool = True,
        stream_chunks: Sequence[str | Exception] = (),
        stream_hook: Callable[[int], None] | None = None,
        load_error: Exception | None = None,
    ) -> None:
        self.model = model or FakeModel()
        self.tokenizer = tokenizer or FakeTokenizer()
        self.load_supports_revision = load_supports_revision
        self.expose_get_model_path = expose_get_model_path
        self.stream_chunks = list(stream_chunks)
        self.stream_hook = stream_hook
        self.load_error = load_error
        self.load_calls: list[dict[str, Any]] = []
        self.get_model_path_calls: list[tuple[str, str | None]] = []
        self.snapshot_calls: list[dict[str, Any]] = []
        self.sampler_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    # -- mlx_lm.load ---------------------------------------------------------

    def _load_with_revision(
        self,
        path_or_hf_repo: str,
        tokenizer_config: dict[str, Any] | None = None,
        model_config: dict[str, Any] | None = None,
        revision: str | None = None,
        **kwargs: Any,
    ) -> tuple[FakeModel, FakeTokenizer]:
        return self._record_load(path_or_hf_repo, tokenizer_config, model_config, revision)

    def _load_legacy(
        self,
        path_or_hf_repo: str,
        tokenizer_config: dict[str, Any] | None = None,
        model_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[FakeModel, FakeTokenizer]:
        return self._record_load(path_or_hf_repo, tokenizer_config, model_config, None)

    def _record_load(
        self,
        path: str,
        tokenizer_config: dict[str, Any] | None,
        model_config: dict[str, Any] | None,
        revision: str | None,
    ) -> tuple[FakeModel, FakeTokenizer]:
        self.load_calls.append(
            {
                "path": path,
                "tokenizer_config": tokenizer_config,
                "model_config": model_config,
                "revision": revision,
            }
        )
        if self.load_error is not None:
            raise self.load_error
        return self.model, self.tokenizer

    # -- snapshot resolution -------------------------------------------------

    def _get_model_path(
        self, path_or_hf_repo: str, revision: str | None = None
    ) -> tuple[Path, str]:
        self.get_model_path_calls.append((path_or_hf_repo, revision))
        return Path("/fake/snapshots") / path_or_hf_repo / (revision or "main"), path_or_hf_repo

    def _snapshot_download(
        self, repo_id: str, revision: str | None = None, allow_patterns: Any = None
    ) -> str:
        self.snapshot_calls.append(
            {"repo_id": repo_id, "revision": revision, "allow_patterns": allow_patterns}
        )
        return str(Path("/fake/hub") / repo_id / (revision or "main"))

    # -- generation ----------------------------------------------------------

    def _make_sampler(self, temp: float = 0.0, **kwargs: Any) -> tuple[str, float]:
        self.sampler_calls.append({"temp": temp, **kwargs})
        return ("sampler", temp)

    def _stream_generate(
        self, model: Any, tokenizer: Any, prompt: Any, **kwargs: Any
    ) -> Iterator[SimpleNamespace]:
        self.stream_calls.append(
            {"model": model, "tokenizer": tokenizer, "prompt": prompt, **kwargs}
        )
        for index, chunk in enumerate(self.stream_chunks):
            if isinstance(chunk, Exception):
                raise chunk
            if self.stream_hook is not None:
                self.stream_hook(index)
            yield SimpleNamespace(text=chunk, token=index, finish_reason=None)

    # -- installation --------------------------------------------------------

    def install(self, monkeypatch: pytest.MonkeyPatch) -> FakeRuntime:
        mx = make_mx_module()
        mlx = types.ModuleType("mlx")
        mlx.core = mx  # type: ignore[attr-defined]
        mlx_lm = types.ModuleType("mlx_lm")
        mlx_lm.load = (  # type: ignore[attr-defined]
            self._load_with_revision if self.load_supports_revision else self._load_legacy
        )
        mlx_lm.stream_generate = self._stream_generate  # type: ignore[attr-defined]
        sample_utils = types.ModuleType("mlx_lm.sample_utils")
        sample_utils.make_sampler = self._make_sampler  # type: ignore[attr-defined]
        utils = types.ModuleType("mlx_lm.utils")
        if self.expose_get_model_path:
            utils.get_model_path = self._get_model_path  # type: ignore[attr-defined]
        mlx_lm.sample_utils = sample_utils  # type: ignore[attr-defined]
        mlx_lm.utils = utils  # type: ignore[attr-defined]
        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = self._snapshot_download  # type: ignore[attr-defined]
        modules = {
            "mlx": mlx,
            "mlx.core": mx,
            "mlx_lm": mlx_lm,
            "mlx_lm.sample_utils": sample_utils,
            "mlx_lm.utils": utils,
            "huggingface_hub": hub,
        }
        for name, module in modules.items():
            monkeypatch.setitem(sys.modules, name, module)
        return self


def uninstall_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every runtime module un-importable (``None`` in ``sys.modules``
    makes ``import`` raise ``ModuleNotFoundError``), regardless of what
    the host has installed."""
    for name in RUNTIME_MODULE_NAMES:
        monkeypatch.setitem(sys.modules, name, None)
