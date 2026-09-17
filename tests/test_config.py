"""Config validation tests (SPEC §6).

These are security-relevant: they exercise the enforcement mechanism
for CLAUDE.md non-negotiables #1, #2, and #6, plus the D6
`mcp.public.scope`-has-no-default rule and the secret-literal
rejection rule. Every rejection rule gets its own test — see the task
brief this build was written against.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from imsg.config.loader import load_config_dict
from imsg.config.schema import Config
from imsg.errors import ConfigError

pytestmark = pytest.mark.usefixtures("messages_dir")


# --------------------------------------------------------------------------
# Baseline: a valid config loads and round-trips sensible values
# --------------------------------------------------------------------------


def test_valid_config_loads(config_dict_factory: object) -> None:
    cfg = load_config_dict(config_dict_factory())  # type: ignore[operator]
    assert isinstance(cfg, Config)
    assert cfg.database.dsn.endswith(":5433/imsgindex")
    assert cfg.embedding.dim == 2048
    assert cfg.embedding.multimodal.dim == 1280
    assert cfg.mcp.public.scope == "allowlist"


def test_unknown_top_level_key_rejected(config_dict_factory: object) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["totally_unexpected_key"] = 1
    with pytest.raises(ConfigError, match=r"totally_unexpected_key|extra"):
        load_config_dict(raw)


def test_unknown_nested_key_rejected(config_dict_factory: object) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["paths"]["unexpected"] = "nope"
    with pytest.raises(ConfigError):
        load_config_dict(raw)


# --------------------------------------------------------------------------
# Hard requirement #1: exactly one path permitted under ~/Library/Messages
# --------------------------------------------------------------------------


def test_live_chat_db_must_resolve_under_messages_dir(
    config_dict_factory: object, tmp_path: Path
) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["paths"]["live_chat_db"] = str(tmp_path / "somewhere-else" / "chat.db")
    with pytest.raises(ConfigError, match="live_chat_db"):
        load_config_dict(raw)


def test_second_distinct_path_under_messages_dir_rejected(
    config_dict_factory: object, messages_dir: Path
) -> None:
    """A second, different file under ~/Library/Messages must be rejected
    even though it is a `sync.sources[].chat_db` entry, not a random
    output path — hard requirement #1 permits exactly one path there."""
    raw = config_dict_factory()  # type: ignore[operator]
    raw["sync"]["sources"].append(
        {"name": "second-mini", "chat_db": str(messages_dir / "another-chat.db")}
    )
    with pytest.raises(ConfigError, match="Messages"):
        load_config_dict(raw)


def test_data_root_under_messages_dir_rejected(
    config_dict_factory: object, messages_dir: Path
) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["paths"]["data_root"] = str(messages_dir / "derived")
    with pytest.raises(ConfigError, match="Messages"):
        load_config_dict(raw)


def test_mini_source_matching_live_chat_db_is_allowed(config_dict_factory: object) -> None:
    """The one declared exception: sync.sources[].chat_db == paths.live_chat_db is fine."""
    cfg = load_config_dict(config_dict_factory())  # type: ignore[operator]
    assert cfg.sync.sources[0].chat_db == cfg.paths.live_chat_db


# --------------------------------------------------------------------------
# Hard requirement #2: derived paths must resolve under data_root
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "database.cluster_fingerprint_file",
        "segmentation.boundary_prompt",
        "eval.seed_queries",
        "eval.runs_dir",
    ],
)
def test_derived_path_absolute_outside_data_root_rejected(
    config_dict_factory: object, field: str, tmp_path: Path
) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    section, key = field.split(".")
    raw[section][key] = str(tmp_path / "escaped" / "somewhere.txt")
    with pytest.raises(ConfigError, match="data_root"):
        load_config_dict(raw)


@pytest.mark.parametrize(
    "field",
    [
        "database.cluster_fingerprint_file",
        "segmentation.boundary_prompt",
        "eval.seed_queries",
        "eval.runs_dir",
    ],
)
def test_derived_path_dotdot_escape_rejected(config_dict_factory: object, field: str) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    section, key = field.split(".")
    raw[section][key] = "../../../etc/passwd"
    with pytest.raises(ConfigError, match="data_root"):
        load_config_dict(raw)


def test_derived_path_symlink_escape_rejected(config_dict_factory: object) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    data_root = Path(raw["paths"]["data_root"])
    data_root.mkdir(parents=True, exist_ok=True)
    outside = data_root.parent / "outside-eval-runs"
    outside.mkdir(exist_ok=True)
    (data_root / "eval").mkdir(exist_ok=True)
    (data_root / "eval" / "runs_link").symlink_to(outside)
    raw["eval"]["runs_dir"] = "eval/runs_link"
    with pytest.raises(ConfigError, match="data_root"):
        load_config_dict(raw)


def test_relative_derived_paths_are_fine(config_dict_factory: object) -> None:
    cfg = load_config_dict(config_dict_factory())  # type: ignore[operator]
    assert cfg.eval.runs_dir == Path("eval/runs")


# --------------------------------------------------------------------------
# Hard requirement #6: dedicated Postgres instance (port 5433)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://imsg@127.0.0.1:5432/imsgindex",  # wrong port (default PG port)
        "postgresql://imsg@127.0.0.1/imsgindex",  # missing port
        "not-a-postgres-uri-at-all",
        "mysql://imsg@127.0.0.1:5433/imsgindex",  # wrong scheme
    ],
)
def test_dsn_must_target_dedicated_instance(config_dict_factory: object, dsn: str) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["database"]["dsn"] = dsn
    with pytest.raises(ConfigError, match=r"5433|postgresql"):
        load_config_dict(raw)


# --------------------------------------------------------------------------
# Secrets: literals rejected, only keychain:/env: references accepted
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "literal",
    ["hunter2", "sk-abcdef123456", "postgresql://u:p@h/d", ""],
)
def test_database_password_literal_rejected(config_dict_factory: object, literal: str) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["database"]["password"] = literal
    with pytest.raises(ConfigError, match=r"keychain|env"):
        load_config_dict(raw)


def _enabled_public_mcp_overrides(messages_dir: Path) -> dict[str, object]:
    return {
        "enabled": True,
        "bind": "127.0.0.1:8700",
        "external_url": "https://example.invalid/mcp",
        "scope": "allowlist",
        "allowed_origins": ["https://vertexaisearch.cloud.google.com"],
        "allowed_hosts": ["example.invalid"],
        "oauth": {
            "issuer": "google",
            "client_id": "env:IMSG_OAUTH_CLIENT_ID",
            "client_secret": "keychain:imsgindex-oauth",
            "owner_subject": "keychain:imsgindex-owner-sub",
        },
    }


def test_oauth_client_secret_literal_rejected(config_dict_factory: object, messages_dir: Path) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    public = _enabled_public_mcp_overrides(messages_dir)
    public["oauth"]["client_secret"] = "literal-oauth-secret"  # type: ignore[index]
    raw["mcp"]["public"] = public
    with pytest.raises(ConfigError, match=r"keychain|env"):
        load_config_dict(raw)


def test_oauth_owner_subject_literal_rejected(config_dict_factory: object, messages_dir: Path) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    public = _enabled_public_mcp_overrides(messages_dir)
    public["oauth"]["owner_subject"] = "115792089237316195"  # type: ignore[index]
    raw["mcp"]["public"] = public
    with pytest.raises(ConfigError, match=r"keychain|env"):
        load_config_dict(raw)


# --------------------------------------------------------------------------
# mcp.public.scope: REQUIRED, no default (D6)
# --------------------------------------------------------------------------


def test_scope_omitted_is_a_validation_error(config_dict_factory: object) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    del raw["mcp"]["public"]["scope"]
    with pytest.raises(ConfigError, match="scope"):
        load_config_dict(raw)


def test_scope_never_silently_defaults_to_full(config_dict_factory: object) -> None:
    """Belt-and-braces: even if some future refactor added a default, the
    field must reject an omitted key rather than silently becoming 'full'."""
    raw = config_dict_factory()  # type: ignore[operator]
    del raw["mcp"]["public"]["scope"]
    try:
        cfg = load_config_dict(raw)
    except ConfigError:
        return  # correct: omission is rejected
    pytest.fail(
        f"omitting mcp.public.scope must be a validation error, not silently "
        f"resolve to {cfg.mcp.public.scope!r}"
    )


def test_scope_invalid_enum_value_rejected(config_dict_factory: object) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["mcp"]["public"]["scope"] = "everything"
    with pytest.raises(ConfigError):
        load_config_dict(raw)


def test_public_enabled_requires_external_url_and_origins_and_hosts(
    config_dict_factory: object, messages_dir: Path
) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["mcp"]["public"] = {"enabled": True, "scope": "allowlist"}
    with pytest.raises(ConfigError, match="external_url"):
        load_config_dict(raw)


def test_public_enabled_scope_full_requires_scope_approval_id(
    config_dict_factory: object, messages_dir: Path
) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    public = _enabled_public_mcp_overrides(messages_dir)
    public["scope"] = "full"
    raw["mcp"]["public"] = public
    with pytest.raises(ConfigError, match="scope_approval_id"):
        load_config_dict(raw)


def test_public_enabled_scope_full_with_approval_id_succeeds(
    config_dict_factory: object, messages_dir: Path
) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    public = _enabled_public_mcp_overrides(messages_dir)
    public["scope"] = "full"
    public["scope_approval_id"] = "ops-approval-2026-07-30"
    raw["mcp"]["public"] = public
    cfg = load_config_dict(raw)
    assert cfg.mcp.public.scope == "full"
    assert cfg.mcp.public.scope_approval_id == "ops-approval-2026-07-30"


def test_public_disabled_scope_full_without_approval_id_is_fine(
    config_dict_factory: object,
) -> None:
    """scope is always required, but the rest of the enabled=true bundle
    (external_url, approval id, ...) is only required once enabled=true —
    an operator may stage `scope: full` in config ahead of turning the
    surface on."""
    raw = config_dict_factory()  # type: ignore[operator]
    raw["mcp"]["public"]["scope"] = "full"
    cfg = load_config_dict(raw)
    assert cfg.mcp.public.enabled is False
    assert cfg.mcp.public.scope == "full"
    assert cfg.mcp.public.scope_approval_id is None


# --------------------------------------------------------------------------
# Embedding dims: pgvector index caps + must match the migration DDL
# --------------------------------------------------------------------------


def test_embedding_dim_must_match_migration_0001(config_dict_factory: object) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["embedding"]["dim"] = 1024
    with pytest.raises(ConfigError, match="2048"):
        load_config_dict(raw)


def test_embedding_dim_over_halfvec_index_cap_flagged_clearly(config_dict_factory: object) -> None:
    """Reproduces the exact v1.0 defect at the config layer: 4096 is legal
    for the halfvec TYPE but exceeds the HNSW/IVFFlat INDEX cap of 4000."""
    raw = config_dict_factory()  # type: ignore[operator]
    raw["embedding"]["dim"] = 4096
    with pytest.raises(ConfigError, match="4000"):
        load_config_dict(raw)


def test_multimodal_dim_must_match_migration_0002(config_dict_factory: object) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["embedding"]["multimodal"]["dim"] = 512
    with pytest.raises(ConfigError, match="1280"):
        load_config_dict(raw)


def test_multimodal_provider_google_rejected(config_dict_factory: object) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["embedding"]["multimodal"]["provider"] = "google"
    with pytest.raises(ConfigError):
        load_config_dict(raw)


def test_multimodal_scope_allowlisted_rejected(config_dict_factory: object) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["embedding"]["multimodal"]["scope"] = "allowlisted"
    with pytest.raises(ConfigError):
        load_config_dict(raw)


# --------------------------------------------------------------------------
# Other closed-set / bounds checks
# --------------------------------------------------------------------------


def test_sync_interval_below_minimum_rejected(config_dict_factory: object) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["sync"]["interval_seconds"] = 60
    with pytest.raises(ConfigError, match="300"):
        load_config_dict(raw)


def test_sync_sources_must_be_nonempty(config_dict_factory: object) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["sync"]["sources"] = []
    with pytest.raises(ConfigError):
        load_config_dict(raw)


def test_sync_sources_names_must_be_unique(config_dict_factory: object, messages_dir: Path) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["sync"]["sources"].append(
        {"name": "mini", "chat_db": str(messages_dir / "chat.db")}
    )
    with pytest.raises(ConfigError, match="unique"):
        load_config_dict(raw)


def test_identity_default_region_invalid_rejected(config_dict_factory: object) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["identity"]["default_region"] = "ZZ"
    with pytest.raises(ConfigError):
        load_config_dict(raw)


def test_identity_default_region_is_case_normalized(config_dict_factory: object) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["identity"]["default_region"] = "us"
    cfg = load_config_dict(raw)
    assert cfg.identity.default_region == "US"


def test_render_timezone_invalid_rejected(config_dict_factory: object) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["render"]["timezone"] = "Mars/Phobos"
    with pytest.raises(ConfigError):
        load_config_dict(raw)


@pytest.mark.parametrize("window", ["25:00-07:00", "01:00_07:00", "1:00-7:00", "garbage"])
def test_enrichment_window_malformed_rejected(config_dict_factory: object, window: str) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["enrichment"]["window"] = window
    with pytest.raises(ConfigError):
        load_config_dict(raw)


def test_segmentation_min_greater_than_max_rejected(config_dict_factory: object) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["segmentation"]["topical_min_messages"] = 100
    raw["segmentation"]["max_messages"] = 50
    with pytest.raises(ConfigError):
        load_config_dict(raw)


def test_export_required_string_fields_reject_empty(config_dict_factory: object) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["export"]["gcp_project"] = ""
    with pytest.raises(ConfigError):
        load_config_dict(raw)


def test_enrichment_egress_only_local_only_is_valid(config_dict_factory: object) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["enrichment"]["egress"] = "hosted"
    with pytest.raises(ConfigError):
        load_config_dict(raw)


# --------------------------------------------------------------------------
# models.backend and the model-pin fields (SPEC §4.1, §6; imsg.providers)
# --------------------------------------------------------------------------


def test_models_backend_defaults_to_real(config_dict_factory: object) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    del raw["models"]
    cfg = load_config_dict(raw)
    assert cfg.models.backend == "real"


def test_models_backend_fake_is_an_explicit_opt_in(config_dict_factory: object) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["models"] = {"backend": "fake"}
    assert load_config_dict(raw).models.backend == "fake"


@pytest.mark.parametrize("value", ["stub", "mock", "", "REAL", None])
def test_models_backend_rejects_anything_but_real_or_fake(
    config_dict_factory: object, value: object
) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["models"] = {"backend": value}
    with pytest.raises(ConfigError, match=r"models\.backend"):
        load_config_dict(raw)


def test_models_unknown_key_rejected(config_dict_factory: object) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["models"] = {"backend": "fake", "device": "mps"}
    with pytest.raises(ConfigError, match=r"models\.device"):
        load_config_dict(raw)


def test_boundary_revision_defaults_to_the_manifest_pin(config_dict_factory: object) -> None:
    from imsg import constants

    cfg = load_config_dict(config_dict_factory())  # type: ignore[operator]
    assert cfg.segmentation.boundary_revision == constants.BOUNDARY_MODEL_REVISION
    raw = config_dict_factory()  # type: ignore[operator]
    raw["segmentation"]["boundary_revision"] = "abc123"
    assert load_config_dict(raw).segmentation.boundary_revision == "abc123"


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("segmentation", "boundary_revision"),
        ("segmentation", "boundary_model"),
        ("enrichment", "transcription_revision"),
        ("enrichment", "caption_model"),
        ("embedding", "model"),
        ("retrieval", "reranker_model"),
    ],
)
def test_model_pin_fields_reject_empty_strings(
    config_dict_factory: object, section: str, field: str
) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw[section][field] = ""
    with pytest.raises(ConfigError, match=rf"{section}\.{field}"):
        load_config_dict(raw)


def test_enrichment_model_fields_default_to_the_manifest_pins(config_dict_factory: object) -> None:
    from imsg import constants

    cfg = load_config_dict(config_dict_factory())  # type: ignore[operator]
    e = cfg.enrichment
    assert (e.transcription_model, e.transcription_revision) == (
        constants.TRANSCRIPTION_MODEL_REPO,
        constants.TRANSCRIPTION_MODEL_REVISION,
    )
    assert (e.caption_model, e.caption_revision) == (
        constants.CAPTION_MODEL_REPO,
        constants.CAPTION_MODEL_REVISION,
    )
    assert e.caption_prompt == Path("prompts/caption.txt")
    assert e.transcription_language is None
    assert e.ocr_languages is None
    assert e.ocr_minimum_text_height is None
    assert cfg.embedding.multimodal.batch_size == 16


def test_caption_prompt_must_resolve_under_data_root(
    config_dict_factory: object, tmp_path: Path
) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["enrichment"]["caption_prompt"] = str(tmp_path / "elsewhere" / "caption.txt")
    with pytest.raises(ConfigError, match=r"enrichment\.caption_prompt.*must resolve under"):
        load_config_dict(raw)


def test_caption_prompt_may_not_escape_via_dotdot(config_dict_factory: object) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["enrichment"]["caption_prompt"] = "../caption.txt"
    with pytest.raises(ConfigError, match=r"enrichment\.caption_prompt"):
        load_config_dict(raw)


def test_ocr_languages_validation(config_dict_factory: object) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["enrichment"]["ocr_languages"] = []
    with pytest.raises(ConfigError, match=r"enrichment\.ocr_languages"):
        load_config_dict(raw)

    raw = config_dict_factory()  # type: ignore[operator]
    raw["enrichment"]["ocr_languages"] = ["en-US", "  "]
    with pytest.raises(ConfigError, match=r"enrichment\.ocr_languages"):
        load_config_dict(raw)

    raw = config_dict_factory()  # type: ignore[operator]
    raw["enrichment"]["ocr_languages"] = [" en-US ", "fr-FR"]
    assert load_config_dict(raw).enrichment.ocr_languages == ["en-US", "fr-FR"]


def test_transcription_language_blank_rejected(config_dict_factory: object) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["enrichment"]["transcription_language"] = "   "
    with pytest.raises(ConfigError, match=r"enrichment\.transcription_language"):
        load_config_dict(raw)
    raw["enrichment"]["transcription_language"] = " en "
    assert load_config_dict(raw).enrichment.transcription_language == "en"


@pytest.mark.parametrize("value", [0, -0.1, 1.5])
def test_ocr_minimum_text_height_must_be_a_fraction(
    config_dict_factory: object, value: float
) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["enrichment"]["ocr_minimum_text_height"] = value
    with pytest.raises(ConfigError, match=r"enrichment\.ocr_minimum_text_height"):
        load_config_dict(raw)


def test_multimodal_batch_size_must_be_positive(config_dict_factory: object) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["embedding"]["multimodal"]["batch_size"] = 0
    with pytest.raises(ConfigError, match=r"embedding\.multimodal\.batch_size"):
        load_config_dict(raw)


# --------------------------------------------------------------------------
# retrieval.rerank_top / rerank_doc_max_tokens: the reranker's latency knobs
# --------------------------------------------------------------------------


def test_rerank_defaults_are_the_measured_latency_settings(config_dict_factory: object) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["retrieval"].pop("rerank_top", None)
    raw["retrieval"].pop("rerank_doc_max_tokens", None)
    retrieval = load_config_dict(raw).retrieval
    assert (retrieval.rerank_top, retrieval.rerank_doc_max_tokens) == (10, 64)


@pytest.mark.parametrize("value", [None, 1, 256, 8192])
def test_rerank_doc_max_tokens_accepts_null_or_a_positive_count(
    config_dict_factory: object, value: int | None
) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["retrieval"]["rerank_doc_max_tokens"] = value
    assert load_config_dict(raw).retrieval.rerank_doc_max_tokens == value


@pytest.mark.parametrize("value", [0, -1, "many"])
def test_rerank_doc_max_tokens_rejects_non_positive_values(
    config_dict_factory: object, value: object
) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["retrieval"]["rerank_doc_max_tokens"] = value
    with pytest.raises(ConfigError, match=r"retrieval\.rerank_doc_max_tokens"):
        load_config_dict(raw)


# --------------------------------------------------------------------------
# retrieval.reranker_model: a Hub repo id, or a directory under data_root
# --------------------------------------------------------------------------


def test_reranker_model_defaults_to_the_lock_pin(config_dict_factory: object) -> None:
    from imsg import constants

    raw = config_dict_factory()  # type: ignore[operator]
    del raw["retrieval"]["reranker_model"]
    cfg = load_config_dict(raw)
    assert cfg.retrieval.reranker_model == constants.RERANKER_MODEL
    assert cfg.retrieval.reranker_model.startswith("models/")  # the local conversion's output_dir


@pytest.mark.parametrize(
    "value",
    [
        "Qwen/Qwen3-Reranker-8B",
        "models/qwen3-reranker-8b-mxfp8-77d193c7",
        " models/nested/dir/ ",
    ],
)
def test_reranker_model_accepts_a_repo_id_or_a_data_root_relative_dir(
    config_dict_factory: object, value: str
) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["retrieval"]["reranker_model"] = value
    assert load_config_dict(raw).retrieval.reranker_model == value.strip()


@pytest.mark.parametrize(
    "value",
    [
        "/Volumes/elsewhere/models/x",
        "~/models/x",
        "../models/x",
        "models/../../x",
        " ",
    ],
)
def test_reranker_model_rejects_forms_that_could_leave_data_root(
    config_dict_factory: object, value: str
) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    raw["retrieval"]["reranker_model"] = value
    with pytest.raises(ConfigError) as excinfo:
        load_config_dict(raw)
    message = str(excinfo.value)
    assert "retrieval.reranker_model" in message
    # The message says which two forms are accepted.
    assert "Hugging Face repo id ('owner/name')" in message
    assert "directory relative to paths.data_root" in message


def test_reranker_model_symlink_escape_rejected(config_dict_factory: object) -> None:
    raw = config_dict_factory()  # type: ignore[operator]
    data_root = Path(raw["paths"]["data_root"])
    (data_root / "models").mkdir(parents=True, exist_ok=True)
    outside = data_root.parent / "outside-models"
    outside.mkdir(exist_ok=True)
    (data_root / "models" / "escaped").symlink_to(outside)
    raw["retrieval"]["reranker_model"] = "models/escaped"
    with pytest.raises(ConfigError, match=r"retrieval\.reranker_model.*resolve under paths\.data_root"):
        load_config_dict(raw)
