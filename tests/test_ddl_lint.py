"""DDL lint tests (SPEC §7.2, D6) — reproduces the exact v1.0 blocker
(a `halfvec(4096)` HNSW index that could never be created) as a fixture,
plus checks the real migrations stay clean."""

from __future__ import annotations

from pathlib import Path

from imsg import constants
from imsg.ddl_lint import lint_migrations, parse_migration

REAL_MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"


def test_real_migrations_are_clean() -> None:
    assert lint_migrations(REAL_MIGRATIONS_DIR) == []


def test_constants_match_the_actual_migration_ddl() -> None:
    """Single-source-of-truth guard: if migration 0001/0002 ever changes a
    dim, this fails until imsg.constants is updated too."""
    sql_0001 = (REAL_MIGRATIONS_DIR / "0001_initial.sql").read_text()
    sql_0002 = (REAL_MIGRATIONS_DIR / "0002_multimodal_vectors.sql").read_text()
    assert f"halfvec({constants.PRIMARY_EMBEDDING_DIM})" in sql_0001
    assert f"CHECK (dim = {constants.PRIMARY_EMBEDDING_DIM})" in sql_0001
    assert f"halfvec({constants.MULTIMODAL_EMBEDDING_DIM})" in sql_0002
    assert f"CHECK (dim = {constants.MULTIMODAL_EMBEDDING_DIM})" in sql_0002


def test_reproduces_the_v1_0_blocker_halfvec_4096_hnsw(tmp_path: Path) -> None:
    """The exact defect that produced SPEC v1.1: halfvec(4096) is legal
    column DDL (type cap 16000) but its HNSW index cap is 4000 — this must
    be flagged, not silently accepted."""
    bad = tmp_path / "0001_bad.sql"
    bad.write_text(
        """
        CREATE TABLE segment_embedding (
          segment_id bigint PRIMARY KEY,
          vec halfvec(4096) NOT NULL
        );
        CREATE INDEX segment_embedding_hnsw ON segment_embedding
          USING hnsw (vec halfvec_cosine_ops) WITH (m = 16, ef_construction = 200);
        """
    )
    errors = lint_migrations(tmp_path)
    assert len(errors) == 1
    assert "4000" in errors[0]
    assert "halfvec(4096)" in errors[0] or "4096" in errors[0]


def test_vector_type_hnsw_cap_2000_enforced(tmp_path: Path) -> None:
    bad = tmp_path / "0001_bad.sql"
    bad.write_text(
        """
        CREATE TABLE t (id bigint PRIMARY KEY, vec vector(2001) NOT NULL);
        CREATE INDEX t_hnsw ON t USING hnsw (vec vector_cosine_ops);
        """
    )
    errors = lint_migrations(tmp_path)
    assert any("2000" in e for e in errors)


def test_vector_column_at_exactly_the_index_cap_is_clean(tmp_path: Path) -> None:
    ok = tmp_path / "0001_ok.sql"
    ok.write_text(
        """
        CREATE TABLE t (id bigint PRIMARY KEY, vec vector(2000) NOT NULL);
        CREATE INDEX t_hnsw ON t USING hnsw (vec vector_cosine_ops);
        """
    )
    assert lint_migrations(tmp_path) == []


def test_halfvec_column_at_exactly_the_index_cap_is_clean(tmp_path: Path) -> None:
    ok = tmp_path / "0001_ok.sql"
    ok.write_text(
        """
        CREATE TABLE t (id bigint PRIMARY KEY, vec halfvec(4000) NOT NULL);
        CREATE INDEX t_hnsw ON t USING hnsw (vec halfvec_cosine_ops);
        """
    )
    assert lint_migrations(tmp_path) == []


def test_vector_column_with_no_index_is_flagged(tmp_path: Path) -> None:
    bad = tmp_path / "0001_bad.sql"
    bad.write_text(
        """
        CREATE TABLE t (id bigint PRIMARY KEY, vec halfvec(1280) NOT NULL);
        """
    )
    errors = lint_migrations(tmp_path)
    assert len(errors) == 1
    assert "no HNSW/IVFFlat index" in errors[0]


def test_ivfflat_index_is_recognized_too(tmp_path: Path) -> None:
    ok = tmp_path / "0001_ok.sql"
    ok.write_text(
        """
        CREATE TABLE t (id bigint PRIMARY KEY, vec vector(512) NOT NULL);
        CREATE INDEX t_ivf ON t USING ivfflat (vec vector_cosine_ops) WITH (lists = 100);
        """
    )
    assert lint_migrations(tmp_path) == []


def test_mismatched_opclass_is_flagged(tmp_path: Path) -> None:
    bad = tmp_path / "0001_bad.sql"
    bad.write_text(
        """
        CREATE TABLE t (id bigint PRIMARY KEY, vec halfvec(1280) NOT NULL);
        CREATE INDEX t_hnsw ON t USING hnsw (vec halfvec_l2_ops);
        """
    )
    errors = lint_migrations(tmp_path)
    assert len(errors) == 1
    assert "opclass" in errors[0]
    assert "halfvec_l2_ops" in errors[0]


def test_type_cap_violation_flagged_even_without_an_index(tmp_path: Path) -> None:
    bad = tmp_path / "0001_bad.sql"
    bad.write_text("CREATE TABLE t (id bigint PRIMARY KEY, vec vector(16001) NOT NULL);")
    errors = lint_migrations(tmp_path)
    assert any("type's own cap" in e for e in errors)
    # and it should also be flagged for having no index, independently
    assert any("no HNSW/IVFFlat index" in e for e in errors)


def test_parse_migration_ignores_table_level_constraints_as_columns() -> None:
    sql = """
    CREATE TABLE t (
      id bigint PRIMARY KEY,
      status text NOT NULL CHECK (status IN ('a','b')),
      vec halfvec(8) NOT NULL,
      UNIQUE (id, status),
      CHECK (id > 0)
    );
    """
    columns, _indexes = parse_migration(sql, "test.sql")
    assert [c.column for c in columns] == ["vec"]


def test_parse_migration_handles_nested_parens_in_check_and_comments() -> None:
    sql = """
    -- a comment with a semicolon; and a paren )
    CREATE TABLE search_index_event (
      event_id bigint PRIMARY KEY,
      operation text NOT NULL,
      content_sha256 text,
      CHECK ((operation = 'delete' AND content_sha256 IS NULL)
          OR (operation = 'upsert' AND content_sha256 IS NOT NULL))
    );
    CREATE UNIQUE INDEX person_owner_singleton ON search_index_event ((true)) WHERE operation = 'x';
    """
    columns, indexes = parse_migration(sql, "test.sql")
    assert columns == []
    # the ((true)) functional index is not an hnsw/ivfflat index, so it's
    # correctly not picked up as a vector index — just confirm parsing
    # didn't crash or misparse statement boundaries.
    assert indexes == []
