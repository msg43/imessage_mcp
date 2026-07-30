"""pgvector `halfvec` text-literal serialization (SPEC §7.2/§7.4)."""

from __future__ import annotations

from imsg.embed.vector_codec import vector_literal


def test_basic_format() -> None:
    assert vector_literal([1.0, 2.0, 3.0]) == "[1.0,2.0,3.0]"


def test_empty_vector() -> None:
    assert vector_literal([]) == "[]"


def test_negative_and_fractional_values() -> None:
    result = vector_literal([-0.5, 0.25, -1.0])
    assert result.startswith("[") and result.endswith("]")
    assert "-0.5" in result
    assert "0.25" in result


def test_roundtrips_through_a_real_postgres_halfvec_cast() -> None:
    """No mocking of the cast itself -- if a live scratch Postgres is
    reachable, actually ask it to parse the literal."""
    import os

    import psycopg
    import pytest

    host = os.environ.get("IMSG_TEST_PG_HOST", "/tmp/imsgpg1")
    port = os.environ.get("IMSG_TEST_PG_PORT", "55432")
    user = os.environ.get("IMSG_TEST_PG_USER", "postgres")
    dsn = f"postgresql://{user}@/postgres?host={host}&port={port}"
    try:
        conn = psycopg.connect(dsn, connect_timeout=2)
    except Exception:
        pytest.skip(f"no reachable scratch Postgres instance (tried {host}:{port})")
        return

    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        values = [0.1, -0.2, 0.3, 1.5]
        literal = vector_literal(values)
        row = conn.execute("SELECT %s::halfvec(4)", (literal,)).fetchone()
        assert row is not None
        parsed = str(row[0])
        assert parsed.startswith("[") and parsed.endswith("]")
    finally:
        conn.close()
