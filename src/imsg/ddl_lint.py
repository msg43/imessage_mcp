"""Static DDL lint for the pgvector index-cap trap (SPEC §7.2, D6).

**The bug this exists to prevent**: pgvector's `vector`/`halfvec`
column *types* accept up to 16,000 dimensions — `CREATE TABLE` with a
wide vector column succeeds either way — but an HNSW or IVFFlat
*index* on that column caps at 2,000 dims for `vector` and 4,000 dims
for `halfvec` (empirically verified against a live pgvector 0.8.6
instance, 2026-07-30; see `imsg.constants`). A column can therefore be
perfectly legal DDL whose `CREATE INDEX` fails — or, worse, whose index
is simply never created and the table silently degrades to a
sequential scan. That exact defect (a `halfvec(4096)` column paired
with an HNSW index) was the blocker that produced SPEC v1.1; this lint
exists so it cannot recur silently.

This is a small hand-rolled parser, not a general SQL engine — it only
needs to understand the shapes our own migrations use (see
`_split_statements`/`_split_top_level` below for what it tolerates:
line comments, single-quoted strings, and arbitrarily nested
parentheses). It is not meant to lint arbitrary third-party SQL.

Three checks, run over every `CREATE TABLE`/`CREATE INDEX` statement in
every migration file:

1. Every `vector`/`halfvec` column's dimension must fit under its
   type's *index* cap (not just the type's own cap) for the ANN index
   defined on it.
2. Every `vector`/`halfvec` column must have at least one HNSW/IVFFlat
   index — an un-indexed vector column is a lint failure, not merely a
   missed optimization, because it means every retrieval query against
   it silently falls back to a full sequential scan.
3. Every such index's operator class must be the cosine-distance
   family (`vector_cosine_ops`/`halfvec_cosine_ops`) — this build's
   retrieval layer (SPEC §9.4) computes and compares scores assuming
   cosine distance throughout; a mismatched opclass would silently
   return wrong distances/rankings rather than an error.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from imsg import constants

_VECTOR_TYPE_RE = re.compile(r"\b(vector|halfvec)\s*\(\s*(\d+)\s*\)", re.IGNORECASE)
_CREATE_TABLE_RE = re.compile(r"^CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s*\(", re.IGNORECASE)
_CREATE_INDEX_RE = re.compile(
    r"^CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s+ON\s+(\w+)\s+"
    r"USING\s+(hnsw|ivfflat)\s*\(\s*(\w+)\s+(\w+)",
    re.IGNORECASE,
)
_TABLE_LEVEL_CONSTRAINT_KEYWORDS = {"primary", "unique", "check", "foreign", "constraint"}

_INDEX_DIM_CAP = {
    "vector": constants.VECTOR_INDEX_MAX_DIM,
    "halfvec": constants.HALFVEC_INDEX_MAX_DIM,
}
_TYPE_DIM_CAP = {
    "vector": constants.VECTOR_TYPE_MAX_DIM,
    "halfvec": constants.HALFVEC_TYPE_MAX_DIM,
}
_EXPECTED_OPCLASS = {"vector": "vector_cosine_ops", "halfvec": "halfvec_cosine_ops"}


@dataclass(frozen=True, slots=True)
class VectorColumn:
    table: str
    column: str
    kind: str  # 'vector' | 'halfvec'
    dim: int
    source_file: str


@dataclass(frozen=True, slots=True)
class VectorIndex:
    index_name: str
    table: str
    method: str  # 'hnsw' | 'ivfflat'
    column: str
    opclass: str
    source_file: str


# --- a small statement/clause splitter that respects parens, strings, and comments ---


def _split_statements(sql: str) -> list[str]:
    """Split a SQL file into top-level (`;`-terminated) statements.

    Tracks paren depth and single-quoted strings so semicolons and
    parens inside string literals or `CHECK (...)` clauses do not
    confuse the split; strips `--` line comments.
    """
    statements: list[str] = []
    buf: list[str] = []
    depth = 0
    in_string = False
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        if in_string:
            buf.append(ch)
            if ch == "'":
                if i + 1 < n and sql[i + 1] == "'":  # escaped '' inside a string
                    buf.append(sql[i + 1])
                    i += 2
                    continue
                in_string = False
            i += 1
            continue
        if ch == "'":
            in_string = True
            buf.append(ch)
            i += 1
            continue
        if ch == "-" and i + 1 < n and sql[i + 1] == "-":
            while i < n and sql[i] != "\n":
                i += 1
            continue
        if ch == "(":
            depth += 1
            buf.append(ch)
            i += 1
            continue
        if ch == ")":
            depth -= 1
            buf.append(ch)
            i += 1
            continue
        if ch == ";" and depth == 0:
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


def _split_top_level(s: str, sep: str = ",") -> list[str]:
    """Split `s` on `sep`, but only where paren depth is 0 and outside strings."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    in_string = False
    for ch in s:
        if in_string:
            buf.append(ch)
            if ch == "'":
                in_string = False
            continue
        if ch == "'":
            in_string = True
            buf.append(ch)
            continue
        if ch == "(":
            depth += 1
            buf.append(ch)
            continue
        if ch == ")":
            depth -= 1
            buf.append(ch)
            continue
        if ch == sep and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _extract_paren_body(statement: str) -> str:
    """Return the text strictly between the first `(` and its matching `)`."""
    start = statement.index("(")
    depth = 0
    in_string = False
    for i in range(start, len(statement)):
        ch = statement[i]
        if in_string:
            if ch == "'":
                in_string = False
            continue
        if ch == "'":
            in_string = True
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return statement[start + 1 : i]
    raise ValueError("unbalanced parentheses in statement")


def _parse_create_table(statement: str, source_file: str) -> list[VectorColumn]:
    m = _CREATE_TABLE_RE.match(statement)
    if not m:
        return []
    table = m.group(1)
    body = _extract_paren_body(statement)
    columns: list[VectorColumn] = []
    for clause in _split_top_level(body):
        tokens = clause.split(None, 1)
        if not tokens:
            continue
        first = tokens[0].lower()
        if first in _TABLE_LEVEL_CONSTRAINT_KEYWORDS:
            continue  # table-level constraint, not a column definition
        column_name = tokens[0]
        type_match = _VECTOR_TYPE_RE.search(clause)
        if type_match:
            kind = type_match.group(1).lower()
            dim = int(type_match.group(2))
            columns.append(
                VectorColumn(table=table, column=column_name, kind=kind, dim=dim, source_file=source_file)
            )
    return columns


def _parse_create_index(statement: str, source_file: str) -> VectorIndex | None:
    m = _CREATE_INDEX_RE.match(statement)
    if not m:
        return None
    index_name, table, method, column, opclass = m.groups()
    return VectorIndex(
        index_name=index_name,
        table=table,
        method=method.lower(),
        column=column,
        opclass=opclass,
        source_file=source_file,
    )


def parse_migration(sql: str, source_file: str) -> tuple[list[VectorColumn], list[VectorIndex]]:
    columns: list[VectorColumn] = []
    indexes: list[VectorIndex] = []
    for statement in _split_statements(sql):
        stripped = statement.strip()
        if _CREATE_TABLE_RE.match(stripped):
            columns.extend(_parse_create_table(stripped, source_file))
        elif _CREATE_INDEX_RE.match(stripped):
            idx = _parse_create_index(stripped, source_file)
            if idx is not None:
                indexes.append(idx)
    return columns, indexes


def lint_migrations(migrations_dir: Path) -> list[str]:
    """Run every check over every `*.sql` file in `migrations_dir`.

    Returns a list of human-readable error strings; empty means clean.
    Pure function, no database required — safe to call from unit tests.
    """
    errors: list[str] = []
    all_columns: list[VectorColumn] = []
    all_indexes: list[VectorIndex] = []

    for path in sorted(migrations_dir.glob("*.sql")):
        sql = path.read_text()
        columns, indexes = parse_migration(sql, path.name)
        all_columns.extend(columns)
        all_indexes.extend(indexes)

    indexes_by_table_column: dict[tuple[str, str], list[VectorIndex]] = {}
    for idx in all_indexes:
        indexes_by_table_column.setdefault((idx.table, idx.column), []).append(idx)

    for col in all_columns:
        type_cap = _TYPE_DIM_CAP[col.kind]
        if col.dim > type_cap:
            errors.append(
                f"{col.source_file}: {col.table}.{col.column} is {col.kind}({col.dim}), "
                f"which exceeds the {col.kind} type's own cap of {type_cap} dims — not "
                f"valid DDL at all"
            )

        matching_indexes = indexes_by_table_column.get((col.table, col.column), [])
        if not matching_indexes:
            errors.append(
                f"{col.source_file}: {col.table}.{col.column} ({col.kind}({col.dim})) has "
                f"no HNSW/IVFFlat index — queries against it will silently sequential-scan "
                f"the whole table instead of using an ANN index"
            )
            continue

        index_cap = _INDEX_DIM_CAP[col.kind]
        expected_opclass = _EXPECTED_OPCLASS[col.kind]
        for idx in matching_indexes:
            if col.dim > index_cap:
                errors.append(
                    f"{col.source_file}: {col.table}.{col.column} is {col.kind}({col.dim}) "
                    f"with a {idx.method} index ('{idx.index_name}') — but {idx.method.upper()} "
                    f"and IVFFlat indexes on {col.kind} cap at {index_cap} dims (the {col.kind} "
                    f"TYPE itself allows up to {type_cap}, so this is legal column DDL whose "
                    f"index can never be created; this is precisely the bug that produced "
                    f"SPEC v1.1 — a halfvec(4096) HNSW index that could never CREATE INDEX)"
                )
            if idx.opclass.lower() != expected_opclass:
                errors.append(
                    f"{col.source_file}: index '{idx.index_name}' on {col.table}.{col.column} "
                    f"uses opclass '{idx.opclass}', expected '{expected_opclass}' — this "
                    f"build's retrieval layer computes cosine distance throughout (SPEC §9.4); "
                    f"a mismatched operator class would silently rank results by the wrong "
                    f"distance metric rather than raising an error"
                )

    return errors


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    migrations_dir = Path(args[0]) if args else Path(__file__).resolve().parents[2] / "migrations"

    if not migrations_dir.is_dir():
        print(f"lint_ddl: migrations directory not found: '{migrations_dir}'", file=sys.stderr)
        return 2

    errors = lint_migrations(migrations_dir)
    if errors:
        print(f"lint_ddl: {len(errors)} problem(s) found in '{migrations_dir}':\n")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(f"lint_ddl: clean — checked migrations in '{migrations_dir}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
