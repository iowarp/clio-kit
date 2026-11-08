"""Contract tests for the search/converter separation.

Scientific units, measurements, formulas and operators moved to clio-converter.
These tests pin what search keeps (structure-aware chunking, cell-level citations,
deterministic ranking) and how removed scientific inputs are rejected.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

from clio_agentic_search.api.app import _registry, app
from clio_agentic_search.cli.main import main
from clio_agentic_search.connectors.filesystem import FilesystemConnector
from clio_agentic_search.retrieval.coordinator import RetrievalCoordinator
from clio_agentic_search.storage import DuckDBStorage

_PAPER = (
    "# Reactor Run A\n"
    "Table 1: Pressure and velocity readings.\n"
    "| Time (min) | Pressure (kPa) | Velocity (km/h) |\n"
    "| --- | --- | --- |\n"
    "| 1 | 120 | 36 |\n"
    "| 2 | 150 | 54 |\n"
    "Figure 1: Trend remains stable.\n"
    "Energy relation: $E = m c^2$.\n"
    "$$F = m a$$\n"
)


def _connector(tmp_path: Path) -> FilesystemConnector:
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "paper.md").write_text(_PAPER, encoding="utf-8")
    (docs / "notes.txt").write_text("Shift schedule rotates weekly.\n", encoding="utf-8")
    connector = FilesystemConnector(
        namespace="ns", root=docs, storage=DuckDBStorage(tmp_path / "s.duckdb")
    )
    connector.connect()
    connector.index(full_rebuild=True)
    return connector


def test_structure_chunking_keeps_kinds_and_cell_fragments(tmp_path: Path) -> None:
    connector = _connector(tmp_path)
    try:
        metadata = [
            connector.storage.get_chunk_metadata("ns", chunk.chunk_id)
            for chunk in connector.storage.list_chunks("ns")
        ]
        kinds = {item.get("structure.kind", "") for item in metadata}
        assert {"section", "caption", "equation", "table", "table_cell"}.issubset(kinds)
        cells = [item for item in metadata if item.get("structure.kind") == "table_cell"]
        assert all("citation.fragment" in item for item in cells)
        assert any(item.get("table.column_name") == "Pressure (kPa)" for item in cells)
        assert not any(key.startswith("scientific.") for item in metadata for key in item)
    finally:
        connector.teardown()


def test_table_cell_citation_uri_points_at_cell(tmp_path: Path) -> None:
    connector = _connector(tmp_path)
    try:
        result = RetrievalCoordinator().query(
            connector=connector,
            query="Pressure 150",
            top_k=5,
            metadata_filters={"structure.kind": "table_cell", "table.row": "2"},
        )
        uris = [citation.uri for citation in result.citations]
        assert "paper.md#table=1&row=2&column=2" in uris
    finally:
        connector.teardown()


def test_repeated_and_reindexed_queries_are_deterministic(tmp_path: Path) -> None:
    connector = _connector(tmp_path)
    coordinator = RetrievalCoordinator()
    try:
        runs = [
            [
                (c.uri, c.score)
                for c in coordinator.query(
                    connector=connector, query="pressure readings", top_k=5
                ).citations
            ]
            for _ in range(3)
        ]
        assert runs[0] == runs[1] == runs[2]
        connector.index(full_rebuild=True)
        after = [
            (c.uri, c.score)
            for c in coordinator.query(
                connector=connector, query="pressure readings", top_k=5
            ).citations
        ]
        assert after == runs[0]
    finally:
        connector.teardown()


def test_legacy_scientific_tables_are_dropped_on_connect(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.duckdb"
    with duckdb.connect(str(db_path)) as connection:
        connection.execute("CREATE TABLE scientific_measurements (namespace TEXT)")
        connection.execute("CREATE TABLE scientific_formulas (namespace TEXT)")
    storage = DuckDBStorage(db_path)
    storage.connect()
    storage.teardown()
    with duckdb.connect(str(db_path)) as connection:
        tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    assert "scientific_measurements" not in tables
    assert "scientific_formulas" not in tables


def test_api_rejects_removed_scientific_operators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("pressure readings", encoding="utf-8")
    monkeypatch.setenv("CLIO_LOCAL_ROOT", str(docs))
    monkeypatch.setenv("CLIO_STORAGE_PATH", str(tmp_path / "api.duckdb"))
    _registry.cache_clear()
    try:
        response = TestClient(app).post(
            "/query",
            json={
                "query": "pressure",
                "scientific_operators": {"numeric_range": {"unit": "kPa", "minimum": 1}},
            },
        )
        assert response.status_code == 422
    finally:
        _registry().teardown()
        _registry.cache_clear()


@pytest.mark.parametrize("flag", ["--numeric-range", "--unit-match", "--formula"])
def test_cli_rejects_removed_scientific_flags(flag: str) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["query", "--q", "pressure", flag, "x"])
    assert exit_info.value.code == 2
