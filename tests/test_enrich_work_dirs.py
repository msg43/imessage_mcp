"""Each enrichment task's work directory lives under `data_root`, and the
worker removes the ones a killed process left behind
(`imsg.enrich.pipeline.sweep_stale_work_dirs`, called when
`imsg.enrich.worker.run_enrich_worker` starts). No database: the worker's
claim step is a stand-in that finds nothing to do."""

from __future__ import annotations

import subprocess

import pytest

from conftest import ConfigDictFactory
from imsg.config.loader import load_config_dict
from imsg.config.schema import Config
from imsg.db.enrichment_yield_locks import YieldReport
from imsg.enrich.pipeline import enrich_work_root
from imsg.enrich.worker import run_enrich_worker


@pytest.fixture
def config(config_dict_factory: ConfigDictFactory) -> Config:
    return load_config_dict(config_dict_factory())


class NeverYields:
    def wait_until_clear(self) -> YieldReport:
        return YieldReport(paused=False, waited_seconds=0.0)


def test_the_work_root_is_under_data_root(config: Config) -> None:
    root = enrich_work_root(config.paths.data_root)
    assert root == (config.paths.data_root / "artifacts" / "enrich-work").resolve()


def test_the_worker_removes_work_dirs_left_by_dead_processes(config: Config) -> None:
    dead = subprocess.Popen(["/usr/bin/true"])
    dead.wait()
    stale = enrich_work_root(config.paths.data_root) / f"{dead.pid}-1-ocr-x"
    stale.mkdir(parents=True)
    run_enrich_worker(
        None,  # type: ignore[arg-type]
        config,
        providers=None,  # type: ignore[arg-type]
        worker_id="w",
        claim_order=("ocr",),
        limit=1,
        yield_gate=NeverYields(),
        claim=lambda conn, **kw: [],
        process=lambda *args: "done",
    )
    assert not stale.exists()
