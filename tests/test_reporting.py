import json
import pytest
import numpy as np
from pathlib import Path
from propfirm.io.reporting import build_report, save_report


class TestBuildReport:
    def test_has_meta_section(self):
        report = build_report(
            params={"eval": {"stop": 40}, "funded": {"stop": 60}},
            mc_result=None,
            config_snapshot={"test": True},
            data_split="train",
            data_date_range=("2022-01-03", "2024-06-28"),
            seed=42,
        )
        assert "meta" in report
        assert report["meta"]["random_seed"] == 42
        assert report["meta"]["data_split"] == "train"
        assert report["meta"]["config_snapshot"] == {"test": True}

    def test_has_git_hash(self):
        report = build_report(
            params={}, mc_result=None, config_snapshot={},
            data_split="train", data_date_range=("", ""), seed=42,
        )
        assert "git_hash" in report["meta"]

    def test_has_timestamp(self):
        report = build_report(
            params={}, mc_result=None, config_snapshot={},
            data_split="train", data_date_range=("", ""), seed=42,
        )
        assert "timestamp" in report["meta"]
        assert "T" in report["meta"]["timestamp"]


class TestSaveReport:
    def test_saves_valid_json(self, tmp_path):
        report = build_report(
            params={"eval": {}}, mc_result=None, config_snapshot={},
            data_split="train", data_date_range=("", ""), seed=42,
        )
        out_path = tmp_path / "test_report.json"
        save_report(report, out_path)
        assert out_path.exists()
        loaded = json.loads(out_path.read_text())
        assert loaded["meta"]["random_seed"] == 42
        assert loaded["meta"]["config_snapshot"] == {}
