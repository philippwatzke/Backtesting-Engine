import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import asdict


def _get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def build_report(
    params: dict,
    mc_result,
    config_snapshot: dict,
    data_split: str,
    data_date_range: tuple[str, str],
    seed: int,
    diagnostics: dict | None = None,
    stress_test: dict | None = None,
) -> dict:
    report = {
        "meta": {
            "git_hash": _get_git_hash(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "random_seed": seed,
            "config_snapshot": config_snapshot,
            "data_split": data_split,
            "data_date_range": list(data_date_range),
        },
        "params": params,
    }
    if mc_result is not None:
        report["results"] = asdict(mc_result)
    if stress_test is not None:
        report["stress_test"] = stress_test
    if diagnostics is not None:
        report["diagnostics"] = diagnostics
    return report


def save_report(report: dict, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
