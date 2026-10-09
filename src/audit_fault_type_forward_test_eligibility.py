"""Audit whether a historically untouched forward fault-type test exists.

This deliberately reads split metadata only. It never opens reserved sensor,
fault-label, score, or outcome rows and never loads or trains a model.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "output/baseline_recovery/preparation_report.json"
NEW = ROOT / "output/experiments/forward_validation_ab_24h/split_plan.json"
OUT = ROOT / "output/experiments/fault_type_forward_test"


def main() -> None:
    old = json.loads(OLD.read_text(encoding="utf-8"))
    new = json.loads(NEW.read_text(encoding="utf-8"))
    old_by_tool = {item["tool"]: item for item in old["tools"]}
    rows = []
    for item in new["tools"]:
        tool = item["tool_id"]
        previous = old_by_tool[tool]
        old_start = int(previous["split_cutoff_time"])
        reserved_start = int(item["test_start_second"])
        rows.append({
            "tool_id": tool,
            "new_reserved_test_start_second": reserved_start,
            "old_baseline_validation_start_second": old_start,
            "reserved_start_before_old_validation_start": reserved_start < old_start,
            "old_validation_rows": int(previous["validation_rows"]),
            "old_validation_was_evaluated": True,
            "historically_untouched_later_suffix_exists_in_current_dataset": False,
        })
    assert len(rows) == 20
    assert all(row["old_validation_rows"] > 0 for row in rows)
    assert all(row["reserved_start_before_old_validation_start"] for row in rows)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "split_exposure_audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "status": "not_run_no_historically_untouched_forward_interval",
        "requested_evaluation": "first manager notification, frozen Model 1 and Tree fault-type classifier",
        "tools_audited": len(rows),
        "old_baseline_split": "per-tool chronological first 80% train / last 20% validation",
        "new_split": new["scheme"],
        "all_new_reserved_test_starts_before_old_validation_starts": True,
        "all_late_suffixes_previously_evaluated_in_old_baseline": True,
        "test_sensor_rows_read": 0,
        "test_fault_rows_read": 0,
        "test_predictions_created": 0,
        "models_loaded_or_fitted": 0,
        "threshold_or_confidence_selection": "none",
        "reason": "The final chronological suffix of all 20 tools was the old recovered-baseline validation set. Thus the current 60/20/20 reserved Test overlaps prior development/validation, and no later labeled suffix exists within this recorded dataset.",
        "required_next_input": "New, later labeled sensor and fault observations beyond the previously evaluated end for each tool, with a frozen evaluation protocol before outcomes are opened.",
        "source_files": [str(OLD.relative_to(ROOT)), str(NEW.relative_to(ROOT)),
                         "output/baseline_recovery/reproduction_report.json"],
    }
    (OUT / "eligibility_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
