"""Count all logged validation faults behind the replay's conditional event recall."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "experiments" / "historical_exact_replay" / "reproduced"


def main() -> None:
    prep = json.loads((OUT / "preparation_report.json").read_text(encoding="utf-8"))
    cutoffs = {tool: item["split_cutoff_time"] for tool, item in prep["tools"].items()}
    log = pd.read_csv(ROOT / "data" / "processed" / "fault_events.csv", usecols=["tool_id", "fault_type", "time"])
    log = log.loc[log.tool_id.isin(cutoffs)].copy()
    log["validation_cutoff"] = log.tool_id.map(cutoffs)
    validation = log.loc[log.time >= log.validation_cutoff]
    unique = validation.drop_duplicates(["tool_id", "fault_type", "time"])
    report = {"all_logged_validation_fault_rows": len(validation),
              "all_unique_logged_validation_fault_events": len(unique),
              "by_fault_type": unique.groupby("fault_type").size().to_dict()}
    (OUT / "validation_fault_population.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
