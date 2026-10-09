"""Read archived assistant command output to recover reported experiment values.

This script never treats chat text as code: it parses only one archived JSON
table and Python literal dictionaries printed by the old metrics command.
"""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SESSION = Path(r"C:\Users\USER\.codex\sessions\2026\09\22\rollout-2026-09-22T17-41-36-01a0c847-133e-7a52-8217-d512e5355fe0.jsonl")
OUT = ROOT / "output" / "experiments" / "historical_report_recovery"


def archived_command_output(line_number):
    with SESSION.open(encoding="utf-8", errors="replace") as handle:
        for i, line in enumerate(handle, 1):
            if i == line_number:
                record = json.loads(line)
                return record["payload"]["item"]["formatted_output"]
    raise ValueError(f"Archived line {line_number} not found")


continuous_text = archived_command_output(5104)
continuous_table = json.loads(continuous_text.splitlines()[0])
if continuous_table["sheet"] != "요약" or len(continuous_table["values"]) != 5:
    raise AssertionError("Archived continuous-time summary table changed")
continuous = []
for hours, row in zip((3, 6, 12, 24), continuous_table["values"][1:]):
    detected, denominator = (int(v) for v in row[7].split("/"))
    continuous.append({
        "horizon_hours": hours,
        "validation_rows": row[1], "normal_rows": row[2],
        "positive_rows": row[3], "censored_rows": row[4],
        "positive_prevalence": row[5], "average_precision": row[6],
        "detected_fault_events": detected, "fault_event_denominator": denominator,
        "fault_event_recall": detected / denominator,
        "first_alert_lead_median_minutes": row[8],
        "detected_at_least_1h_before_fault": row[9],
        "detected_at_least_h_hours_before_fault": row[10],
    })

calendar_text = archived_command_output(5914)
classification, regression, joint = [], [], []
for line in calendar_text.splitlines():
    line = line.strip()
    if not line or " {'" not in line:
        continue
    name, raw_dict = line.split(" ", 1)
    if not raw_dict.startswith("{"):
        continue
    values = ast.literal_eval(raw_dict)
    if name.startswith("binary_"):
        parts = name.split("_")
        model = "XGBoost" if parts[1] == "xgb" else "HGB"
        train_fraction = parts[2].replace("pct", "%")
        horizon_hours = int(parts[3][1:]) // 3600
        p, r = values["precision"], values["recall"]
        classification.append({
            "horizon_hours": horizon_hours, "model": model,
            "train_fraction": train_fraction,
            "validation_rows": values["validation_rows"],
            "positive_rows": values["positive_rows"],
            "precision": p, "recall": r,
            "f1_derived_from_precision_recall": 2*p*r/(p+r) if p+r else 0,
            "average_precision": values["average_precision"],
            "evaluable_fault_events": values["events"],
            "all_logged_fault_events": values["logged_events"],
            "detected_fault_events": values["detected_events"],
            "event_recall_evaluable": values["detected_events"] / values["events"],
            "events_without_prior_sensor": values["events_without_prior_sensor_in_horizon"],
            "first_alert_lead_median_minutes": values["median_lead_minutes"],
            "false_alerts_per_observed_hour": values["false_alerts_per_observed_hour"],
        })
    elif name.startswith("ttf_") or name.startswith("lstm_"):
        regression.append({"model_fault_key": name, **values})
    elif name.startswith("xgb_100pct_h"):
        joint.append({"system_key": name, **values})

if (len(continuous), len(classification), len(regression), len(joint)) != (4, 6, 9, 6):
    raise AssertionError("Archived outputs were incomplete")
OUT.mkdir(parents=True, exist_ok=True)
result = {
    "provenance": "Archived Codex conversation command outputs; not recovered original XLSX bytes",
    "archived_session": str(SESSION),
    "source_line_continuous": 5104,
    "source_line_calendar": 5914,
    "missing_original_filenames": [
        "LS_team6/output_extended/LS_team6_3h_6h_12h_24h_검증보고서.xlsx",
        "LS_team6/output_calendar_12_24/LS_team6_시계시간_12h_24h_검증보고서.xlsx",
    ],
    "continuous_operating_time_summary": continuous,
    "calendar_time_classification": classification,
    "calendar_time_regression_partial": regression,
    "calendar_time_joint_partial": joint,
    "limits": [
        "These are values preserved in command output, not the deleted workbook itself.",
        "Continuous-operating-time and calendar-time horizons have different definitions and evaluated populations.",
        "The original 2172-row fault event ledger and some workbook metrics were not preserved in these outputs.",
        "F1 in the calendar classification table is derived from archived precision and recall, not copied from the workbook.",
    ],
}
(OUT / "archived_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Recovered {len(continuous)} continuous-time and {len(classification)} calendar-time classification rows")
