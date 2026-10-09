"""Summarize no-training 3-class event availability and freshness."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/experiments/full_fault_type_data_audit"
TYPES = ["pressure_low", "pressure_high", "leak"]


def main() -> None:
    e = pd.read_csv(OUT / "event_availability.csv")
    h = pd.read_csv(OUT / "history_coverage.csv")
    hz = pd.read_csv(OUT / "horizon_anchor_coverage.csv")
    multi = pd.read_csv(OUT / "multi_anchor_interval_coverage.csv")
    balance = pd.read_csv(OUT / "class_balance.csv")
    tool = pd.read_csv(OUT / "tool_fault_counts.csv")
    assert len(e) == 1211
    assert e.fault_type.value_counts().reindex(TYPES).tolist() == [741, 333, 137]
    assert tool.groupby("fault_type").total_events.sum().reindex(TYPES).tolist() == [741, 333, 137]
    freshness = []
    for minutes in (10, 30, 60, 180):
        for typ in TYPES:
            for phase in ("train", "validation", "reserved_or_excluded", "all"):
                part = e[e.fault_type == typ]
                if phase != "all":
                    part = part[part.phase == phase]
                usable = part[f"history_{minutes}m"]
                freshness.append({"history_minutes": minutes, "fault_type": typ, "phase": phase,
                    "total_events": len(part), "history_at_last_observation": int(usable.sum()),
                    "history_and_last_sensor_within_60s_of_fault": int((usable & part.last_sensor_lead_seconds.le(60)).sum()),
                    "history_and_last_sensor_within_30m_of_fault": int((usable & part.last_sensor_lead_seconds.le(1800)).sum()),
                    "history_and_last_sensor_within_24h_of_fault": int((usable & part.last_sensor_lead_seconds.le(86400)).sum())})
    pd.DataFrame(freshness).to_csv(OUT / "history_freshness.csv", index=False)
    concentration = []
    for typ in TYPES:
        part = tool[tool.fault_type == typ].sort_values("total_events", ascending=False)
        total = int(part.total_events.sum())
        concentration.append({"fault_type": typ, "events": total,
            "tools_with_any_event": int((part.total_events > 0).sum()),
            "top_tool": part.iloc[0].tool_id, "top_tool_events": int(part.iloc[0].total_events),
            "top_tool_share_pct": float(100 * part.iloc[0].total_events / total),
            "top_3_tool_events": int(part.head(3).total_events.sum()),
            "top_3_tool_share_pct": float(100 * part.head(3).total_events.sum() / total),
            "train_30m_usable_tools": int((part.train_30m_usable > 0).sum()),
            "validation_30m_usable_tools": int((part.validation_30m_usable > 0).sum())})
    pd.DataFrame(concentration).to_csv(OUT / "tool_concentration.csv", index=False)
    totals = []
    for minutes in (10, 30, 60, 180):
        for phase in ("train", "validation", "reserved_or_excluded", "all"):
            part = h[(h.history_minutes == minutes) & (h.phase == phase)]
            totals.append({"history_minutes": minutes, "phase": phase,
                "usable_low": int(part.loc[part.fault_type == "pressure_low", "usable_events"].iloc[0]),
                "usable_high": int(part.loc[part.fault_type == "pressure_high", "usable_events"].iloc[0]),
                "usable_leak": int(part.loc[part.fault_type == "leak", "usable_events"].iloc[0]),
                "usable_total": int(part.usable_events.sum()),
                "original_total": int(part.total_events.sum()),
                "coverage_pct": float(100 * part.usable_events.sum() / part.total_events.sum())})
    pd.DataFrame(totals).to_csv(OUT / "history_totals.csv", index=False)
    summary = {"unique_events": len(e), "type_counts": e.fault_type.value_counts().reindex(TYPES).to_dict(),
        "train_30m": h[(h.history_minutes == 30) & (h.phase == "train")].set_index("fault_type").usable_events.to_dict(),
        "validation_30m": h[(h.history_minutes == 30) & (h.phase == "validation")].set_index("fault_type").usable_events.to_dict(),
        "train_60m": h[(h.history_minutes == 60) & (h.phase == "train")].set_index("fault_type").usable_events.to_dict(),
        "validation_60m": h[(h.history_minutes == 60) & (h.phase == "validation")].set_index("fault_type").usable_events.to_dict(),
        "total_30m_history_events": int(h[(h.history_minutes == 30) & (h.phase == "all")].usable_events.sum()),
        "train_validation_30m_history_events": int(h[(h.history_minutes == 30) & h.phase.isin(["train", "validation"])].usable_events.sum()),
        "horizon_30m_history_events_train_validation": hz.groupby("horizon_minutes_before_fault").usable_events.sum().to_dict(),
        "multi_anchor_interval_events_train_validation": multi.groupby("interval").usable_events.sum().to_dict(),
        "note": "Intervals and fixed horizons overlap across events; counts must not be summed as distinct events. No model trained."}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(pd.DataFrame(concentration).to_string(index=False))
    print(balance.to_string(index=False))


if __name__ == "__main__":
    main()
