"""Summarize the no-training low-pressure audit; only own output is updated."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/experiments/low_pressure_rul_audit"
PLAN = ROOT / "output/experiments/forward_validation_ab_24h/split_plan.json"


def main() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    events = pd.read_csv(OUT / "low_pressure_events.csv")
    anchors = pd.read_csv(OUT / "anchor_summary.csv")
    history = pd.read_csv(OUT / "history_coverage.csv")
    distribution = pd.read_csv(OUT / "ttf_distribution.csv")
    crossing = pd.read_csv(OUT / "intervening_fault_rows.csv")
    tools = pd.read_csv(OUT / "tool_event_summary.csv")
    missing_tools = sorted(set(t["tool_id"] for t in plan["tools"]) - set(tools.tool_id))
    if missing_tools:
        tools = pd.concat([tools, pd.DataFrame([{
            "tool_id": tool, "total_events": 0, "train_events": 0,
            "validation_events": 0, "reserved_or_purge_events": 0,
            "first_fault_time": None, "last_fault_time": None,
            "low_interval_median_hours": None,
            "low_interval_p75_hours": None, "low_interval_p90_hours": None,
        } for tool in missing_tools])], ignore_index=True).sort_values("tool_id")
    tools.to_csv(OUT / "tool_event_summary_all20.csv", index=False)
    continuity = []
    for minutes in (30, 60, 180):
        for phase, part in [("all", events), ("train", events[events.phase == "train"]),
                            ("validation", events[events.phase == "validation"])]:
            available = part[f"A_history_{minutes}m"]
            near = part.last_sensor_lead_hours.le(24)
            strict = part[f"A_max_gap_{minutes}m_seconds"].le(90)
            continuity.append({"history_minutes": minutes, "scope": phase,
                "A_available_at_last_observation": int(available.sum()),
                "A_available_and_last_observation_within_24h": int((available & near).sum()),
                "A_available_and_max_gap_at_most_90s": int((available & strict).sum()),
                "A_available_near_fault_and_max_gap_at_most_90s": int((available & near & strict).sum())})
    pd.DataFrame(continuity).to_csv(OUT / "continuity_sensitivity.csv", index=False)
    train = events[events.phase == "train"]
    validation = events[events.phase == "validation"]
    tv = events[events.phase.isin(["train", "validation"])]
    a1 = anchors[(anchors.rule == "A") & (anchors.history_minutes == 60)]
    b1 = anchors[(anchors.rule == "B") & (anchors.history_minutes == 60)]
    def anchor(cadence: int, rule: str) -> tuple[int, int]:
        part = a1 if rule == "A" else b1
        part = part[part.cadence_minutes == cadence]
        return int(part.candidate_samples.sum()), int(part.usable_events.sum())
    summary = {
        "model_training": False,
        "low_fault_name": "FlowCool Pressure Dropped Below Limit",
        "total_unique_low_events": len(events),
        "train_events": len(train), "validation_events": len(validation),
        "reserved_or_purge_events": int((events.phase == "reserved_or_purge").sum()),
        "train_tools": train.tool_id.nunique(), "validation_tools": validation.tool_id.nunique(),
        "common_train_validation_tools": len(set(train.tool_id) & set(validation.tool_id)),
        "event_overlap_train_validation": len(set(train.fault_event_id) & set(validation.fault_event_id)),
        "top_tool": events.tool_id.value_counts().index[0],
        "top_tool_events": int(events.tool_id.value_counts().iloc[0]),
        "top_two_tool_share_pct": float(100 * events.tool_id.value_counts().head(2).sum() / len(events)),
        "high_between_low_events": int((events.other_high_between_lows > 0).sum()),
        "leak_between_low_events": int((events.other_leak_between_lows > 0).sum()),
        "any_other_between_low_events": int(events.any_other_between_lows.sum()),
        "any_other_between_low_pct": float(100 * events.any_other_between_lows.mean()),
        "any_other_between_low_train_validation_events": int(tv.any_other_between_lows.sum()),
        "train_validation_raw_rows_with_other_fault_before_next_low_pct":
            float(100 * crossing.rows_any_intervenes.sum() / crossing.rows.sum()),
        "A_1h_history_all_events": int(events.A_history_60m.sum()),
        "B_1h_history_all_events": int(events.B_history_60m.sum()),
        "A_to_B_1h_30m_anchor_usable_events_train_validation":
            [anchor(30, "A")[1], anchor(30, "B")[1]],
        "A_1h_anchor_samples_by_cadence_10_30_60m": {str(x): anchor(x, "A")[0] for x in (10,30,60)},
        "recommended_history_minutes": 30,
        "recommendation": "Pilot only: 30m gives most events and smaller gaps; evaluate event-equal Dummy before any model claim. Keep other-fault interruption flag and cap/equal-weight events.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
