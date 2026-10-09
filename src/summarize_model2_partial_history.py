"""Add set-overlap accounting and verify the partial-history count tables."""
from __future__ import annotations

import pandas as pd

from audit_model2_partial_history import CANDIDATES, OUT


def counts(q, condition, phase, typ):
    selected = pd.Series(True, index=q.index)
    if phase != "combined":
        selected &= q.phase.eq(phase)
    if typ != "total":
        selected &= q.fault_type.eq(typ)
    return int((condition & ~q.eligible & selected).sum()), int((~condition & q.eligible & selected).sum())


def main():
    q = pd.read_csv(OUT / "history_quality.csv")
    c = pd.read_csv(OUT / "candidate_comparison.csv")
    c["new_not_strict"] = 0
    c["strict_lost"] = 0
    for i, row in c.iterrows():
        if row.candidate == "strict":
            continue
        ratio, span, age = CANDIDATES[row.candidate]
        condition = (q.observation_ratio >= ratio) & q.coverage_duration_seconds.ge(span) & q.last_sensor_age_seconds.le(age)
        added, lost = counts(q, condition, row.phase, row.fault_type)
        c.loc[i, ["new_not_strict", "strict_lost"]] = added, lost
        assert row.events - row.strict_events == added - lost
    c.to_csv(OUT / "candidate_comparison.csv", index=False, encoding="utf-8-sig")
    t = pd.read_csv(OUT / "threshold_event_counts.csv")
    t["new_not_strict"] = 0
    t["strict_lost"] = 0
    for i, row in t.iterrows():
        age = None if str(row.freshness_max_seconds) == "none" else int(row.freshness_max_seconds)
        condition = (q.observation_ratio >= row.ratio_min) & q.coverage_duration_seconds.ge(int(row.coverage_min_seconds))
        if age is not None:
            condition &= q.last_sensor_age_seconds.le(age)
        added, lost = counts(q, condition, row.phase, row.fault_type)
        t.loc[i, ["new_not_strict", "strict_lost"]] = added, lost
        assert row.usable_events - row.strict_events == added - lost
    t.to_csv(OUT / "threshold_event_counts.csv", index=False, encoding="utf-8-sig")
    tool = pd.read_csv(OUT / "tool_distribution.csv")
    tool["new_not_strict"] = 0
    tool["strict_lost"] = 0
    for i, row in tool.iterrows():
        if row.candidate == "strict":
            continue
        ratio, span, age = CANDIDATES[row.candidate]
        condition = (q.observation_ratio >= ratio) & q.coverage_duration_seconds.ge(span) & q.last_sensor_age_seconds.le(age)
        phase = q.phase.eq(row.phase) if row.phase != "combined" else True
        target = q.tool_id.eq(row.tool_id) & phase
        added = int((condition & ~q.eligible & target).sum())
        lost = int((~condition & q.eligible & target).sum())
        tool.loc[i, ["new_not_strict", "strict_lost"]] = added, lost
        assert row.added_events == added - lost
    tool.to_csv(OUT / "tool_distribution.csv", index=False, encoding="utf-8-sig")
    print(c[c.phase.isin(["train", "validation"]) & c.fault_type.eq("total")].to_string(index=False))


if __name__ == "__main__":
    main()
