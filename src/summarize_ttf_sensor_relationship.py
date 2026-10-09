"""Summarize the fixed descriptive TTF audit without fitting a model."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/experiments/ttf_sensor_relationship"
PAIRS = (("0-6h", "48-72h"), ("6-12h", "48-72h"),
         ("12-24h", "48-72h"), ("24-48h", ">72h_to_7d"))


def main():
    p = OUT / "event_bin_feature_stats.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    if (OUT / "paired_interval_separation.csv").exists():
        raise FileExistsError("Summary already exists")
    data = pd.read_csv(p, usecols=["event_id", "tool_id", "fault_type", "ttf_bin", "feature", "median", "std", "q25", "q75"])
    rows = []
    for near, far in PAIRS:
        both = data[data.ttf_bin.isin((near, far))].pivot_table(
            index=["event_id", "tool_id", "fault_type", "feature"],
            columns="ttf_bin", values="median", aggfunc="first").dropna(subset=[near, far]).reset_index()
        for (fault, feature), group in both.groupby(["fault_type", "feature"]):
            x = group[near].to_numpy(float)
            y = group[far].to_numpy(float)
            if len(x) < 2:
                continue
            shift = x-y
            ranks = rankdata(np.r_[x,y])
            auc = (ranks[:len(x)].sum()-len(x)*(len(x)+1)/2)/len(x)**2
            abs_auc = max(auc,1-auc)
            direction = np.sign(np.median(shift))
            tool_signs = group.assign(delta=shift).groupby("tool_id")["delta"].median()
            supporting = int((np.sign(tool_signs)==direction).sum()) if direction else 0
            rows.append({"fault_type": fault, "near_bin": near, "far_bin": far,
                "feature": feature, "paired_events": len(group), "tools": group.tool_id.nunique(),
                "median_near_minus_far": float(np.median(shift)),
                "increases": int((shift>0).sum()), "decreases": int((shift<0).sum()),
                "direction_consistency": max((shift>0).sum(),(shift<0).sum())/len(shift),
                "supporting_tool_direction": supporting,
                "auc_direction_free": float(abs_auc),
                "ks_event_medians": float(ks_2samp(x,y).statistic),
                "near_event_median_iqr": float(np.quantile(x,.75)-np.quantile(x,.25)),
                "far_event_median_iqr": float(np.quantile(y,.75)-np.quantile(y,.25)),
                "shift_vs_pooled_event_median_iqr": float(abs(np.median(shift))/(np.quantile(np.r_[x,y],.75)-np.quantile(np.r_[x,y],.25))) if np.quantile(np.r_[x,y],.75)>np.quantile(np.r_[x,y],.25) else None})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT/"paired_interval_separation.csv", index=False, encoding="utf-8-sig")
    print(summary[(summary.near_bin=="0-6h") & (summary.far_bin=="48-72h") &
          (summary.paired_events>=10) & (summary.feature.str.startswith(("FLOWCOOL", "IONGAUGE", "ETCHGAS", "ETCHPB")))].sort_values(
          ["fault_type", "auc_direction_free"], ascending=[True,False]).groupby("fault_type").head(10)[
          ["fault_type", "feature", "paired_events", "tools", "direction_consistency",
           "supporting_tool_direction", "auc_direction_free", "ks_event_medians",
           "shift_vs_pooled_event_median_iqr"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
