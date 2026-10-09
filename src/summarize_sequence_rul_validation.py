"""Event-safe chronological stability and residual summaries for saved predictions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output/experiments/sequence_rul_baseline"


def main():
    target=OUT/"event_safe_fold_metrics.csv"
    if target.exists():
        raise FileExistsError(target)
    prediction=pd.read_csv(OUT/"validation_predictions.csv.gz")
    rows=[]
    for (window,length,model),part in prediction.groupby(["window_h","sequence_minutes","model"]):
        unique=np.sort(part.fault_time.unique())
        edges=[unique[int(len(unique)*q)] for q in (1/3,2/3)]
        fold=1+np.searchsorted(edges,part.fault_time.to_numpy(),side="right")
        for k in (1,2,3):
            sub=part.iloc[np.flatnonzero(fold==k)]
            if sub.empty:
                continue
            event_mae=sub.groupby("event_id").abs_error_h.mean().mean()
            rows.append({"window_h":window,"sequence_minutes":length,"model":model,"fold":k,
                "rows":len(sub),"events":sub.event_id.nunique(),
                "first_fault_time":int(sub.fault_time.min()),"last_fault_time":int(sub.fault_time.max()),
                "mae_h":float(sub.abs_error_h.mean()),"event_equal_mae_h":float(event_mae)})
    pd.DataFrame(rows).to_csv(target,index=False,encoding="utf-8-sig")
    chosen=prediction[(prediction.window_h==6)&(prediction.sequence_minutes==30)&(prediction.model=="Sequence_GRU")]
    residual=chosen.predicted_h.to_numpy()-chosen.ttf_h.to_numpy()
    calibration={"selected_validation_residual_quantiles_h":dict(zip(
        ("p05","p10","p25","median","p75","p90","p95"),
        map(float,np.quantile(residual,[.05,.1,.25,.5,.75,.9,.95])))),
        "selected_absolute_error_p80_h":float(np.quantile(np.abs(residual),.8)),
        "interpretation":"Empirical development-validation residuals on true 0-6h rows only. Not a calibrated prospective prediction interval; model selection reused the same validation."}
    (OUT/"residual_range_report.json").write_text(json.dumps(calibration,ensure_ascii=False,indent=2),encoding="utf-8")
    trajectory=pd.read_csv(OUT/"model1_first_alarm_rul_trajectory.csv")
    delta=trajectory.estimated_rul_change_h.dropna()
    summary={"model1_detected_events":len(trajectory),
        "events_with_any_sequence":int((trajectory.scored_trigger_rows>0).sum()),
        "events_with_two_or_more_sequence_scores":len(delta),
        "estimated_rul_change_median_h":float(delta.median()) if len(delta) else None,
        "estimated_rul_change_fraction_decreasing":float((delta<0).mean()) if len(delta) else None,
        "note":"First-to-last estimate can decrease merely because time passes; this is not proof of degradation sensitivity. Conditional GRU is applied outside its training population here."}
    (OUT/"model1_rul_trajectory_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"residual":calibration,"trajectory":summary},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
