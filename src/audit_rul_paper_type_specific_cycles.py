"""Count fault-type-specific run-to-fault history without fitting a model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output/experiments/rul_paper_reproduction_design"


def main():
    target=OUT/"paper_type_specific_cycles.csv"
    if target.exists():
        raise FileExistsError(target)
    faults=pd.read_csv(ROOT/"data/processed/fault_events.csv").drop_duplicates(
        ["tool_id","time","fault_type"]).sort_values(["tool_id","fault_type","time"])
    rows=[]
    for tool,subset in faults.groupby("tool_id"):
        table=pq.read_table(ROOT/f"data/processed/{tool}_sensor_ttf.parquet",
                            columns=["time","FIXTURESHUTTERPOSITION"])
        t=table.column("time").to_numpy().astype(np.int64)
        shutter=pd.to_numeric(table.column("FIXTURESHUTTERPOSITION").to_pandas(),errors="coerce").to_numpy()
        if np.any(np.diff(t)<0):
            order=np.argsort(t,kind="stable");t=t[order];shutter=shutter[order]
        active=np.r_[0,np.cumsum((shutter==1).astype(np.int64))]
        big_gap=np.flatnonzero(np.diff(t)>1800)+1
        for fault_type,events in subset.groupby("fault_type"):
            ft=events.time.to_numpy(np.int64)
            for i,fault_time in enumerate(ft):
                start=int(np.searchsorted(t,ft[i-1],side="right")) if i else 0
                end=int(np.searchsorted(t,fault_time,side="left"))
                start5=max(start,int(np.searchsorted(t,fault_time-18000,side="left")))
                gaps=big_gap[(big_gap>start5)&(big_gap<end)]
                tail_start=int(gaps[-1]) if len(gaps) else start5
                last=t[end-1] if end>start else None
                rows.append({"tool_id":tool,"fault_type":fault_type,"fault_time":int(fault_time),
                    "type_cycle_rows":max(0,end-start),
                    "type_cycle_shutter1_rows":int(active[end]-active[start]),
                    "last5h_rows":max(0,end-start5),
                    "last5h_shutter1_rows":int(active[end]-active[start5]),
                    "last5h_tail_after_30m_gap_rows":max(0,end-tail_start),
                    "last_sensor_gap_seconds":int(fault_time-last) if last is not None else None})
    frame=pd.DataFrame(rows)
    assert len(frame)==1211
    frame.to_csv(target,index=False,encoding="utf-8-sig")
    criteria={"any_type_cycle_sensor":frame.type_cycle_rows>0,
        "paper1_any_2000_shutter1_type_cycle_upper":frame.type_cycle_shutter1_rows>=2000,
        "paper1_2000_shutter1_last5h_upper":frame.last5h_shutter1_rows>=2000,
        "paper3_3000_last5h_upper":frame.last5h_rows>=3000,
        "paper3_3000_last5h_tail_30m_gap_upper":frame.last5h_tail_after_30m_gap_rows>=3000,
        "paper3_3000_tail_recent_1h_upper":(frame.last5h_tail_after_30m_gap_rows>=3000)&
                                              (frame.last_sensor_gap_seconds<=3600)}
    report={"total_unique_fault_events":len(frame),
        "counts":{key:int(value.sum()) for key,value in criteria.items()},
        "by_fault_type":{typ:{key:int(value.loc[part.index].sum()) for key,value in criteria.items()}
                         for typ,part in frame.groupby("fault_type")},
        "method":"Per tool and fault type, history begins after previous fault of that same type; intervening other-type faults are retained only as prior history and must be explicitly flagged in any eventual training set. All counts are potential upper bounds, before sensor-missingness screening and train/validation boundaries.",
        "gap_sensitivity":"30-minute last-gap threshold is an explicit local sensitivity assumption, not a number reported by the paper.",
        "paper1_note":"The 2000-point paper windows can occur anywhere in a type-specific history, so full-cycle shutter=1 count is the broad upper bound; last-5h count is a narrower optional design.",
        "paper3_note":"Paper reports maxL=300, SampleR=15, N=15, Mmin=3000 (about 5h history) after a large-gap truncation. Raw row counts alone do not guarantee valid 21-dimensional feature sequences."}
    (OUT/"paper_specific_availability.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
