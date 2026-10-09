"""Read-only count of paper-style run-to-fault observation availability.

Counts are candidate upper bounds: no model fitting and no sequence tensors.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output/experiments/rul_paper_reproduction_design"
FAULTS=ROOT/"data/processed/fault_events.csv"
RAW=ROOT/"data/processed"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    log=pd.read_csv(FAULTS).sort_values(["tool_id","time","fault_type","fault_event_id"])
    log=log.drop_duplicates(["tool_id","time","fault_type"],keep="first")
    if log.duplicated(["tool_id","time"]).any():
        raise AssertionError("Simultaneous events need a policy")
    records=[]
    for tool, events in log.groupby("tool_id",sort=True):
        path=RAW/f"{tool}_sensor_ttf.parquet"
        tab=pq.read_table(path,columns=["time","FIXTURESHUTTERPOSITION"])
        t=tab.column("time").to_numpy().astype(np.int64)
        shutter=pd.to_numeric(tab.column("FIXTURESHUTTERPOSITION").to_pandas(),errors="coerce").to_numpy()
        if np.any(np.diff(t)<0):
            order=np.argsort(t,kind="stable");t=t[order];shutter=shutter[order]
        active=(shutter==1).astype(np.int64)
        active_cumulative=np.r_[0,np.cumsum(active)]
        gap_index=np.flatnonzero(np.diff(t)>1800)+1
        for ordinal, event in enumerate(events.itertuples(index=False)):
            fault=int(event.time)
            previous=int(events.iloc[ordinal-1].time) if ordinal else int(t[0])-1
            end=int(np.searchsorted(t,fault,side="left"))
            cycle_start=int(np.searchsorted(t,previous,side="right"))
            start5=max(cycle_start,int(np.searchsorted(t,fault-5*3600,side="left")))
            last_gap=gap_index[gap_index<end]
            tail_start=max(start5,int(last_gap[-1]) if len(last_gap) else start5)
            last_sensor=int(t[end-1]) if end>cycle_start else None
            cycle_rows=max(0,end-cycle_start)
            five_rows=max(0,end-start5)
            tail_rows=max(0,end-tail_start)
            records.append({"tool_id":tool,"fault_time":fault,"fault_type":event.fault_type,
                "event_id":f"{tool}|{fault}","previous_fault_time":previous if ordinal else None,
                "cycle_observations_4s":cycle_rows,
                "last_5h_observations_4s":five_rows,
                "last_5h_shutter1_observations":int(active_cumulative[end]-active_cumulative[start5]),
                "last_5h_tail_after_30m_gap_observations":tail_rows,
                "last_sensor_to_fault_seconds":fault-last_sensor if last_sensor is not None else None,
                "interfault_seconds":fault-previous if ordinal else None})
        print(tool,len(t),len(events),flush=True)
    frame=pd.DataFrame(records)
    if len(frame)!=1211 or frame.event_id.duplicated().any():
        raise AssertionError("Unique fault population changed")
    frame["any_cycle_sensor"]=frame.cycle_observations_4s>0
    frame["paper1_2000_shutter1_cycle_upper"]=frame.last_5h_shutter1_observations>=2000
    frame["paper3_3000_raw_5h_upper"]=frame.last_5h_observations_4s>=3000
    frame["paper3_3000_tail_30m_gap_upper"]=frame.last_5h_tail_after_30m_gap_observations>=3000
    frame["paper3_3000_tail_and_recent_1h_upper"]=(frame.paper3_3000_tail_30m_gap_upper &
                                                      (frame.last_sensor_to_fault_seconds<=3600))
    OUT.mkdir(parents=True)
    frame.to_csv(OUT/"fault_event_history_audit.csv",index=False,encoding="utf-8-sig")
    flag_cols=["any_cycle_sensor","paper1_2000_shutter1_cycle_upper","paper3_3000_raw_5h_upper",
               "paper3_3000_tail_30m_gap_upper","paper3_3000_tail_and_recent_1h_upper"]
    summary={"unique_fault_events":len(frame),"by_type":frame.fault_type.value_counts().to_dict(),
        "candidate_counts":{col:int(frame[col].sum()) for col in flag_cols},
        "candidate_by_type":{typ:{col:int(part[col].sum()) for col in flag_cols}
            for typ,part in frame.groupby("fault_type")},
        "definitions":{"cycle":"After previous unique fault, before current fault, same tool. Initial cycle begins at first sensor observation.",
            "paper1_2000_shutter1_cycle_upper":"At least 2000 raw 4-second observations with shutter=1 in the 5h before the event, after previous fault; only an upper bound for 2000-point windows because missing sensor values and source authors' exact screening are not checked.",
            "paper3_3000_raw_5h_upper":"At least 3000 raw observations within last 5h after previous fault; upper bound for paper Mmin=3000.",
            "paper3_3000_tail_30m_gap_upper":"As above, but truncate history after the latest internal >30-minute gap. Thirty minutes is our explicit sensitivity rule; paper does not state a numerical large-gap threshold.",
            "paper3_3000_tail_and_recent_1h_upper":"Tail criterion and latest sensor at most 1h before fault. One-hour recency is our application rule, not a paper rule."},
        "limitations":"These are event history availability estimates, not actual usable model sequences or performance. Exact paper count requires full sensor missingness, sample screening, onset labeling and fold-specific construction. No synthetic rows or model fits."}
    (OUT/"availability_report.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
