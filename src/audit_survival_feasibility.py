"""Read-only failure-cycle feasibility audit of all actual train sensor/fault data.

No model is fitted. Time-to-next-event, type, and censoring are labels only.
Sensor history is counted strictly before a fault and after prior fault.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
OUT = ROOT / "output/experiments/survival_feasibility"
FAULTS = {"pressure_low", "pressure_high", "leak"}


def groups_with_gap(times, seconds):
    return int(1 + np.count_nonzero(np.diff(times)>seconds)) if len(times) else 0


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    manifest = pd.read_csv(DATA/"sensor_ttf_manifest.csv")
    raw = pd.read_csv(DATA/"fault_events.csv")
    if manifest.tool_id.nunique()!=20 or set(raw.fault_type)!=FAULTS:
        raise AssertionError("Unexpected tool or fault population")
    unique = raw.sort_values(["tool_id","time","fault_type","fault_event_id"]).drop_duplicates(
        ["tool_id","time","fault_type"],keep="first")
    if len(unique)!=1211 or len(raw)!=1236:
        raise AssertionError("Fault source changed; re-audit deduplication")
    cycles, tools, out_of_range, close_pairs, fault_rows = [], [], [], [], []
    for item in manifest.itertuples():
        name = item.tool_id
        path = DATA/f"{name}_sensor_ttf.parquet"
        pf = pq.ParquetFile(path)
        if pf.metadata.num_rows != item.rows:
            raise AssertionError(f"Sensor manifest row mismatch: {name}")
        t = pf.read(columns=["time"]).column("time").to_numpy().astype(np.int64)
        if len(t)!=item.rows or np.any(np.diff(t)<0):
            raise AssertionError(f"Sensor time order mismatch: {name}")
        first, last = int(t[0]), int(t[-1])
        if first!=item.first_time or last!=item.last_time:
            raise AssertionError(f"Sensor bounds mismatch: {name}")
        events = unique[unique.tool_id==name].sort_values("time")
        before = events[events.time<first]
        after = events[events.time>last]
        for row in pd.concat([before,after]).itertuples():
            out_of_range.append({"tool_id":name,"fault_type":row.fault_type,"fault_time":int(row.time),
                "reason":"before_first_sensor" if row.time<first else "after_last_sensor"})
        in_range = events[(events.time>=first)&(events.time<=last)]
        fault_times = in_range.time.to_numpy(np.int64)
        fault_types = in_range.fault_type.to_numpy()
        full_times = events.time.to_numpy(np.int64)
        for a,b in zip(full_times[:-1],full_times[1:]):
            gap=int(b-a)
            if gap<=24*3600:
                close_pairs.append({"tool_id":name,"earlier_fault_time":int(a),
                    "later_fault_time":int(b),"gap_seconds":gap,
                    "has_sensor_between":bool(np.searchsorted(t,b,side="left")>
                                              np.searchsorted(t,a,side="right"))})
        for i in range(len(fault_times)+1):
            start = first if i==0 else int(fault_times[i-1])
            end = last if i==len(fault_times) else int(fault_times[i])
            # Prevent post-event sensor values from entering the event cycle.
            left = int(np.searchsorted(t,start,side="left" if i==0 else "right"))
            right = int(np.searchsorted(t,end,side="right" if i==len(fault_times) else "left"))
            history = t[left:right]
            lag = int(end-history[-1]) if len(history) else None
            gaps = np.diff(history) if len(history)>1 else np.empty(0,dtype=np.int64)
            is_event = i<len(fault_times)
            cycles.append({"tool_id":name,"cycle_index":i,"cycle_id":f"{name}|{i}",
                "start_time":start,"end_time":end,"duration_seconds":end-start,
                "duration_hours":(end-start)/3600,"event_observed":is_event,
                "censored_at_data_end":not is_event,
                "fault_type":str(fault_types[i]) if is_event else "",
                "fault_time":end if is_event else None,
                "initial_observation_cycle":i==0,
                "prior_logged_fault_before_sensor_start":bool(len(before)) if i==0 else False,
                "sensor_rows_strictly_inside":len(history),
                "first_sensor_time":int(history[0]) if len(history) else None,
                "last_sensor_time":int(history[-1]) if len(history) else None,
                "gap_from_last_sensor_to_cycle_end_seconds":lag,
                "within_cycle_gaps_over_120s":int(np.count_nonzero(gaps>120)),
                "within_cycle_gaps_over_1h":int(np.count_nonzero(gaps>3600)),
                "max_observed_gap_seconds":int(gaps.max()) if len(gaps) else None})
        tools.append({"tool_id":name,"sensor_rows":len(t),"first_sensor_time":first,
            "last_sensor_time":last,"raw_fault_rows":int((raw.tool_id==name).sum()),
            "unique_fault_events":len(events),"events_before_sensor":len(before),
            "events_after_sensor":len(after),"in_range_fault_events":len(in_range),
            "cycles":len(fault_times)+1,"event_cycles":len(fault_times),
            "censored_cycles":1,"repeat_faults_after_first":max(len(events)-1,0),
            "clusters_10m":groups_with_gap(full_times,600),
            "clusters_30m":groups_with_gap(full_times,1800)})
        fault_rows.extend({"tool_id":name,"fault_type":r.fault_type,"fault_time":int(r.time),
                           "in_sensor_range":first<=r.time<=last} for r in events.itertuples())
        print(f"{name}: {len(events)} unique faults; {len(in_range)+1} observable cycles",flush=True)
    c=pd.DataFrame(cycles)
    per_tool=pd.DataFrame(tools)
    near=pd.DataFrame(close_pairs)
    fr=pd.DataFrame(fault_rows)
    event_cycles = c[c.event_observed]
    usable = event_cycles[event_cycles.sensor_rows_strictly_inside>0]
    censored = c[c.censored_at_data_end]
    if len(c)!=int(per_tool.cycles.sum()) or int(c.event_observed.sum())!=int(per_tool.in_range_fault_events.sum()):
        raise AssertionError("Cycle accounting mismatch")
    OUT.mkdir(parents=True)
    c.to_csv(OUT/"failure_cycles.csv",index=False,encoding="utf-8-sig")
    per_tool.to_csv(OUT/"tool_summary.csv",index=False,encoding="utf-8-sig")
    near.to_csv(OUT/"close_fault_pairs_le_24h.csv",index=False,encoding="utf-8-sig")
    fr.to_csv(OUT/"unique_fault_events.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(out_of_range).to_csv(OUT/"faults_outside_sensor_range.csv",index=False,encoding="utf-8-sig")
    duration=c.groupby("event_observed").duration_hours.agg(["count","min","median","max"])
    q=c.groupby("event_observed").duration_hours.quantile([.1,.25,.75,.9,.95]).unstack()
    duration.join(q).reset_index().to_csv(OUT/"cycle_duration_distribution.csv",index=False,encoding="utf-8-sig")
    report={"scope":"All 20 real train sensor partitions and full train fault log; no model fitted",
        "raw_fault_rows":len(raw),"unique_fault_events":len(unique),
        "duplicates_same_tool_type_time":len(raw)-len(unique),
        "unique_faults_by_type":unique.groupby("fault_type").size().to_dict(),
        "faults_before_first_sensor":sum(r["reason"]=="before_first_sensor" for r in out_of_range),
        "faults_after_last_sensor":sum(r["reason"]=="after_last_sensor" for r in out_of_range),
        "cycle_count":len(c),"event_cycles":len(event_cycles),"censored_cycles":len(censored),
        "event_cycles_with_sensor_history":len(usable),
        "event_cycles_with_sensor_history_by_type":usable.groupby("fault_type").size().to_dict(),
        "event_cycles_without_sensor_history":int((event_cycles.sensor_rows_strictly_inside==0).sum()),
        "event_cycles_with_recent_sensor_le_1h":int((event_cycles.gap_from_last_sensor_to_cycle_end_seconds<=3600).sum()),
        "event_cycles_with_recent_sensor_le_1h_by_type":event_cycles[event_cycles.gap_from_last_sensor_to_cycle_end_seconds<=3600].groupby("fault_type").size().to_dict(),
        "censored_cycles_with_sensor_history":int((censored.sensor_rows_strictly_inside>0).sum()),
        "recurrent_fault_tools":int((per_tool.unique_fault_events>=2).sum()),
        "repeated_faults_beyond_first":int(per_tool.repeat_faults_after_first.sum()),
        "fault_gap_pairs_le_1m":int((near.gap_seconds<=60).sum()),
        "fault_gap_pairs_le_10m":int((near.gap_seconds<=600).sum()),
        "fault_gap_pairs_le_30m":int((near.gap_seconds<=1800).sum()),
        "fault_gap_pairs_le_6h":int((near.gap_seconds<=21600).sum()),
        "fault_gap_pairs_le_24h":len(near),
        "fault_gap_pairs_le_30m_without_sensor_between":int(((near.gap_seconds<=1800)&~near.has_sensor_between).sum()),
        "illustrative_30m_fault_clusters_all_unique_records":int(per_tool.clusters_30m.sum()),
        "event_cycles_last_sensor_more_than_1h_before_fault":int((event_cycles.gap_from_last_sensor_to_cycle_end_seconds>3600).sum()),
        "event_cycles_last_sensor_more_than_24h_before_fault":int((event_cycles.gap_from_last_sensor_to_cycle_end_seconds>86400).sum()),
        "event_cycle_duration_hours":event_cycles.duration_hours.quantile([0,.1,.25,.5,.75,.9,1]).to_dict(),
        "censored_cycle_duration_hours":censored.duration_hours.quantile([0,.1,.25,.5,.75,.9,1]).to_dict(),
        "cycle_definition":"Per tool, first sensor time to first in-range fault; then successive fault timestamps; last in-range fault to last sensor timestamp is right-censored. Event-time sensor rows are excluded; post-fault rows belong only to the following cycle.",
        "interpretation_caveat":"These are observed inter-fault intervals, not proven repaired/restarted operational cycles. Initial cycle is left-truncated/unknown age. Closely spaced faults may represent one unresolved episode. Gaps in sensor observations are stops, not filled exposure.",
        "target_only":"Future fault time, TTF, event status and fault type are labels only; no feature matrix was produced.",
        "candidate":"Discrete-time, time-varying, recurrent competing-risks survival dataset, after defining episode merging and off-state treatment; do not infer calibrated time predictions from structure alone."}
    (OUT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("unique_fault_events","cycle_count","event_cycles","censored_cycles","event_cycles_with_sensor_history","fault_gap_pairs_le_30m")}),flush=True)


if __name__=="__main__":
    main()
