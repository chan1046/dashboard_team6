"""Read-only decomposition of existing 30-minute sequence availability."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiment_sequence_rul_baseline import ROOT, OUT, PLAN, FAULTS, SCORES1, load_tool


def main():
    destination=OUT.parent/"rul_paper_reproduction_design/sequence_coverage_reasons.json"
    if destination.exists():
        raise FileExistsError(destination)
    plan=json.loads(PLAN.read_text(encoding="utf-8"))
    faults=pd.read_csv(FAULTS).drop_duplicates(["tool_id","time","fault_type"])
    with np.load(SCORES1) as saved:
        times,codes,p=(saved[k].copy() for k in ("time","tool_code","probability"))
    counts={"model1_trigger":0,"not_aligned_to_prepared_minute":0,"fewer_than_30_prior_observations":0,
            "session_or_gap_break_in_history":0,"nonfinite_sensor_or_stage":0,"available":0}
    for code,item in enumerate(plan["tools"]):
        d=load_tool(item,faults)
        local=times[(codes==code)&(p>=.34)]
        counts["model1_trigger"]+=len(local)
        ix=np.searchsorted(d["t"],local)
        aligned=(ix<len(d["t"]))&(d["t"][np.minimum(ix,len(d["t"])-1)]==local)
        counts["not_aligned_to_prepared_minute"]+=int((~aligned).sum())
        ix=ix[aligned]
        enough=ix>=29
        counts["fewer_than_30_prior_observations"]+=int((~enough).sum())
        ix=ix[enough]
        same=d["run_start"][ix]<=ix-29
        counts["session_or_gap_break_in_history"]+=int((~same).sum())
        ix=ix[same]
        valid=0
        for start in range(0,len(ix),4096):
            batch=ix[start:start+4096]
            positions=batch[:,None]-np.arange(29,-1,-1)[None,:]
            valid+=int(np.isfinite(d["v"][positions]).all(axis=(1,2)).sum())
        counts["nonfinite_sensor_or_stage"]+=len(ix)-valid
        counts["available"]+=valid
    assert counts["model1_trigger"]==sum(v for k,v in counts.items() if k!="model1_trigger")
    destination.write_text(json.dumps(counts,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(counts,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
