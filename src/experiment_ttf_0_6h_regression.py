"""Conditional 0-6h TTF regression audit on actual pre-test minute observations.

The true 0-6h window is used to construct this *research* population only.
Production cannot know membership in that window without the true TTF.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from xgboost import XGBRegressor

from audit_ttf_sensor_relationship import causal_features
from compare_hgb_log_ttf import PARAMS as HGB_PARAMS
from evaluate_event_thresholds_24h import write_csv
from experiment_tool_stage_interaction_24h import file_hash
from forward_validation_ab_24h import validation_events

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "output/experiments/forward_validation_ab_24h/split_plan.json"
PREP = ROOT / "output/baseline_recovery/prepared"
OUT = ROOT / "output/experiments/ttf_0_6h_regression"
HORIZON = 21600
GAP = HORIZON
FEATURES = {
    "pressure_high": ["IONGAUGEPRESSURE", "IONGAUGEPRESSURE_std_10m"],
    "leak": ["FLOWCOOLPRESSURE", "FLOWCOOLPRESSURE_mean_30m"],
    "pressure_low": ["FLOWCOOLPRESSURE", "FLOWCOOLFLOWRATE",
                     "FLOWCOOLPRESSURE_mean_30m", "FLOWCOOLFLOWRATE_mean_30m"],
}
XGB_PARAMS = {"objective": "reg:squarederror", "n_estimators": 300,
              "learning_rate": .04, "max_depth": 3, "min_child_weight": 4,
              "subsample": .8, "colsample_bytree": .8, "reg_lambda": 2.,
              "tree_method": "hist", "n_jobs": 4, "random_state": 42}


def metrics(y, p):
    err = np.abs(y-p)
    return {"mae_minutes": float(mean_absolute_error(y,p)),
            "mae_hours": float(mean_absolute_error(y,p)/60),
            "rmse_minutes": float(np.sqrt(mean_squared_error(y,p))),
            "median_absolute_error_minutes": float(median_absolute_error(y,p)),
            "r2": float(r2_score(y,p)),
            "within_30m": float(np.mean(err<=30)),
            "within_1h": float(np.mean(err<=60)),
            "within_2h": float(np.mean(err<=120))}


def build_population(plan, events):
    frames = []
    raw = ["time", "session_id", "IONGAUGEPRESSURE", "FLOWCOOLPRESSURE",
           "FLOWCOOLFLOWRATE"]
    for item in plan["tools"]:
        name = item["tool_id"]
        faults = events[events.tool_id==name]
        if faults.empty:
            continue
        frame = pq.read_table(PREP/f"{name}.parquet", columns=raw).to_pandas()
        frame = frame.loc[(frame.time>=item["validation_start_second"]-1800) &
                          (frame.time<item["validation_end_exclusive_second"])].reset_index(drop=True)
        time = frame.time.to_numpy(dtype=np.int64)
        session = frame.session_id.to_numpy()
        ion = frame.IONGAUGEPRESSURE.to_numpy(dtype=np.float32)
        frame["IONGAUGEPRESSURE_std_10m"] = causal_features(time, session, ion)["std_10m"]
        for sensor in ("FLOWCOOLPRESSURE", "FLOWCOOLFLOWRATE"):
            frame[f"{sensor}_mean_30m"] = causal_features(
                time, session, frame[sensor].to_numpy(dtype=np.float32))["mean_30m"]
        for fault_type in FEATURES:
            fault_times = np.sort(faults.loc[faults.fault_type==fault_type,"fault_time"].to_numpy(np.int64))
            if not len(fault_times):
                continue
            index = np.searchsorted(fault_times, time, side="right")
            possible = index<len(fault_times)
            next_time = fault_times[np.minimum(index,len(fault_times)-1)]
            ttf = next_time-time
            chosen = possible & (ttf>0) & (ttf<=HORIZON) & (time>=item["validation_start_second"])
            if not np.any(chosen):
                continue
            selected = frame.loc[chosen, FEATURES[fault_type]].copy()
            selected.insert(0,"tool_id",name)
            selected.insert(1,"fault_type",fault_type)
            selected.insert(2,"time",time[chosen])
            selected.insert(3,"fault_time",next_time[chosen])
            selected.insert(4,"event_id",[f"{name}|{fault_type}|{ft}" for ft in next_time[chosen]])
            selected.insert(5,"ttf_minutes",ttf[chosen]/60.0)
            frames.append(selected)
        print(f"{name}: {len(frame):,} pre-test minute rows", flush=True)
    return pd.concat(frames, ignore_index=True)


def plot_result(valid, fault, model):
    # Display is decimated only; all rows are included in metrics and CSV.
    shown = valid.iloc[::max(1,len(valid)//5000)]
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    axes[0].scatter(shown.ttf_minutes, shown.predicted_minutes,s=3,alpha=.25)
    axes[0].plot([0,360],[0,360],"k--",linewidth=1)
    axes[0].set(xlabel="True TTF (minutes)",ylabel="Predicted TTF (minutes)",title=f"{fault} / {model}")
    axes[1].scatter(shown.ttf_minutes,shown.predicted_minutes-shown.ttf_minutes,s=3,alpha=.25)
    axes[1].axhline(0,color="black",linestyle="--",linewidth=1)
    axes[1].set(xlabel="True TTF (minutes)",ylabel="Residual (prediction - truth)")
    fig.tight_layout()
    fig.savefig(OUT/f"scatter_residual_{fault}_{model}.png",dpi=150)
    plt.close(fig)


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment: {OUT}")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    events = pd.DataFrame(validation_events(plan))
    population = build_population(plan,events)
    OUT.mkdir(parents=True)
    all_metrics, splits, event_metrics, bin_metrics, predictions = [], [], [], [], []
    for fault in ("pressure_high","leak","pressure_low"):
        data = population[population.fault_type==fault].copy()
        event_times = np.sort(data.fault_time.unique())
        if len(event_times)<6:
            raise AssertionError(f"Too few {fault} events for time split")
        boundary = int(event_times[int(len(event_times)*.7)])
        train = data[data.fault_time < boundary-GAP]
        valid = data[data.fault_time >= boundary+GAP]
        train_events, valid_events = set(train.event_id), set(valid.event_id)
        if not train_events or not valid_events or train_events & valid_events:
            raise AssertionError(f"Invalid event-separated split: {fault}")
        if train.time.max() >= valid.time.min():
            raise AssertionError(f"Time overlap: {fault}")
        features = FEATURES[fault]
        xtrain = train[features].to_numpy(np.float32)
        xvalid = valid[features].to_numpy(np.float32)
        ytrain = train.ttf_minutes.to_numpy(float)
        yvalid = valid.ttf_minutes.to_numpy(float)
        splits.append({"fault_type":fault,"candidate_events":data.event_id.nunique(),
            "train_events":len(train_events),"validation_events":len(valid_events),
            "train_rows":len(train),"validation_rows":len(valid),
            "boundary_second":boundary,"purge_seconds_each_side":GAP,
            "train_last_time":int(train.time.max()),"validation_first_time":int(valid.time.min()),
            "feature_names":",".join(features)})
        models = {"Dummy_median": DummyRegressor(strategy="median"),
            "HGB_log": HistGradientBoostingRegressor(**HGB_PARAMS),
            "XGBoost": XGBRegressor(**XGB_PARAMS)}
        for model_name, model in models.items():
            target = np.log1p(ytrain*60) if model_name=="HGB_log" else ytrain
            model.fit(xtrain,target)
            predicted = model.predict(xvalid)
            predicted = np.expm1(predicted)/60 if model_name=="HGB_log" else predicted
            predicted = np.clip(predicted,0,360)
            result = {"fault_type":fault,"model":model_name,
                "train_rows":len(train),"validation_rows":len(valid),
                "train_events":len(train_events),"validation_events":len(valid_events),
                **metrics(yvalid,predicted)}
            all_metrics.append(result)
            pred = valid[["tool_id","fault_type","time","fault_time","event_id","ttf_minutes"]].copy()
            pred["model"] = model_name
            pred["predicted_minutes"] = predicted
            pred["residual_minutes"] = predicted-yvalid
            predictions.append(pred)
            plot_result(pred,fault,model_name)
            for eid, group in pred.groupby("event_id"):
                event_metrics.append({"fault_type":fault,"model":model_name,"event_id":eid,
                    "tool_id":group.tool_id.iloc[0],"rows":len(group),
                    "mae_minutes":float(np.mean(np.abs(group.residual_minutes))),
                    "median_absolute_error_minutes":float(np.median(np.abs(group.residual_minutes)))})
            for lo,hi in ((0,120),(120,240),(240,360)):
                mask = (yvalid>lo) & (yvalid<=hi)
                if mask.any():
                    bin_metrics.append({"fault_type":fault,"model":model_name,
                        "actual_ttf_bin":f"{lo//60}-{hi//60}h", "rows":int(mask.sum()),
                        "events":int(valid.loc[mask,"event_id"].nunique()),
                        **metrics(yvalid[mask],predicted[mask]),
                        "mean_prediction_minutes":float(np.mean(predicted[mask]))})
            print(json.dumps(result),flush=True)
    pd.DataFrame(all_metrics).to_csv(OUT/"metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(splits).to_csv(OUT/"split_summary.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(event_metrics).to_csv(OUT/"event_metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(bin_metrics).to_csv(OUT/"ttf_bin_metrics.csv",index=False,encoding="utf-8-sig")
    pd.concat(predictions).to_csv(OUT/"validation_predictions.csv.gz",index=False,compression="gzip")
    report = {"population": "Actual first-stage forward validation only; 1-minute prepared sensor observations within true 0-6h before next same-type event",
        "conditional_selection_caveat": "True TTF is used to select 0-6h rows for this research evaluation. An online system cannot know this membership; these results do not validate end-to-end predicted TTF.",
        "ttf_feature_check": "No TTF, fault time, event ID, absolute time, run number, or lot in FEATURES",
        "split": "Per fault type, global chronological event-time 70/30 with 6h purge each side; no event in both subsets",
        "same_validation_reuse_caveat": "This population reuses the previously analyzed forward validation. It is exploratory, not independent final validation; reserved test remains unopened.",
        "feature_causality": "ION 10-minute std and FlowCool 30-minute means are recomputed using elapsed seconds and only same-session current and preceding observations. Existing baseline mean_30m fields are row-count windows and are not used.",
        "hgb_existing_parameters":HGB_PARAMS,"xgboost_existing_parameters":XGB_PARAMS,
        "eda_report_sha256":file_hash(ROOT/"output/experiments/ttf_sensor_relationship/report.json"),
        "split_summary":splits}
    (OUT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
