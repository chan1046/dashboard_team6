# 12h/24h wall-clock validation: 100% eligible train data

This is a **new experiment report**, not the deleted historical workbook. It
uses real PHM 2018 train data from all 20 tools. Every observed 4-second sensor
row contributes to one-minute aggregates. For each tool, the earlier roughly
80% is training and the later roughly 20% is validation; the 24 hours before
the cutoff are purged from training. The same 1,142,112 labeled validation
minutes are used for 12h and 24h classification.

## Data flow and targets

1. One-minute current sensor and process values plus past-only rolling,
   difference, slope, EWMA, gap, and restart features form 85 inputs. Raw TTF,
   fault time/type, absolute time, Lot, and runnum are not inputs.
2. The binary target is 1 when the next logged low-pressure, high-pressure, or
   leak event occurs within 12h or 24h of the row's **actual timestamp**.
   Equipment stops consume wall-clock horizon time. Incomplete follow-up is
   censored rather than declared normal.
3. Each LSTM learns one same-type time-to-fault target independently. All
   eligible training rows with a known same-type event **before the validation
   cutoff** are used, whether the binary classifier would alert or not.
   A five-minute sequence never crosses a stop longer than two minutes.
4. At inference only, a binary alert gates three independent TTF predictions;
   the shortest prediction selects the proposed type and RUL.

## Findings

The 12h/24h XGBoost 100% models are existing real-data runs. Their saved
validation probabilities were recomputed and verified in this experiment.
At threshold 0.5, 12h precision/recall/AP are 9.71%/48.51%/9.95%; 24h are
17.44%/49.30%/16.68%. The 24h result is exactly the recovered baseline, so
there is no measured improvement over it.

The three newly trained LSTM regressions all have negative R² and large MAE.
At the first alert before a detected fault, minimum-predicted-TTF typing is
correct for only 30/106 (12h) and 32/112 (24h) events. The predicted minimum
favors leak (60/106 and 67/112) while pressure_low is the most frequent actual
type among detected events (54/106 and 57/112). The connected output is not
reliable enough for operational RUL/type display.

HGB classification and HGB-log regression were attempted but **not run**:
Windows application control blocks scikit-learn's histogram extension with
WinError 4551. Historical scores from a different experiment were not used to
fill this gap. The workbook explicitly marks the requested model comparison
as incomplete. There is no untouched future test interval after this
validation; replaying its 20% in a dashboard is a demonstration, not a fresh
final performance test.

## Files

- `시계시간_12h_24h_100퍼센트_모델비교_연결검증포함.xlsx`: readable report.
- `classification_metrics.csv`: 12h/24h row and event classification metrics.
- `regression_metrics.csv`: independently evaluated three-type LSTM metrics.
- `unavailable_models.csv`: HGB execution failures, without invented scores.
- `report.json`: detailed settings and limitations.
- `../connected_lstm_100pct_12h_24h/fault_event_details.csv`: event-by-event
  classification and predicted-type audit.

Next controlled experiment: execute the blocked HGB/HGB-log families in an
environment where scikit-learn's histogram module can load, using this same
data split and feature definition before making a model ranking.
