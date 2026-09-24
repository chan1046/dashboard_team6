# Project: FlowCool Predictive Maintenance

## Goal
반도체 냉각설비 센서 데이터로 12h/24h 내 고장 여부를 예측하고,
결과를 Dash 예지보전 대시보드에 연결한다.

## Current baseline
24h XGBoost:
- Precision: 16.7%
- Recall: 50.6%
- F1: 25.1%
- Balanced Accuracy: 67.4%
- AP: 17.2%
- FP: 170002
- FN: 33238
- TP: 34100
- False alarm events (10 min): 4052

## Evaluation priority
Accuracy보다 아래 지표를 우선한다.
1. AP / PR-AUC
2. Recall
3. Precision / F1
4. Balanced Accuracy
5. 10분 기준 False Alarm Event 수

Recall만 높이고 오경보가 크게 증가하는 모델은 좋은 결과로 간주하지 않는다.

## Data rules
- 실제 데이터만 사용한다.
- 임의 데이터 생성 금지.
- 시계열 데이터 누수를 방지한다.
- 미래 데이터가 feature에 들어가지 않도록 한다.
- 같은 고장 이벤트 주변 데이터가 train/validation에 섞이는지 확인한다.
- 데이터 분할은 시간 또는 고장 이벤트 기준을 우선한다.

## Experiment order
아래 순서로 개선한다.

1. Baseline 재현
2. Data leakage 검사
3. Threshold sweep
4. Feature engineering
5. Class imbalance 처리
6. Hyperparameter tuning
7. 다른 모델 비교

한 번에 여러 요소를 변경하지 않는다.

## Feature engineering candidates
주요 센서에 대해:
- current value
- diff
- rate of change
- rolling mean
- rolling std
- rolling min
- rolling max
- slope

과거 데이터만 사용해 계산한다.

## Class imbalance
XGBoost의 scale_pos_weight 등 class weight를 비교한다.
단순 accuracy 개선이 아니라 AP, Recall, False Alarm 기준으로 평가한다.

## Threshold
threshold=0.5를 고정하지 않는다.

PR curve와 threshold sweep을 사용해:
- Recall 50%
- Recall 60%
- Recall 70%

부근의 Precision, F1, False Alarm을 비교한다.

## Alarm evaluation
row-level metric과 event-level metric을 둘 다 평가한다.

특히:
- 10분 기준 false alarm event
- 연속 positive 처리
- cooldown 또는 event merging

효과를 비교한다.

## Output rules
- 기존 baseline 결과를 덮어쓰지 않는다.
- 실험 결과는 output/experiments에 저장한다.
- metric 요약은 CSV로 저장한다.
- 모델 파일은 output/models에 저장한다.
- 그래프는 output/plots에 저장한다.
- random_state를 고정한다.
- 실험 조건과 threshold를 결과와 함께 저장한다.

## Coding rules
- 기존 코드를 먼저 읽고 재사용한다.
- 불필요한 파일 전체 재작성 금지.
- 기존 파일 삭제 금지.
- 큰 구조 변경 전 현재 구조를 확인한다.
- 문제가 없으면 불필요한 질문 없이 진행한다.
- 완료 후에는 아래만 짧게 보고한다:
  1. 변경 파일
  2. 핵심 결과
  3. baseline 대비 차이
  4. 다음 실험 1개