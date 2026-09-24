# 02_M01 예지보전 EDA

## 센서+TTF 통합 데이터셋

원본의 중복 시각 행을 삭제하지 않고 sensor와 TTF의 행별 `time` 일치를 검사한 뒤, 설비별 압축 CSV 20개로 저장합니다. 전체 고장 로그는 센서 행에 반복 결합하지 않고 별도 사건표 하나로 통합합니다.

```powershell
& ".\.venv\Scripts\python.exe" src/build_sensor_ttf_dataset.py
```

결과는 `data/processed/sensor_ttf/*.csv.gz`, `data/processed/fault_events.csv`, `data/processed/sensor_ttf_manifest.csv`에 생성됩니다. VS Code에서는 **Run and Debug**의 `Build data - sensor+TTF 20 tools`를 실행할 수 있습니다.

`02_M01`의 각 고장 시각을 기준으로 직전 30분 센서 데이터를 추출하고, 마지막 5분의 변화를 비교합니다.

VS Code 터미널에서 다음 명령으로 실행합니다.

```powershell
& "C:\Users\USER\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" src/analyze_prefault_02_m01.py
```

기본 데이터 위치는 `C:\Users\USER\Desktop\LS_Project_01\data\train`입니다. 다른 위치를 사용할 경우 `--train-dir`로 지정할 수 있습니다.

```powershell
& "C:\Users\USER\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" src/analyze_prefault_02_m01.py --train-dir "D:\path\to\data\train"
```

결과는 `output/02_M01_prefault`에 생성됩니다.

## XGBoost 기본 모델

기본 모델은 TTF를 사용해 `향후 30분 안에 Pressure Drop이 발생하는가`를 예측합니다. `runnum`과 다음 고장 이벤트를 기준으로 학습·검증 데이터를 분리하여 인접한 센서 행이 양쪽에 섞이지 않게 합니다.

먼저 프로젝트 전용 Python 환경과 패키지를 준비합니다.

```powershell
& "C:\Users\USER\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements-xgboost.txt
```

`02_M01` 전처리와 학습을 실행합니다.

```powershell
& ".\.venv\Scripts\python.exe" src/train_xgboost_baseline.py
```

VS Code의 실행 버튼을 사용할 때는 **Run and Debug**에서 `XGBoost baseline - 02_M01` 구성을 선택합니다. 이 구성은 프로젝트의 `.venv`와 프로젝트 루트 작업 폴더를 명시적으로 사용합니다.

패키지 설치 전에 전처리만 확인하려면 다음 명령을 사용합니다.

```powershell
& "C:\Users\USER\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" src/train_xgboost_baseline.py --prepare-only
```

전체 장비로 확장할 때는 `--tools` 뒤에 파일 접두사를 나열합니다.

```powershell
& ".\.venv\Scripts\python.exe" src/train_xgboost_baseline.py --tools 01_M01 01_M02 02_M01 02_M02 03_M01 03_M02 04_M01 04_M02 05_M01 05_M02 06_M01 06_M02 07_M01 07_M02 08_M01 08_M02 09_M01 09_M02 10_M01 10_M02
```

다른 고장 유형은 `--fault pressure_high` 또는 `--fault leak`으로 선택할 수 있습니다.

## Random Forest 20개 설비 통합 파이프라인

`src/train_rf_multi_tool.py`는 20개 설비를 한 번에 메모리에 올리지 않습니다. 각 설비의 원본 sensor와 TTF를 chunk 단위로 대조하고 1분 특징 파일로 저장한 뒤, 저장된 특징을 사용해 통합 모델을 학습합니다.

학습 모델은 다음 네 개입니다.

- Pressure Low TTF Random Forest 회귀
- Pressure High TTF Random Forest 회귀
- Leak TTF Random Forest 회귀
- 다음 고장 유형 Random Forest 분류

동일 시각의 여러 센서 행은 삭제하지 않습니다. 모든 행을 1분 통계에 반영하고 `raw_rows_1m`, `unique_times_1m`, `duplicate_time_rows_1m`으로 데이터 밀도와 중복 정보를 보존합니다. 정상적인 4초 수집이라면 분당 15개 시각이 존재하므로 `coverage_ratio_1m=unique_times_1m/15`도 생성합니다. `fault_name`과 TTF 정답 열은 입력 특성에서 제외합니다.

장시간 무기록 구간은 데이터 수집 실패가 아니라 설비 정지로 해석합니다. 정지 시간을 보간하거나 앞뒤 값으로 채우지 않습니다. 기본적으로 120초보다 긴 무기록 구간 뒤의 첫 관측에는 `restart_after_stop=1`을 표시하고 새로운 `operating_session`으로 분리합니다. 이동평균·표준편차·변화량은 재가동 시점부터 다시 계산합니다. 모델 입력의 일부 센서값이 비어 있을 때만 학습 데이터의 중앙값으로 대체하며, 정지 시간의 행 자체는 새로 만들지 않습니다. 정지 구간 판정 기준은 `--stop-gap-seconds`로 변경할 수 있습니다.

대시보드는 무기록 구간을 `STOPPED`로 표시하고 그동안에는 고장 유형과 TTF를 새로 예측하지 않습니다. 센서가 다시 들어오면 `RUNNING`으로 전환합니다. 모델에는 `session_observation_index`와 `rolling_ready_5m/15m/30m`를 전달하여 재가동 후 센서 이력이 얼마나 쌓였는지 구분합니다. 화면에 별도의 `데이터 부족` 상태를 만들지 않습니다.

현재 운전 세션이 끝난 뒤에 발생하는 고장의 TTF는 해당 세션의 학습 정답에서 제외합니다. 예를 들어 운전 중단 후 정지 상태에서 기록된 고장을 정지 전 센서 패턴의 결과로 학습하지 않습니다. 제외 행 수는 `preparation_report.csv`의 `excluded_across_stop_*` 열에 기록합니다.

### 실행 순서

먼저 20개 설비의 sensor, TTF, faults 파일과 필수 열을 확인합니다.

```powershell
& ".\.venv\Scripts\python.exe" src/train_rf_multi_tool.py manifest
```

20개 원본을 설비별 1분 특징 파일로 변환합니다. 전체 과정에서 가장 오래 걸리는 단계이며 완료된 설비의 캐시는 다음 실행에서 다시 사용합니다.

```powershell
& ".\.venv\Scripts\python.exe" src/train_rf_multi_tool.py prepare
```

각 설비의 앞 80%를 학습, 뒤 20%를 검증으로 사용해 네 개 모델을 학습합니다. 분할 경계 앞 30분은 학습에서 제외합니다.

분할 이전 행의 TTF가 분할 이후의 고장 사건을 가리키면 그 TTF 정답도 학습에서 제외합니다. 따라서 동일한 미래 고장 사건이 학습 구간과 내부 평가 구간에 동시에 포함되지 않습니다. 제외 건수는 `metrics.json`의 `training_labels_excluded_across_cutoff`에 저장합니다.

기본값인 `--max-valid-rows 0`에서는 내부 평가 구간을 표본추출하지 않고 마지막 20% 전체로 지표를 계산합니다. 공식 `data/test` 파일은 정답이 없는 최종 시험 데이터이므로 학습과 성능지표 계산에는 사용하지 않고, 학습이 끝난 모델의 예측 결과만 생성합니다.

```powershell
& ".\.venv\Scripts\python.exe" src/train_rf_multi_tool.py train
```

학습된 모델로 `data/test`의 다섯 설비를 예측합니다.

```powershell
& ".\.venv\Scripts\python.exe" src/train_rf_multi_tool.py predict
```

한 번에 실행하려면 다음 명령을 사용합니다.

```powershell
& ".\.venv\Scripts\python.exe" src/train_rf_multi_tool.py all
```

VS Code에서는 **Run and Debug**에서 `RF multi-tool - 1. manifest`부터 `4. predict test`까지 순서대로 선택할 수 있습니다. `RF multi-tool - all`은 전체 단계를 연속 실행합니다.

기본 결과 경로는 `output/rf_multi_tool`입니다.

- `tool_manifest.csv`: 20개 설비 입력 파일 검증
- `prepared/*_features.csv.gz`: 설비별 1분 특징 캐시
- `preparation_report.csv`: 행 수, 시간축 정렬, 고장 사건 품질
- `models/*.joblib`: TTF 회귀 3개와 고장유형 분류 모델
- `regression_metrics.csv`: 전체 및 설비별 회귀 성능
- `classification_metrics.csv`: 전체 및 설비별 분류 성능
- `feature_importance/*.csv`: 모델별 변수 중요도
- `predictions/*_dashboard_prediction.csv.gz`: 테스트 설비 대시보드 예측

특정 설비만 시험하려면 `--tools`를 사용합니다.

```powershell
& ".\.venv\Scripts\python.exe" src/train_rf_multi_tool.py prepare --tools 01_M01 01_M02
```

`--max-raw-rows-per-tool`은 개발용 실행 제한입니다. 이 옵션으로 만든 특징 파일은 전체 학습에 사용하면 안 됩니다.
