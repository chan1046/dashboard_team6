# 다음 고장 유형 3-class 사건 단위 baseline

기존 Model 1, RUL 모델, 시간 분할은 수정하지 않았다. 현재 실험은 실제 고장 **30분 전**의 센서로 *그다음* 고장 유형을 예측하는 **조건부 개발 실험**이다. 고장 직후 유형 진단이나 임의의 현재 시점 예측 성능으로 해석하면 안 된다. 입력은 9개 센서의 실제 관측만이며 Tool ID·TTF·고장시각·유형은 feature에 넣지 않았다. 결과는 이 Validation에서 후보를 고른 개발 수치이며 독립 최종 Test 성능이 아니다.

## 표본과 파이프라인

고장별 예측 시각은 `fault−30분`, 입력 이력은 그 시각 직전 30분이다. 각 센서에서 마지막값·mean·std·min·max·median·실제 경과시간 기준 slope·마지막−첫값을 계산해 **72개 feature**를 만들었다. 기존 감사와 동일하게 실제 센서 **60행 이상**, 첫~마지막 관측 **24분 이상**, 마지막 관측이 예측 시각에서 **60초 이내**인 경우만 사용했다. 이전 모든 유형 고장과 기존 분할 경계를 넘어 이력을 연결하지 않았다. 보간·합성 행·SMOTE는 사용하지 않았다. 사용된 이력의 관측 행 수 중앙값은 Train **413행**, Validation **410행**, 최대 내부 gap 중앙값은 각각 **84초·92초**다. [표본과 품질](samples.csv), [사건별 탈락 사유/품질](eligibility.csv), [설정](config.json).

| 유형 | 기존 감사의 *고장 직전* 30분 이력 Train/Val | 기존 감사의 *고장 30분 전* 이력 Train/Val | 이번 엄격한 예측 창 Train/Val |
|---|---:|---:|---:|
| Low Pressure | 373 / 151 | 186 / 73 | **186 / 73** |
| High Pressure | 102 / 34 | 32 / 14 | **32 / 14** |
| Leak | 52 / 10 | 38 / 4 | **38 / 4** |
| 전체 | **527 / 195** | **256 / 91** | **256 / 91** |

요청에 적힌 527/195건은 **고장 직전**에 이력이 있는 사건 수다. 이번 모델은 **고장 30분 전**에 이미 30분 이력이 있어야 하므로 256/91건만 남는다. 숫자를 맞추기 위해 고장에 더 가까운 센서를 가져오지 않았다. Train 256건의 유형 구성은 Low **186**, High **32**, Leak **38**이고 Validation 91건은 Low **73**, High **14**, Leak **4**다. 동일 사건의 Train/Validation 중복과 예약·제외 사건 사용은 **0건**이다. [원 감사와 동일 사건 기준 대조](population_comparison.csv).

## 모델 비교

각 모델은 동일 256/91 사건과 동일 72개 feature를 사용했다. Tree/Random Forest는 기본·class_weight=`balanced`, XGBoost는 기본·Train 유형 빈도 역수 가중을 비교했다. 검증 결과로 hyperparameter를 반복 조정하지 않았다. [전체 지표](model_comparison.csv), [유형별 Precision·Recall·F1](class_metrics.csv), [모든 혼동 행렬](confusion_matrices/).

| 모델 | Macro F1 | Balanced Accuracy | Accuracy | Low Recall | High Recall | Leak Recall |
|---|---:|---:|---:|---:|---:|---:|
| Dummy: 항상 Low | 0.297 | 0.333 | 0.802 | 1.000 | 0 | 0 |
| Decision Tree | 0.439 | 0.421 | 0.736 | 0.836 | 0.429 | 0 |
| Decision Tree balanced | 0.475 | 0.546 | 0.703 | 0.781 | 0.357 | 0.500 |
| Random Forest | **0.603** | **0.611** | **0.813** | **0.904** | 0.429 | 0.500 |
| Random Forest balanced | 0.497 | 0.590 | 0.670 | 0.699 | 0.571 | 0.500 |
| XGBoost | 0.446 | 0.431 | 0.758 | 0.863 | 0.429 | 0 |
| XGBoost balanced | 0.398 | 0.399 | 0.681 | 0.767 | 0.429 | 0 |

Validation 표본이 바뀌어 Dummy Accuracy도 요청의 예상 77.4%가 아니라 **80.2%**다. 기본 Random Forest의 유형별 Precision / Recall / F1은 Low **0.868 / 0.904 / 0.886**, High **0.667 / 0.429 / 0.522**, Leak **0.333 / 0.500 / 0.400**이다. Leak Recall 0.5는 단 **2/4건**에서 나온다. 가장 큰 오류는 실제 **High → Low 8건**이고, 실제 Leak → Low **2건**이다. Low → High **3건**, Low → Leak **4건**이다. 가중 Random Forest는 High Recall을 0.429→0.571로 높였지만 Low Recall과 Macro F1이 크게 낮아졌다. XGBoost 가중은 이번 조건에서 개선되지 않았다.

## 판단 보류와 Tool 편중

class score는 **보정된 확률이 아니다**. 기본 Random Forest의 최대 score별 유형 표시는 [전체 threshold 표](confidence_thresholds.csv)에 저장했다.

| 최대 score 기준 | 표시/전체 | coverage | 표시된 사건 정답률 | 오진 수 | Leak 표시/정답 |
|---|---:|---:|---:|---:|---:|
| 0.5 | 79/91 | 86.8% | 83.5% | 13 | 3/0 |
| 0.6 | 66/91 | 72.5% | 86.4% | 9 | 2/0 |
| 0.7 | 54/91 | 59.3% | 94.4% | 3 | 0/0 |
| 0.8 | 48/91 | 52.7% | 95.8% | 2 | 0/0 |

높은 score 기준의 높은 정답률은 **주로 Low만 표시한 결과**다. `0.7`이나 `0.8`을 운영 threshold로 고른 것이 아니며, Leak을 올바르게 표시할 수 있다는 증거도 아니다.

Validation의 Low 73건 중 `03_M02`가 **31건**, High 14건 중 `09_M02`가 **5건**, Leak 4건은 `04_M02`와 `06_M02`에 각 **2건**이다. Tool ID를 feature로 쓰지 않았어도 센서 절대값이 Tool의 상태를 암시할 수 있다. 기본 Random Forest에서 `03_M02` 제외 Macro F1은 **0.585**, `09_M02` 제외 **0.661**, `06_M02` 제외 **0.665**다. 이는 동일 모델의 Validation 부분집합을 다시 점수화한 **민감도**이며 각 부분집합의 유형 구성이 바뀌므로 모델의 일반화 성능 향상으로 해석할 수 없다. [Tool별 결과](tool_metrics.csv), [Tool 제외 민감도](tool_exclusion_sensitivity.csv).

기본 Random Forest의 불순도 기반 상위 feature는 Etch Gas Channel 1 시간 기울기, Etch Suppressor 전압·전류 기울기, FlowCool Pressure 마지막값·최댓값 등이다. [상위 20개](feature_importance.csv). 이 순위는 센서가 고장의 원인이라는 증거가 아니고, Tool별 기준 차이와 상관된 분할일 수도 있다.

## 누수·후보 판단

자동 검사에서 `sensor timestamp > prediction timestamp`, `prediction >= fault`, Train/Validation 사건 중복, TTF·고장 유형·Tool ID feature, 예약·제외 사용 위반은 모두 **0건**이다. [검사표](leakage_check.csv). 실제 TTF·fault timestamp는 사건 선택과 평가 metadata에만 사용했다.

이번 개발 Validation의 **Model 2 후보는 기본 Random Forest**다. Dummy 대비 Macro F1 **+0.306**, Balanced Accuracy **+0.278**로 명확한 *이 표본 안의* 개선이 있다. 그러나 High 14건 중 8건을 Low로 오진했고 Leak은 4건뿐이다. 더 중요하게, 이 모델은 실제 고장을 알고 **고장 30분 전 시점**을 골라 만든 표본에서만 검증됐다. Dashboard의 Model 1 경보는 고장 30분 전으로 제한되지 않으므로 현재 수치를 그 전체 경보에 적용할 수 없다. 지금은 단정적인 `다음 고장 유형: …` 표현을 정당화하기 어렵다. `참고 고장 유형: …`과 낮은 score의 `유형 판단 보류`는 **향후 Model 1 실제 경보 시점에서 별도 연결 검증을 통과한 뒤** 검토할 수 있다. 현재 score threshold도 보정·고정된 운영 기준이 아니다. [선정 모델 metadata](best_model.json), [저장 모델](../../models/full_fault_type_best.joblib).
