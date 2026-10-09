# 전체 자료의 생존분석 가능성 점검

모델 학습 없이 20대의 실제 센서·TTF 정렬 데이터 82,189,440행과 전체 fault log를 조사했다. 원본 fault 기록 1,236행에서 같은 Tool·fault type·발생시각의 중복 25행을 제거하여 고유 사건 1,211건을 얻었다. 유형별로 저압 741, 고압 333, 누출 137건이다. 네 사건은 해당 Tool의 첫 센서 관측보다 앞에 있고, 관측이 끝난 뒤의 사건은 없다.

| 항목 | 건수 |
|---|---:|
| 고유 fault event | 1,211 |
| 센서 관측 범위 안의 fault event | 1,207 |
| 정의 가능한 관측상 inter-fault 구간 | 1,227 |
| 실제 fault로 끝난 구간 | 1,207 |
| 데이터 끝에서 오른쪽 censoring된 구간 | 20 |
| fault 종료 구간 중 이전 센서 이력 존재 | 1,047 |
| fault 종료 구간 중 이전 센서 이력 없음 | 160 |
| fault 1시간 이내에 센서 관측 존재 | 717 |

각 Tool의 첫 센서 시각→첫 관측범위 내 fault, 이어서 fault 시각→다음 fault 시각, 마지막 fault→마지막 센서 시각으로 구간을 정했다. 첫 구간은 장비의 이전 가동연령을 모르므로 실제 수리 후 시작한 생존 cycle이라고 할 수 없다. 이후 구간도 고장 후 수리·재가동 여부가 확인되지 않아 **관측상 inter-fault 간격**이다. 센서 feature로 사용할 수 있는 이력은 이전 fault보다 뒤, 다음 fault보다 앞의 관측치뿐이다. 다음 fault 시각·TTF·fault type은 target 또는 검열 표지에만 사용한다.

반복 고장은 20대 모두에서 나타난다(설비별 고유 사건 2~356건; 첫 사건 이후 반복 기록 총 1,191건). 인접 고장 1,191쌍 중 3쌍은 1분, 27쌍은 10분, 118쌍은 30분, 381쌍은 6시간 이내다. 30분 이내 118쌍 중 69쌍에는 사이에 센서 기록이 없다. 이를 무조건 별도의 수리 완료 cycle로 취급하면 같은 미해결 고장 episode를 여러 번 셀 위험이 있다. 30분 이내를 하나로 묶으면 기록상 cluster는 1,093개이나, **병합 기준은 운영 기록으로 검증해야 한다.** 30분을 확정값으로 사용하지 않는다.

실제 event cycle 길이 중앙값은 20.37시간, 사분위 3.97~79.05시간이며 최단 6초, 최장 약 7,399시간이다. censored cycle은 20개뿐이라 검열 비율이 매우 낮다. 고장으로 끝난 구간 중 709개에는 1시간 넘는 센서 공백이 있고, 330개는 마지막 센서가 고장보다 1시간 이상 앞선다(그중 58개는 24시간 이상). 공백은 설비 정지일 수 있으므로 값을 앞뒤로 채우거나 계속 가동했다고 간주해서는 안 된다. 달력시간과 실제 가동시간 중 어떤 시간축이 운영 목적에 맞는지 별도로 정의해야 한다.

### 방법 검토

* **기본 Cox proportional hazards:** cycle당 고정 feature 하나로 만들면 시간에 따라 바뀌는 센서 정보가 사라진다. Counting-process 형태의 time-varying Cox라면 가능하지만, 비례위험 가정과 반복 사건의 설비 내 의존성·다중 유형을 따로 다뤄야 한다. [lifelines의 시간가변 공변량 설명](https://lifelines.readthedocs.io/en/latest/Time%20varying%20survival%20regression.html).
* **일반 Random Survival Forest:** cycle 시작의 고정 센서 요약으로는 가능하나, 시간가변 센서를 그대로 투입하는 표준 RSF와는 자료 구조가 맞지 않는다. 모델 입력은 개체별 고정 X와 event/time 형태다. [scikit-survival RSF 문서](https://scikit-survival.readthedocs.io/en/latest/api/generated/sksurv.ensemble.RandomSurvivalForest.html).
* **시간가변 discrete-time survival:** 가동 중 관측 시점마다 현재·과거 센서로 위험집합 행을 만들고, 다음 구간에서 사건 없음/저압/고압/누출의 원인별 hazard를 추정하는 구조가 적합하다. 반복 구간은 Tool·episode ID로 묶고, 마지막 관측 끝에서 censoring을 표시할 수 있다. 정지 상태는 별도 상태·노출 규칙으로 정해야 한다. 이 선택은 현재 자료 구조에 대한 설계 판단이며 성능이 검증된 모델이라는 뜻이 아니다. [반복 사건과 시간가변 공변량의 discrete-time 모델 연구](https://pmc.ncbi.nlm.nih.gov/articles/PMC8546916/).

모델이 검증된다면 대시보드에 향후 12/24시간 유형별 위험 확률이나 생존곡선을 제공할 **가능성**은 있다. 정확한 고장 시각이나 잔여시간 중앙값을 제공할 수 있는지는 아직 알 수 없다. sensor history가 없는 160개 사건, 매우 가까운 사건, 긴 정지 공백, 적은 censored cycle, Tool별 사건 쏠림을 먼저 정리하고 시간순 후속 검증에서 calibration·사건 탐지·오경보를 확인해야 한다. 따라서 두 번째 모델의 **연구 후보로 진행 가능**하지만 운영 모델 채택이나 대시보드 연결은 아직 이르다.

상세 데이터: `failure_cycles.csv`, `tool_summary.csv`, `unique_fault_events.csv`, `close_fault_pairs_le_24h.csv`, `faults_outside_sensor_range.csv`, `cycle_duration_distribution.csv`, `report.json`.
