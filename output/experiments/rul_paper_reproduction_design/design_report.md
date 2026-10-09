# PHM 2018 FlowCool RUL 논문 재현용 데이터 설계 (학습 없음)

## 범위와 확인 가능한 근거

본 보고서는 실제 20개 Train 설비의 원본 4초 센서 파일과 중복 제거한 고장 기록 1,211건을 읽어 사건별 이력량만 집계했다. 새 모델 학습, 기존 Model 1 변경, 최종 Test 사용은 없다. 사건별 원자료는 `fault_event_history_audit.csv`와 `paper_type_specific_cycles.csv`, 수치 정의는 각각 `availability_report.json`과 `paper_specific_availability.json`에 있다.

공개 본문: [TV 등, PHM 2018](https://papers.phmsociety.org/index.php/phmconf/article/download/589/phmc_18_589), [Huang 등, PHM 2018](https://papers.phmsociety.org/index.php/phmconf/article/download/590/phmc_18_590). [Wu 등, Control Engineering Practice 2021](https://www.sciencedirect.com/science/article/abs/pii/S0967066121000253)은 공개 초록·하이라이트까지만 확인했다. 2021년 논문의 세부 수치·절차는 공개 접근이 불가능한 항목을 추측하지 않는다.

## 논문과 현행 실험 비교

| 항목 | 현재 `sequence_rul_baseline` | TV 등 2018 | Wu 등 2021 | Huang 등 2018 |
|---|---|---|---|---|
| Target | 다음 모든 유형 중 가장 가까운 고장까지 실제 시계시간, 0~6/12/24h 조건부 | 세 유형별 RUL 벡터, 상한 `ru=500`으로 clip | FlowCool RUL; 세 고장 모드별 결과 제시 | 실제 시계시간 TTF를 남은 관측치 수로 변환 |
| 사건/사이클 | 다음 어떤 고장 1건에 묶음 | 20개 설비, 세 유형 동시 출력, 같은 설비에 반복 고장 존재 | 상세 cycle 정의는 공개 초록에서 미확인 | 각 유형별 run-to-fault sequence, 세 개의 독립 모델 |
| 입력 | 원본 숫자 센서 5개 + stage, 1분 행; 85 engineered feature 자체는 sequence에 미사용 | 원본 숫자 9개와 shutter=1 screening; stage, Lot, runnum 제외 | sample screening, MLP 추가 feature 및 GRU/GRU-FC; 정확한 입력 목록 미확인 | 센서·가동시간 변수·shutter one-hot, 총 21차원; recipe/runnum 제외 |
| Sequence | 30/60/120개의 1분 관측, 10분 간격 anchor | 원본 관측치 2,000개 창, 창 간 겹침 500개; d1=2 다운샘플, d2=10 T2V | 시계열 길이 ablation; 정확한 값 미확인 | 원본 약 4초 관측 4,515개까지 역추적(약 5h), SampleR=15, maxL=300, N=15; 최소 3,000개 이력 |
| 공백·정지 | 한 간격 >90초 또는 session 변경 시 30분 시퀀스 전체 제외 | >90초 연속성 요구는 기술되지 않음; 셔터 작동 상태 screening | down-time 데이터 제거를 명시; 공백 허용 수치 미확인 | 큰 공백에서 run-to-fault 이력을 자른 뒤 충분한 이력만 사용; 큰 공백의 초 단위 기준 미명시 |
| 열화 시작 | 실제 TTF로 0~6h 등을 선택; online onset 아님 | 시작 시각이 불명확하다고 지적하고 먼 RUL을 clip | RF 기반 early-degradation warning 뒤 GRU/GRU-FC | RF로 정상/열화 감지; 정상 시 숫자 RUL을 표시하지 않음 |
| 정상 구간 | 조건부 학습에서 배제 | 먼 RUL clip 후 일부 정상 창을 표본 추출 | early warning을 이용해 전체 run-to-failure에 적용한다고 설명 | 정상은 RUL 회귀 단계에서 제외; 감지 단계 학습에 포함 |
| 유형 처리 | 모든 유형을 한 scalar next-fault target으로 혼합 | 한 RNN에서 3개 RUL head | 유형별 성능 제시, 구조 세부는 초록만으로 단정 불가 | 3개의 유형별 독립 RUL 모델 |
| Scaling·loss | Train median/IQR; smooth-L1 | 제공 센서가 이미 z-normalized라고 관찰; RUL/ru; MSE | 상세 공개 불가 | 제공 센서는 이미 normalized; RUL은 관측치 수 target; MSE |
| Split·평가 | 설비별 시간순 Train/Validation, 사건 분리; MAE/RMSE/R² | Train 20대 내 labeled holdout 및 시간상 뒤의 대회 Test/Validation; 대회 S1/S2, RMSE 등 | 초록에서 SMAPE 3모드 결과, split 세부 미확인 | 유형별 run-to-fault sequence 80/20 랜덤 train/test와 train 내부 validation; RMSE/SMAPE |
| 동일 사건 다중 창 | 10분 anchor마다 생성 | 2,000점 창을 겹치게 생성 | 상세 공개 불가 | 끝 지점을 한 샘플씩 옮겨 N=15개 단축 시퀀스 생성 |
| Censoring·근접 고장 | 알려진 다음 고장 기준; 미관측 미래는 모델 target 없음 | TTF NaN 존재, RUL clip/표본화; 근접고장 병합 규칙 명시 없음 | 공개 초록에 명시 없음 | 유형별 run-to-fault 분할; 근접고장 병합·검열 규칙 명시 없음 |

논문 1의 원본 데이터는 센서뿐 아니라 공정 상태가 섞인 24개 열이며, 저자들이 실제 사용한 9개 숫자 변수는 ION gauge, beam voltage/current, suppressor voltage/current, FlowCool flow/pressure, gas readback 두 개다. Shutter=1인 관측을 남긴다. 현재 실험은 85개 가공 feature를 시퀀스 입력으로 사용하지 않았다. 코드의 실제 입력은 원본 센서 5개와 stage 하나다. 따라서 “85개 때문에 입력 차원이 커졌다”는 현재 실패 원인으로 말할 수 없다.

## 실제 데이터 가용성 감사

| 정의 | 후보 고장 사건 | 의미 |
|---|---:|---|
| 중복 제거한 전체 fault 기록 | 1,211 | 저압 741, 고압 333, 누출 137 |
| 직전 **같은 유형** 고장 이후 센서가 1개 이상 | 1,085 | 긴 이력 보장은 아님 |
| 유형별 이력 전체에 shutter=1 원본 관측 ≥2,000 | 783 | 논문 1 창 구성의 넓은 **상한**; 저압 509·고압 197·누출 77 |
| 그중 고장 전 마지막 5h에 shutter=1 관측 ≥2,000 | 152 | 선택적인 짧은 열화창 설계, 논문 1의 필수 조건은 아님 |
| 유형별 고장 전 5h 원본 관측 ≥3,000 | 242 | 논문 3의 최소 길이 기준에 가까운 상한 |
| 마지막 30분 초과 공백 뒤의 5h tail ≥3,000 | 189 | 논문 3 공백 절단의 **자체 민감도 가정**; 저압 122·고압 34·누출 33 |
| 위 조건 + 마지막 센서가 고장 1h 이내 | 185 | 운영상 최신성 추가 조건; 논문 조건은 아님 |

이 수치는 **실제 학습 가능 사건 수가 아니라 잠재 상한**이다. 개별 입력 센서 결측, 정확한 작동상태 screening, 동일 사건의 중복 창, Train/Validation 경계, 검열, 가까운 반복고장 정리가 남아 있다. 특히 논문 1은 한 유형이 아닌 3개 출력에 같은 시퀀스를 사용할 수 있어 각 유형 후보 수를 더해 독립 표본 수로 간주하면 안 된다. 반대로 직전 모든 유형의 고장 이후만 허용하면 논문식 유형별 이력을 지나치게 줄인다. 동일 도구의 다른 유형 고장 이후 관측은 과거 정보로만 쓸 수 있으며, episode 경계 여부를 별도로 기록해야 한다.

Model 1 경보 332,303행 중 정확히 252,608행은 과거 30개 1분 행을 모으는 사이 session 변경 또는 90초 초과 공백이 있었다. 최초 30행 부족·센서 결측·시간 정렬 불일치는 각각 0건이다. 나머지 79,695행(24.0%)만 현재 규칙을 통과했다. 즉 낮은 coverage의 직접 원인은 **공백/세션 규칙**이며, 원본 4초 센서 부재를 뜻하지 않는다. 공백을 무시하고 센서 행을 붙이는 것은 금지하고, 경과시간·정지·재시작 정보를 명시한 불규칙 시퀀스로 따로 설계해야 한다.

## 다음 재현 실험용 확정 데이터 명세 (이번에는 생성·학습하지 않음)

1. **원본 및 ID:** 20개 실제 Train 원본 4초 parquet와 중복 제거된 fault log를 사용한다. `event_id=(Tool,fault_type,fault_time)`를 보존한다. 단일 scalar next-any-fault TTF 대신 저압·고압·누출별 `next_same_type_fault_time`과 각 target의 관측 가능 여부를 별도로 기록한다. 고장 시각 및 TTF는 target/검열 판정 전용이다.
2. **경계:** 설비별 시간순 기존 forward Train/Validation/Test 계획을 고정한다. 각 경계에서 ≥24h purge하고, 입력 창의 시작·끝·target fault 모두 해당 구간에 두며, 같은 event는 한 구간에만 속하게 한다. 예약 Test는 구성 수·성능 선택에 쓰지 않는다. 논문 3의 랜덤 80/20은 재현 참고사항으로만 남기고 성능 비교에는 사용하지 않는다.
3. **입력:** 먼저 논문 1에 가장 가까운 원본 숫자 9개(`IONGAUGEPRESSURE`, `ETCHBEAMVOLTAGE`, `ETCHBEAMCURRENT`, `ETCHSUPPRESSORVOLTAGE`, `ETCHSUPPRESSORCURRENT`, `FLOWCOOLFLOWRATE`, `FLOWCOOLPRESSURE`, `ETCHGASCHANNEL1READBACK`, `ETCHPBNGASREADBACK`)와 `FIXTURESHUTTERPOSITION==1`을 사용한다. `Tool`, 사건 ID, 절대시간, Lot, runnum, TTF, 다음 고장 시각은 feature에서 제외한다. 센서 결측률과 Train-only scaling 값을 먼저 저장한다.
4. **창:** 각 설비의 작동 관측치를 시간순으로 유지한다. 현재 시점까지의 최근 원본 관측 2,000개를 기본 창으로 하고, 학습 anchor는 1,500개 관측 간격(500개 겹침)으로 만든다. 2,000개 미만은 짧은 창 별도 그룹으로 기록하며 기본 비교에서는 제외한다. 경과시간과 공백 길이를 metadata로 저장한다. 90초 gap에서 일괄 폐기하지 않는다. 30분 초과 정지에서는 창을 분리한 분석도 함께 만들되, 정지 양쪽을 붙이는 대안에는 명시적 gap mask/시간간격을 넣고 실제 downtime을 센서값으로 채우지 않는다. 각 창의 target은 **마지막 관측 시각** 기준으로 만든다.
5. **Target/검열:** 세 유형 각각 `next_same_type_fault_time - anchor_time`을 실제 초 단위로 만든다. 미래의 같은 유형 고장이 관측되지 않으면 해당 head를 right-censored/masked로 표시한다. 연구용 RUL 상한은 Train만 보고 하나의 사전 명세 값으로 정하고, 논문의 `ru=500`을 단위 검증 없이 시간 초로 복사하지 않는다. 검증에서 실제 TTF로 입력 창을 고르지 않는다.
6. **열화 시작:** 열화 시작은 기록된 shutdown 시각과 다르며 정답으로 주어지지 않는다. Model 1의 기존 24h 위험 경보는 **online 평가 gate 후보**로만 기록하고, 열화 시작의 참값이나 학습 feature로 넣지 않는다. Train 구간에서만 정상/열화 onset 규칙을 정한 뒤 Validation에 고정 적용한다. onset 전에는 정확한 RUL 대신 미표시/상한 상태를 정의한다.
7. **반복 고장:** 같은 유형의 30분 이내 연속 fault와 그 사이 센서 유무를 별도 episode map으로 기록한다. 재가동·수리 증거 없이 자동으로 독립 고장으로 세거나 임의로 합치지 않는다. 서로 다른 유형 사건은 다중 target 관계를 유지하며 다른 유형의 중간 사건 여부를 metadata로 남긴다.
8. **평가 단위:** 동일 사건의 여러 창은 같은 split에 둔다. 행 MAE/RMSE뿐 아니라 사건별 MAE를 동일가중하고, 유형별 사건 수·설비별 오차·추론 가능 coverage·Model 1 경보에서의 조건부 오차를 보고한다. Dummy와 동일 창·동일 split에서 비교한다.

이 명세의 1차 재현 방향은 **논문 1의 3-output 공유 인코더**다. 이는 저압/고압/누출을 한 개의 `next-any-fault` 숫자로 합치는 현재 방식과 다르다. 논문 3에 가까운 **유형별 별도 모델**은 같은 split·동일 관측 집합의 사전 명세된 비교안으로 남긴다. 유형별 후보가 189건 상한까지 줄 수 있어, 작은 누출·고압 그룹을 독립적으로 평가할 때 불확실성을 크게 표시해야 한다.

## 결론과 한계

현재 GRU의 성능 부진은 **실험 구성 차이와 데이터의 실제 한계가 함께 작용할 가능성**이 있다. 90초 공백 규칙, scalar next-any-fault target, 실제 TTF로 제한한 0~6h 조건부 학습, 짧은 30분 이력은 논문식 설계와 다르다. 동시에 논문식 약 5시간의 충분한 관측 이력을 요구하면 활용 가능한 사건이 189건 이하의 상한으로 줄고 고장 전 장시간 정지·불명확한 열화 시작이 남는다. 데이터셋 재구성 후 동일 표본 Dummy와 비교하기 전에는 어느 쪽이 주원인이라고 단정할 수 없다.
