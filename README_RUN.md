# 냉각 설비 고장 예측 대시보드 (FlowCool, 6조) 실행 방법

이온 밀링 장비 FlowCool의 센서 데이터로 24시간 내 고장 위험과 고장 유형을 보여 주는 Dash 대시보드입니다.
이 저장소에는 코드와 작은 결과 리포트만 있고, 용량이 큰 데이터와 학습된 모델 파일은 포함하지 않았습니다.

## 준비
1. Python 3.11 이상에서 가상환경을 만들고 `pip install -r requirements.txt`
2. `.env.example`을 `.env`로 복사하고 두 키를 직접 생성해 채웁니다.
3. 관리자 계정 만들기: `python scripts/create_admin.py`
4. 데이터 배치: PHM 2018 Ion Mill 데이터를 전처리한 `data/processed/` (센서+TTF parquet, `fault_events.csv`)
5. 모델 파일 배치: `output/models/` 아래에 학습된 joblib 파일 (`.gitignore`로 제외됨, 학습 스크립트는 `src/`)

## 실행
```
python app.py
```
브라우저에서 `http://localhost:8063` 접속 후 관리자 계정으로 로그인합니다.

## 구성
- `app.py`: Dash 화면 (모니터링 설정, 설비 현황, 우선 점검, 상세 진단, 조치 기록)
- `src/dashboard_inference.py`, `src/dashboard_replay.py`: 모델 추론과 저장된 결과 재생
- `src/auth`, `src/db`, `src/security`: 로그인, 조치 기록 DB, 개인정보 암호화
- `src/` 나머지: 전처리, 학습, 검증 실험 스크립트
- `tests/`: 추론과 화면 연결 테스트

## 주의
- 성능은 제공된 과거 데이터에 대한 오프라인 평가이며 실제 운영 성능을 보장하지 않습니다.
- 데이터셋은 PHM Society 2018 Data Challenge 공개 데이터입니다. 재배포 조건을 확인하고 원본을 저장소에 올리지 마세요.
