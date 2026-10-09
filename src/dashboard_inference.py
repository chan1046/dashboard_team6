"""Read-only inference from frozen training features and real sensor history.

The dashboard replays each tool's last validation observation, not live data.
Model 1 consumes the exact cache written by recover_xgb24_baseline.prepare_tool;
no approximate rolling/ewma reconstruction or experiment writes occur here.
"""
from functools import lru_cache
from pathlib import Path
import json
import sys

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@lru_cache(maxsize=1)
def _dependencies():
    # Keep the UI usable even when native scientific libraries cannot load.
    global joblib, np, pd, pq, _features, summarize, MODEL2_FEATURES, SENSORS, DISPLAY
    import joblib
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from recover_xgb24_baseline import _features
    from full_fault_type_multi_anchor import summarize, FEATURES as MODEL2_FEATURES, SENSORS
    from train_full_fault_type_baseline import DISPLAY


ROOT = SRC.parent
PREPARED = ROOT / 'output/baseline_recovery/prepared'
RAW = ROOT / 'data/processed'
EXPERIMENTS = ROOT / 'output/experiments'
TOOLS = [f'{i:02d}_M{j:02d}' for i in range(1, 11) for j in (1, 2)]


class InferenceUnavailable(ValueError):
    pass


@lru_cache(maxsize=1)
def load_model1():
    _dependencies()
    return joblib.load(ROOT / 'output/models/xgb_24h_forward_a.joblib')


@lru_cache(maxsize=1)
def load_model2():
    _dependencies()
    return joblib.load(ROOT / 'output/models/full_fault_type_multi_anchor_best.joblib')


@lru_cache(maxsize=1)
def replay_plan():
    data = json.loads((EXPERIMENTS / 'forward_validation_ab_24h/split_plan.json').read_text(encoding='utf-8'))
    return {row['tool_id']: row for row in data['tools']}


@lru_cache(maxsize=1)
def expected_observations():
    _dependencies()
    return pd.read_csv(EXPERIMENTS / 'model2_partial_history_audit/tool_sampling_intervals.csv').set_index('tool_id').expected_rows_in_30m.to_dict()


@lru_cache(maxsize=1)
def fault_events():
    _dependencies()
    return pd.read_csv(RAW / 'fault_events.csv').drop_duplicates(['tool_id', 'time', 'fault_type'])


def validate_features(frame, payload, expected, count):
    _dependencies()
    saved = list(payload['features'])
    if len(saved) != count or saved != list(expected) or list(frame.columns) != saved:
        raise InferenceUnavailable(f'Feature 이름/개수/순서 불일치 (필요: {count}개)')
    if payload['model'].n_features_in_ != count:
        raise InferenceUnavailable('저장 모델의 입력 차원 불일치')
    return frame.to_numpy(dtype=np.float32, copy=True)


def build_model1_features(tool, timestamp):
    _dependencies()
    if tool not in TOOLS:
        raise InferenceUnavailable('알 수 없는 설비')
    path = PREPARED / f'{tool}.parquet'
    signature = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))['signature']
    source = RAW / f'{tool}_sensor_ttf.parquet'
    source_stat = source.stat()
    # ZIP archives preserve timestamps at two-second precision. Keep the size
    # guard while tolerating only that known metadata precision loss.
    source_matches_cache = (
        signature['source_bytes'] == source_stat.st_size
        and abs(signature['source_mtime_ns'] - source_stat.st_mtime_ns)
        <= 2_000_000_000
    )
    if not source_matches_cache:
        raise InferenceUnavailable('원시 데이터가 변경되어 feature 캐시 재검증 필요')
    # _features is the original training feature selector, including its order.
    names = _features(pq.ParquetFile(path).schema_arrow.names)
    frame = pd.read_parquet(path, columns=names, filters=[('time', '==', int(timestamp))])
    if len(frame) != 1:
        raise InferenceUnavailable('해당 기준시각의 학습 방식 feature 없음')
    validate_features(frame, load_model1(), names, 85)
    # XGBoost was trained with NaN; preserve that native missing-value handling.
    if np.isinf(frame.to_numpy()).any() or not np.isfinite(frame[SENSORS].to_numpy()).any():
        raise InferenceUnavailable('유효한 Model 1 입력 부족')
    return frame


def status_from_score(score):
    return '정상' if score < .34 else ('주의' if score < .69 else '위험')


def predict_model1(tool, timestamp):
    try:
        payload = load_model1()
        frame = build_model1_features(tool, timestamp)
        values = validate_features(frame, payload, _features(pq.ParquetFile(PREPARED / f'{tool}.parquet').schema_arrow.names), 85)
        index = list(payload['model'].classes_).index(1)
        score = float(payload['model'].predict_proba(values)[0, index])
        if not np.isfinite(score):
            raise InferenceUnavailable('Model 1 score가 유효하지 않음')
        return {'available': True, 'risk_score': score, 'status': status_from_score(score)}
    except Exception as exc:
        return {'available': False, 'risk_score': None, 'status': '판단 보류', 'error': f'{type(exc).__name__}: {exc}'}


def build_model2_features(tool, timestamp, lower_bound=None):
    _dependencies()
    if tool not in TOOLS:
        raise InferenceUnavailable('알 수 없는 설비')
    timestamp = int(timestamp)
    # Only already-known faults bound the cycle; never inspect a future fault.
    previous = fault_events().loc[(fault_events().tool_id == tool) & (fault_events().time <= timestamp), 'time']
    lower = max(timestamp - 1800, int(previous.max()) + 1 if len(previous) else timestamp - 1800)
    if lower_bound is not None:
        lower = max(lower, int(lower_bound))
    frame = pd.read_parquet(RAW / f'{tool}_sensor_ttf.parquet', columns=['time', *SENSORS],
                            filters=[('time', '>=', lower), ('time', '<=', timestamp)])
    t = frame.time.to_numpy(dtype=np.int64)
    x = frame[SENSORS].to_numpy(dtype=np.float64)
    if len(t) < 2 or np.any(np.diff(t) < 0):
        raise InferenceUnavailable('30분 센서 관측 부족 또는 시간 순서 오류')
    expected = float(expected_observations()[tool])
    ratio, span, age = len(t) / expected, int(t[-1] - t[0]), int(timestamp - t[-1])
    # Exact Candidate A eligibility from full_fault_type_multi_anchor.prepare.
    if not (ratio >= .5 and span >= 900 and 0 <= age <= 300 and np.isfinite(x).all()):
        raise InferenceUnavailable(f'Candidate A 미충족: 관측비율 {ratio:.3f}, 범위 {span}초, 관측지연 {age}초')
    features = summarize(t, x)  # Original 9 sensors x 8 statistics, no filling.
    if features is None or not np.isfinite(list(features.values())).all():
        raise InferenceUnavailable('Model 2 feature 생성 불가')
    result = pd.DataFrame([features])
    validate_features(result, load_model2(), MODEL2_FEATURES, 72)
    return result


def predict_model2(tool, timestamp, lower_bound=None):
    try:
        payload = load_model2()
        frame = build_model2_features(tool, timestamp, lower_bound)
        values = validate_features(frame, payload, MODEL2_FEATURES, 72)
        scores = payload['model'].predict_proba(values)[0]
        classes = list(payload['model'].classes_)
        if set(classes) != set(DISPLAY) or not np.isfinite(scores).all():
            raise InferenceUnavailable('Model 2 클래스 또는 score 불일치')
        selected = classes[int(np.argmax(scores))]
        return {'available': True, 'predicted_class': DISPLAY[selected], 'class_key': selected,
                'max_score': float(scores.max()), 'class_scores': dict(zip(classes, map(float, scores)))}
    except Exception as exc:
        return {'available': False, 'predicted_class': '판단 보류', 'error': f'{type(exc).__name__}: {exc}'}


def rank_results(results):
    eligible = sorted((tool for tool, result in results.items() if result['model1']['available']),
                      key=lambda tool: (-results[tool]['model1']['risk_score'], tool))
    ranks = {tool: i + 1 for i, tool in enumerate(eligible)}
    for tool, result in results.items():
        result['priority'] = ranks.get(tool)
    return results


def inspect_tools(tools, hours=72):
    results = {}
    try:
        _dependencies()
    except Exception as exc:
        error = f'추론 실행 환경 로드 실패: {type(exc).__name__}: {exc}'
        return rank_results({tool: {'timestamp': None,
            'model1': {'available': False, 'status': '판단 보류', 'risk_score': None, 'error': error},
            'model2': {'available': False, 'predicted_class': '판단 보류', 'error': error},
            'fault_count': None, 'seconds_since_fault': None} for tool in tools})
    for tool in tools:
        try:
            plan = replay_plan()[tool]
            timestamp = int(plan['validation_last_time'])
            m1 = predict_model1(tool, timestamp)
            m2 = predict_model2(tool, timestamp, plan['validation_start_second'])
            faults = fault_events().loc[(fault_events().tool_id == tool) & (fault_events().time <= timestamp)]
            in_period = faults if hours == 999 else faults[faults.time >= timestamp - hours * 3600]
            results[tool] = {'timestamp': timestamp, 'time_basis': '원본 데이터 경과 초',
                             'source': '검증 구간 마지막 관측 재생', 'model1': m1, 'model2': m2,
                             'fault_count': len(in_period),
                             'seconds_since_fault': int(timestamp - faults.time.max()) if len(faults) else None}
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            results[tool] = {'timestamp': None, 'model1': {'available': False, 'status': '판단 보류', 'risk_score': None, 'error': error},
                             'model2': {'available': False, 'predicted_class': '판단 보류', 'error': error},
                             'fault_count': None, 'seconds_since_fault': None}
    return rank_results(results)
