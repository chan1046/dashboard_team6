"""Dash wiring checks using saved real Model 1 scores and explicit missing data."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app
from src import dashboard_inference as di


def initial_state():
    return {'page': 0, 'scope': 'all', 'hours': 72, 'unit': 0, 'done': False, 'sensor': 0}


def component_ids(component):
    if hasattr(component, 'id'):
        yield json.dumps(component.id, sort_keys=True)
    children = getattr(component, 'children', None)
    for child in children if isinstance(children, (list, tuple)) else [children]:
        if child is not None:
            yield from component_ids(child)


class DashboardAppChecks(unittest.TestCase):
    def setUp(self):
        self.client = app.server.test_client()
        saved = np.load(di.EXPERIMENTS / 'forward_validation_ab_24h/validation_probabilities_a.npz')
        self.results = {}
        for code in (0, 1, 2):
            i = np.flatnonzero(saved['tool_code'] == code)[0]
            score = float(saved['probability'][i])
            self.results[di.TOOLS[code]] = {'timestamp': int(saved['time'][i]),
                'model1': {'available': True, 'risk_score': score, 'status': di.status_from_score(score)},
                'model2': {'available': False, 'predicted_class': '판단 보류', 'error': 'UI 검사: 동시각 평가 표본 없음'},
                'fault_count': None, 'seconds_since_fault': None}
        di.rank_results(self.results)

    def test_http_layout_and_unique_ids(self):
        for path in ('/', '/_dash-layout', '/_dash-dependencies'):
            self.assertEqual(self.client.get(path).status_code, 200)
        ids = list(component_ids(app.app.layout))
        self.assertEqual(len(ids), len(set(ids)))
        overview, priority = app.result_table(self.results, 'overview'), app.result_table(self.results, 'priority')
        ids = list(component_ids(overview)) + list(component_ids(priority))
        self.assertEqual(len(ids), len(set(ids)))

    def test_render_and_rank_real_scores(self):
        state = initial_state()
        state.update(page=1, done=True)
        self.assertEqual(len(app.render_state(state, self.results)), 32)
        self.assertEqual(len(app.render_predictions(self.results, state)), 12)
        ranked = sorted(self.results.values(), key=lambda r: r['priority'])
        scores = [r['model1']['risk_score'] for r in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))
        # An actual missing-data error must never acquire a numeric rank.
        with patch.object(di, '_dependencies', side_effect=ImportError('dependency unavailable')):
            deferred = di.inspect_tools(['10_M02'])
        self.assertIsNone(deferred['10_M02']['priority'])
        self.assertFalse(deferred['10_M02']['model1']['available'])
        self.results.update(deferred)
        di.rank_results(self.results)
        self.assertIsNone(self.results['10_M02']['priority'])

    def test_dash_render_callback_real_scores(self):
        key = next(key for key in app.app.callback_map if 'overview-kpis.children' in key)
        outputs = app.app.callback_map[key]['output']
        response = self.client.post('/_dash-update-component', json={
            'output': key, 'outputs': [{'id': o.component_id, 'property': o.component_property} for o in outputs],
            'inputs': [{'id': 'prediction-results', 'property': 'data', 'value': self.results},
                       {'id': 'app-state', 'property': 'data', 'value': initial_state()}],
            'state': [], 'changedPropIds': ['prediction-results.data']})
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertIn('overview-results', response.get_json()['response'])

    def test_navigation_and_single_tool(self):
        entry = app.app.callback_map['app-state.data']
        base = [{'id': x['id'], 'property': x['property'], 'value': [] if x['id'].startswith('{') else None} for x in entry['inputs']]
        base[0]['value'] = {'01_M02': self.results['01_M02']}
        # The result itself is irrelevant to navigation; selection uses actual tool option 1.
        payload = {'output': 'app-state.data', 'outputs': {'id': 'app-state', 'property': 'data'},
                   'inputs': base, 'state': [
                       {'id': 'app-state', 'property': 'data', 'value': initial_state()},
                       {'id': 'scope', 'property': 'value', 'value': '1'},
                       {'id': 'period', 'property': 'value', 'value': 24}],
                   'changedPropIds': ['prediction-results.data']}
        response = self.client.post('/_dash-update-component', json=payload)
        self.assertEqual(response.status_code, 200)
        state = response.get_json()['response']['app-state']['data']
        self.assertEqual((state['unit'], state['page'], state['scope']), (1, 3, '1'))
        # Actual row button pattern updates to the clicked tool.
        payload['state'][0]['value'] = dict(state, scope='all')
        pattern = next(x for x in base if x['id'].startswith('{'))
        pattern['value'] = [1]
        trigger = json.dumps({'type': 'detail-btn', 'tool': '02_M01', 'view': 'priority'}, separators=(',', ':'), sort_keys=True)
        payload['changedPropIds'] = [trigger + '.n_clicks']
        response = self.client.post('/_dash-update-component', json=payload)
        self.assertEqual(response.status_code, 200)
        state = response.get_json()['response']['app-state']['data']
        self.assertEqual((state['unit'], state['page']), (2, 3))

    def test_report_save_keeps_local_structure(self):
        state = initial_state()
        saved, error, issue, action = app.save_report(
            1, [], state, '점검자', '2026-09-28', '확인 중', '압력 변동 확인', '현장 점검 요청')
        self.assertEqual(error, '')
        self.assertEqual((issue, action), ('', ''))
        self.assertEqual(saved[0]['unit'], '01_M01')
        self.assertEqual(saved[0]['status'], '확인 중')

    def test_saved_replay_fallback_and_fault_filters(self):
        from src.dashboard_replay import filtered_faults, replay_results
        replay = replay_results(di.TOOLS, 72)
        counts = {name: sum(x['model1']['status'] == name for x in replay.values())
                  for name in ('정상', '주의', '위험', '판단 보류')}
        self.assertEqual(counts, {'정상': 12, '주의': 5, '위험': 2, '판단 보류': 1})
        self.assertEqual([replay[t]['priority'] for t in ('03_M02', '03_M01', '08_M01')], [1, 2, 3])
        self.assertIsNone(replay['02_M01']['priority'])
        self.assertEqual([len(filtered_faults(h)) for h in (12, 24, 72, 999)], [0, 0, 4, 796])
        self.assertEqual(sum(x['model2']['available'] for x in replay.values()), 17)


if __name__ == '__main__':
    unittest.main(verbosity=2)
