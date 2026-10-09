"""Frozen real-validation regression; never trains or writes artifacts."""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import dashboard_inference as di


class DashboardInferenceRegression(unittest.TestCase):
    def test_model1_real_validation(self):
        saved = np.load(di.EXPERIMENTS / 'forward_validation_ab_24h/validation_probabilities_a.npz')
        for code in (0, 1, 2):
            indices = np.flatnonzero(saved['tool_code'] == code)
            i = indices[len(indices) // 2]
            tool, timestamp = di.TOOLS[code], int(saved['time'][i])
            frame = di.build_model1_features(tool, timestamp)
            self.assertEqual(list(frame.columns), di.load_model1()['features'])
            self.assertEqual(frame.shape, (1, 85))
            result = di.predict_model1(tool, timestamp)
            self.assertTrue(result['available'], result)
            delta = abs(result['risk_score'] - float(saved['probability'][i]))
            self.assertLessEqual(delta, 1e-7)
            print(f'Model 1 {tool} t={timestamp}: 85 features; score delta={delta:.3g}', flush=True)

    def test_model2_real_validation(self):
        folder = di.EXPERIMENTS / 'full_fault_type_multi_anchor'
        anchors = pd.read_csv(folder / 'anchor_samples.csv')
        saved = pd.read_csv(folder / 'validation_predictions.csv')
        candidates = anchors[(anchors.phase == 'validation') & anchors.eligible_partial_A]
        for label in ('pressure_low', 'pressure_high', 'leak'):
            row = candidates[candidates.fault_type == label].iloc[0]
            timestamp = int(row.prediction_timestamp)
            frame = di.build_model2_features(row.tool_id, timestamp, row.phase_and_cycle_start)
            self.assertEqual(frame.shape, (1, 72))
            self.assertEqual(list(frame.columns), di.load_model2()['features'])
            np.testing.assert_allclose(frame.iloc[0].to_numpy(), row[frame.columns].to_numpy(dtype=float), rtol=1e-8, atol=1e-8)
            result = di.predict_model2(row.tool_id, timestamp, row.phase_and_cycle_start)
            self.assertTrue(result['available'], result)
            reference = saved[(saved.model == 'partial_A_event_class') & (saved.fault_event_id == row.fault_event_id) & (saved.horizon == row.horizon)].iloc[0]
            actual = [result['class_scores'][c] for c in di.load_model2()['model'].classes_]
            expected = [reference[f'score_{c}'] for c in di.load_model2()['model'].classes_]
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
            self.assertEqual(result['class_key'], reference.predicted)
            print(f'Model 2 {row.tool_id} t={timestamp}: 72 features; class={result["class_key"]}; score delta={np.max(np.abs(np.array(actual)-expected)):.3g}', flush=True)

    def test_status_boundaries(self):
        self.assertEqual([di.status_from_score(x) for x in (.339999, .34, .689999, .69)], ['정상', '주의', '주의', '위험'])

    def test_schema_and_insufficient_history(self):
        anchors = pd.read_csv(di.EXPERIMENTS / 'full_fault_type_multi_anchor/anchor_samples.csv')
        row = anchors[(anchors.phase == 'validation') & anchors.eligible_partial_A].iloc[0]
        frame = di.build_model2_features(row.tool_id, row.prediction_timestamp, row.phase_and_cycle_start)
        with self.assertRaises(di.InferenceUnavailable):
            di.validate_features(frame[frame.columns[::-1]], di.load_model2(), di.MODEL2_FEATURES, 72)
        # Real observation, intentionally restricted to one timestamp: must defer.
        result = di.predict_model2(row.tool_id, row.prediction_timestamp, row.prediction_timestamp)
        self.assertFalse(result['available'])
        self.assertIs(di.load_model1(), di.load_model1())
        self.assertIs(di.load_model2(), di.load_model2())


if __name__ == '__main__':
    unittest.main(verbosity=2)
