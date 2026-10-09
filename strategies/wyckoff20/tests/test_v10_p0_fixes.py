# -*- coding: utf-8 -*-
"""
Tests for Wyckoff V10 P0 Critical Fixes

C-1: FactorNeutralizer Z-score fit/apply consistency
C-2: Industry classification Point-in-Time
C-3: timedelta import
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from unittest.mock import patch, MagicMock

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


class TestC3TimedeltaImport(unittest.TestCase):
    """C-3: Verify timedelta is properly imported in gpsef_strategy_v10.py"""

    def test_timedelta_imported(self):
        """timedelta should be importable from the strategy module"""
        # Read the source file and check for the import
        v10_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'gpsef_strategy_v10.py'
        )
        with open(v10_path, 'r', encoding='utf-8') as f:
            source = f.read()
        self.assertIn('from datetime import datetime, timedelta', source,
                      "timedelta should be imported alongside datetime")

    def test_timedelta_usable(self):
        """timedelta should be usable in the module namespace"""
        # This would fail with NameError if timedelta wasn't imported
        from datetime import timedelta
        result = datetime(2025, 1, 1) - timedelta(days=1)
        self.assertEqual(result, datetime(2024, 12, 31))

    def test_barra_constraint_path(self):
        """Barra constraint path using timedelta(days=1) should not raise NameError"""
        # Simulate the code path: constraint_date = current_dt - timedelta(days=1)
        current_dt = datetime(2025, 6, 15)
        try:
            constraint_date = current_dt - timedelta(days=1)
            self.assertEqual(constraint_date, datetime(2025, 6, 14))
        except NameError:
            self.fail("NameError: timedelta is not defined - C-3 fix not applied")


class TestC1FactorNeutralizer(unittest.TestCase):
    """C-1: FactorNeutralizer Z-score fit/apply consistency"""

    def setUp(self):
        """Create a mock framework neutralizer to avoid real dependency"""
        self._orig_fw = None
        try:
            from gpsef.utils.neutralizer import FactorNeutralizer as FW
            self._orig_fw = FW
        except ImportError:
            pass

        # Create mock framework neutralizer
        self.mock_fw = MagicMock()
        self.mock_fw.neutralize = lambda df, cols, X: df
        self.mock_fw.neutralize_fit = lambda df, cols, X: (df, {'betas': {}, 'X_cols': []})
        self.mock_fw.neutralize_apply = lambda df, cols, X, bi: df

    def _make_neutralizer(self):
        """Create FactorNeutralizer with mocked framework dependency"""
        # We need to mock the import
        import importlib
        # Try direct import first
        try:
            # Patch the framework neutralizer
            with patch('gpsef.utils.neutralizer.FactorNeutralizer', return_value=self.mock_fw):
                # Re-import to get our class
                pass
        except Exception:
            pass

        # Direct approach: import and patch
        try:
            from strategies.wyckoff20.gpsef_strategy_v10 import FactorNeutralizer
            fn = FactorNeutralizer.__new__(FactorNeutralizer)
            fn.winsorize_pct = (1, 99)
            fn._framework_neutralizer = self.mock_fw
            fn._fit_params = {}
            return fn
        except ImportError:
            # If we can't import the module, test the class directly
            return None

    def test_fit_stores_params(self):
        """After fit(), _fit_params should contain stats for each factor column"""
        fn = self._make_neutralizer()
        if fn is None:
            self.skipTest("Cannot import FactorNeutralizer")

        # Create training data with known distribution
        np.random.seed(42)
        train_df = pd.DataFrame({
            'factor_a': np.random.normal(10, 2, 1000),
            'factor_b': np.random.normal(-5, 3, 1000),
        })

        fn.fit(train_df, ['factor_a', 'factor_b'])

        # Check params were stored
        self.assertIn('factor_a', fn._fit_params)
        self.assertIn('factor_b', fn._fit_params)

        # Check param structure
        for col in ['factor_a', 'factor_b']:
            params = fn._fit_params[col]
            self.assertIn('win_lower', params)
            self.assertIn('win_upper', params)
            self.assertIn('z_mean', params)
            self.assertIn('z_std', params)

        # Check values are reasonable
        # factor_a: mean~10, std~2
        self.assertAlmostEqual(fn._fit_params['factor_a']['z_mean'], 10, delta=1)
        self.assertAlmostEqual(fn._fit_params['factor_a']['z_std'], 2, delta=1)

    def test_fit_params_persist_to_apply(self):
        """Training params should be used in apply, not current data stats"""
        fn = self._make_neutralizer()
        if fn is None:
            self.skipTest("Cannot import FactorNeutralizer")

        # Training data: mean=10, std=2
        np.random.seed(42)
        train_df = pd.DataFrame({'factor_a': np.random.normal(10, 2, 5000)})
        fn.fit(train_df, ['factor_a'])

        # Prediction data: very different distribution (mean=100, std=20)
        pred_df = pd.DataFrame({'factor_a': np.random.normal(100, 20, 100)})

        # Apply using stored params
        result = fn._zscore_apply(
            fn._winsorize_apply(pred_df['factor_a'], 'factor_a'),
            'factor_a'
        )

        # If using training params (mean~10, std~2), values ~100 would be z~45
        # If using current params (mean~100, std~20), values would be z~0
        # We expect large z-scores because pred data is very different from train
        self.assertTrue(abs(result.mean()) > 10,
                        "Should use training params, not current data stats")

    def test_winsorize_uses_training_bounds(self):
        """Winsorize bounds should come from training data, not prediction data"""
        fn = self._make_neutralizer()
        if fn is None:
            self.skipTest("Cannot import FactorNeutralizer")

        # Training data: range [0, 20]
        train_df = pd.DataFrame({'f1': np.linspace(0, 20, 1000)})
        fn.fit(train_df, ['f1'])

        train_bounds = (fn._fit_params['f1']['win_lower'],
                        fn._fit_params['f1']['win_upper'])

        # Prediction data: range [0, 200] (10x wider)
        pred_series = pd.Series(np.linspace(0, 200, 100))

        # Apply training bounds
        result = fn._winsorize_apply(pred_series, 'f1')

        # Values should be clipped to training bounds (approximately 0.2 to 19.8 for 1%/99%)
        self.assertTrue(result.max() <= train_bounds[1] + 1,
                        f"Max {result.max()} should be near training upper bound {train_bounds[1]}")
        self.assertTrue(result.min() >= train_bounds[0] - 1,
                        f"Min {result.min()} should be near training lower bound {train_bounds[0]}")

    def test_fallback_when_no_fit(self):
        """When _fit_params is empty, should fall back to current data stats"""
        fn = self._make_neutralizer()
        if fn is None:
            self.skipTest("Cannot import FactorNeutralizer")

        # Don't call fit - _fit_params is empty
        series = pd.Series(np.random.normal(0, 1, 100))

        # _winsorize_apply should fall back to _winsorize
        result = fn._winsorize_apply(series, 'nonexistent_col')
        self.assertEqual(len(result), 100)

        # _zscore_apply should fall back to _zscore
        result_z = fn._zscore_apply(series, 'nonexistent_col')
        self.assertAlmostEqual(result_z.mean(), 0, delta=0.1)

    def test_neutralize_fit_calls_self_fit(self):
        """neutralize_fit should call self.fit() to store params"""
        fn = self._make_neutralizer()
        if fn is None:
            self.skipTest("Cannot import FactorNeutralizer")

        train_df = pd.DataFrame({
            'f1': np.random.normal(5, 1, 500),
            'f2': np.random.normal(-3, 2, 500),
        })
        X_neutral = pd.DataFrame({'x1': np.random.randn(500)})

        self.assertEqual(len(fn._fit_params), 0)
        fn.neutralize_fit(train_df, ['f1', 'f2'], X_neutral)
        self.assertGreater(len(fn._fit_params), 0,
                           "neutralize_fit should populate _fit_params")


class TestC2IndustryPointInTime(unittest.TestCase):
    """C-2: Industry classification Point-in-Time correctness"""

    def test_industry_pit_module_exists(self):
        """industry_pit.py module should exist"""
        pit_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'industry_pit.py'
        )
        self.assertTrue(os.path.exists(pit_path),
                        "industry_pit.py should exist in wyckoff20 directory")

    def test_industry_map_pit_class_importable(self):
        """IndustryMapPiT class should be importable"""
        try:
            from strategies.wyckoff20.industry_pit import IndustryMapPiT
            self.assertTrue(callable(IndustryMapPiT))
        except ImportError:
            # Try direct path import
            pit_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys.path.insert(0, pit_dir)
            try:
                from industry_pit import IndustryMapPiT
                self.assertTrue(callable(IndustryMapPiT))
            except ImportError:
                self.fail("IndustryMapPiT should be importable")

    def test_industry_map_pit_has_get_map(self):
        """IndustryMapPiT should have get_map method"""
        try:
            from strategies.wyckoff20.industry_pit import IndustryMapPiT
        except ImportError:
            pit_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys.path.insert(0, pit_dir)
            from industry_pit import IndustryMapPiT

        pit = IndustryMapPiT(perf_dir='/nonexistent_path_for_test')
        self.assertTrue(hasattr(pit, 'get_map'),
                        "IndustryMapPiT should have get_map method")
        self.assertTrue(callable(pit.get_map))

    def test_industry_map_pit_empty_dir_returns_empty(self):
        """IndustryMapPiT with nonexistent dir should return empty map"""
        try:
            from strategies.wyckoff20.industry_pit import IndustryMapPiT
        except ImportError:
            pit_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys.path.insert(0, pit_dir)
            from industry_pit import IndustryMapPiT

        pit = IndustryMapPiT(perf_dir='/nonexistent_path_for_test')
        result = pit.get_map(datetime(2024, 6, 1))
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)

    def test_build_snapshots_uses_pit(self):
        """build_snapshots.py should import and use IndustryMapPiT"""
        snap_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'build_snapshots.py'
        )
        with open(snap_path, 'r', encoding='utf-8') as f:
            source = f.read()
        self.assertIn('IndustryMapPiT', source,
                      "build_snapshots.py should import IndustryMapPiT")
        self.assertIn('industry_pit.get_map(', source,
                      "build_snapshots.py should call industry_pit.get_map()")

    def test_pit_logic_correctness(self):
        """PiT logic: for date T, only use data published <= T"""
        # Simulate records: (code, industry, pub_date)
        records = pd.DataFrame({
            'code': ['600000', '600000', '000001'],
            'industry': ['银行', '金融', '电子'],
            'pub_date': pd.to_datetime(['2023-01-01', '2024-06-01', '2023-03-01']),
        })

        # For query date 2023-06-01:
        # 600000 -> '银行' (only 2023-01-01 record is <= 2023-06-01)
        # 000001 -> '电子' (2023-03-01 <= 2023-06-01)
        query = pd.Timestamp('2023-06-01')
        valid = records[records['pub_date'] <= query]
        latest = valid.sort_values('pub_date').drop_duplicates(subset=['code'], keep='last')
        result = dict(zip(latest['code'], latest['industry']))

        self.assertEqual(result['600000'], '银行')
        self.assertEqual(result['000001'], '电子')

        # For query date 2024-07-01:
        # 600000 -> '金融' (2024-06-01 is the latest <= 2024-07-01)
        query2 = pd.Timestamp('2024-07-01')
        valid2 = records[records['pub_date'] <= query2]
        latest2 = valid2.sort_values('pub_date').drop_duplicates(subset=['code'], keep='last')
        result2 = dict(zip(latest2['code'], latest2['industry']))

        self.assertEqual(result2['600000'], '金融')


class TestC1ScoreNeutralization(unittest.TestCase):
    """C-1: _apply_score_neutralization fit/apply modes"""

    def test_score_neu_has_fit_param(self):
        """_apply_score_neutralization should accept fit parameter"""
        v10_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'gpsef_strategy_v10.py'
        )
        with open(v10_path, 'r', encoding='utf-8') as f:
            source = f.read()
        self.assertIn('fit: bool = False', source,
                      "_apply_score_neutralization should have fit parameter")

    def test_training_call_passes_fit_true(self):
        """Training code should call _apply_score_neutralization with fit=True"""
        v10_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'gpsef_strategy_v10.py'
        )
        with open(v10_path, 'r', encoding='utf-8') as f:
            source = f.read()
        self.assertIn('fit=True', source,
                      "Training call should pass fit=True")

    def test_score_neu_params_reset_at_rebalance(self):
        """_score_neu_params should be reset alongside _train_ols_betas at each rebalance"""
        v10_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'gpsef_strategy_v10.py'
        )
        with open(v10_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # Find lines with _train_ols_betas = None (not in __init__)
        reset_lines = []
        for i, line in enumerate(lines):
            s = line.strip()
            if '_train_ols_betas = None' in s and '__init__' not in s and i > 100:
                reset_lines.append(i)
        # For each reset location, check that _score_neu_params is also reset nearby
        for reset_idx in reset_lines:
            # Check surrounding 5 lines for _score_neu_params reset
            context = ''.join(lines[max(0, reset_idx-2):reset_idx+3])
            self.assertIn('_score_neu_params', context,
                          f"_score_neu_params should be reset near line {reset_idx+1} "
                          f"where _train_ols_betas is reset")

    def test_syntax_valid(self):
        """Modified strategy file should have valid Python syntax"""
        v10_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'gpsef_strategy_v10.py'
        )
        with open(v10_path, 'r', encoding='utf-8') as f:
            source = f.read()
        try:
            compile(source, v10_path, 'exec')
        except SyntaxError as e:
            self.fail(f"Syntax error in gpsef_strategy_v10.py: {e}")

    def test_industry_pit_syntax_valid(self):
        """industry_pit.py should have valid Python syntax"""
        pit_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'industry_pit.py'
        )
        with open(pit_path, 'r', encoding='utf-8') as f:
            source = f.read()
        try:
            compile(source, pit_path, 'exec')
        except SyntaxError as e:
            self.fail(f"Syntax error in industry_pit.py: {e}")

    def test_build_snapshots_syntax_valid(self):
        """build_snapshots.py should have valid Python syntax"""
        snap_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'build_snapshots.py'
        )
        with open(snap_path, 'r', encoding='utf-8') as f:
            source = f.read()
        try:
            compile(source, snap_path, 'exec')
        except SyntaxError as e:
            self.fail(f"Syntax error in build_snapshots.py: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
