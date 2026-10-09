# -*- coding: utf-8 -*-
"""
V10 测试运行器：执行全部8个测试模块，汇总结果。
"""
import sys
import os
import time
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

TEST_MODULES = [
    ('test_v10_purged_kfold', 'Purged K-Fold CV (P0-1)'),
    ('test_v10_feature_selection', 'Feature Selection (P1-1+P1-2+P3-4)'),
    ('test_v10_ensemble', '5-Seed Ensemble (P0-3)'),
    ('test_v10_calibration', 'Isotonic Calibration (P0-4+P3-2)'),
    ('test_v10_auc_gate', 'AUC Gate (P0-2)'),
    ('test_v10_neutralization', 'Neutralization (P1-4+V5)'),
    ('test_v10_merge', 'V5 Merge Completeness'),
    ('test_v10_e2e', 'End-to-End Smoke Test'),
]


def run_all():
    total_pass = 0
    total_fail = 0
    total_skip = 0
    failed_tests = []
    module_results = []

    for module_name, description in TEST_MODULES:
        print(f"\n{'='*70}")
        print(f"  Running: {description}")
        print(f"{'='*70}")

        try:
            mod = __import__(f'strategies.wyckoff20.tests.{module_name}', fromlist=[module_name])
        except Exception as e:
            print(f"  [IMPORT ERROR] {e}")
            traceback.print_exc()
            total_fail += 1
            module_results.append((module_name, description, 'IMPORT_ERROR', str(e)))
            continue

        # 收集所有 test_ 开头的函数
        test_funcs = sorted([
            (name, getattr(mod, name))
            for name in dir(mod)
            if name.startswith('test_') and callable(getattr(mod, name))
        ])

        for test_name, test_func in test_funcs:
            t0 = time.time()
            try:
                test_func()
                elapsed = time.time() - t0
                total_pass += 1
                module_results.append((module_name, test_name, 'PASS', f'{elapsed:.2f}s'))
            except Exception as e:
                elapsed = time.time() - t0
                total_fail += 1
                failed_tests.append(f"{module_name}.{test_name}: {e}")
                print(f"  [FAIL] {test_name}: {e}")
                traceback.print_exc()
                module_results.append((module_name, test_name, 'FAIL', str(e)[:200]))

    # 汇总
    print(f"\n{'='*70}")
    print(f"  TEST SUMMARY")
    print(f"{'='*70}")
    print(f"  Total: {total_pass + total_fail}  |  PASS: {total_pass}  |  FAIL: {total_fail}")
    print()

    for module_name, test_name, status, info in module_results:
        icon = '✅' if status == 'PASS' else '❌'
        print(f"  {icon} {module_name}.{test_name} — {info}")

    if failed_tests:
        print(f"\n{'='*70}")
        print(f"  FAILED TESTS ({len(failed_tests)})")
        print(f"{'='*70}")
        for ft in failed_tests:
            print(f"  ❌ {ft}")

    return total_fail == 0


if __name__ == '__main__':
    success = run_all()
    sys.exit(0 if success else 1)
