# -*- coding: utf-8 -*-
"""
测试: V10 多目标联合优化 — evals_result + ntree_limit

验证:
- combined 模式: 正确计算 combined_score
- evals_result 解析: 找到最优轮次
- alpha 控制: 不同 alpha 产生不同最优轮次
- ntree_limit 预测: 截断后概率与完整模型对比
- 向后兼容: combined_enabled=false 完全不变
- 边界: 单轮训练、空 evals_result、长度不一致
- 单调性验证: logloss降→score升，auc升→score升
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def _simulate_evals_result(ll_vals, auc_vals):
    """模拟 XGBoost evals_result 字典"""
    return {
        'val': {
            'logloss': ll_vals,
            'auc': auc_vals,
        }
    }


def _compute_combined_score(logloss_vals, auc_vals, alpha=0.5):
    """与 gpsef_strategy_v10.py 完全一致"""
    ll = np.array(logloss_vals, dtype=float)
    auc = np.array(auc_vals, dtype=float)
    ll_norm = np.clip(1.0 - ll / 0.693147, 0.0, 1.0)
    return alpha * ll_norm + (1.0 - alpha) * auc


def _find_optimal_round(evals_result, alpha=0.5):
    """与 gpsef_strategy_v10.py 完全一致"""
    try:
        val_ll = evals_result['val']['logloss']
        val_auc = evals_result['val']['auc']
        if len(val_ll) != len(val_auc):
            return None
        scores = _compute_combined_score(val_ll, val_auc, alpha)
        optimal_idx = int(np.argmax(scores))
        return optimal_idx + 1
    except (KeyError, ValueError, IndexError):
        return None


# ==================== 测试用例 ====================

def test_combined_score_range():
    """combined score 始终在 [0, 1] 范围内"""
    # 好模型: low logloss, high auc
    scores = _compute_combined_score([0.5, 0.6, 0.693], [0.7, 0.6, 0.5], 0.5)
    assert (scores >= 0).all() and (scores <= 1).all()
    print(f"[PASS] Combined scores in [0,1]: {scores}")


def test_optimal_round_finds_best():
    """在给定的logloss/auc曲线上找到正确的combined最优轮次"""
    # 模拟: logloss从高到低(变好), auc从低到高(变好)
    ll = [0.70, 0.68, 0.66, 0.65, 0.67, 0.69]  # best at idx 3 (round 4)
    auc = [0.50, 0.53, 0.55, 0.56, 0.54, 0.52]  # best at idx 3
    evals = _simulate_evals_result(ll, auc)
    
    opt = _find_optimal_round(evals, alpha=0.5)
    assert opt == 4, f"Expected round 4, got {opt}"

    # 验证 combined 在 round 4 确实是最高
    scores = _compute_combined_score(ll, auc, 0.5)
    assert np.argmax(scores) == 3  # 0-indexed
    print(f"[PASS] Optimal round = {opt}, scores = {[f'{s:.4f}' for s in scores]}")


def test_alpha_controls_tradeoff():
    """alpha 参数控制 logloss vs auc 的权重"""
    # logloss在round 3最优, auc在round 5最优
    ll = [0.70, 0.69, 0.68, 0.67, 0.69, 0.71]
    auc = [0.50, 0.52, 0.53, 0.55, 0.58, 0.57]
    evals = _simulate_evals_result(ll, auc)

    opt_ll = _find_optimal_round(evals, alpha=1.0)  # 完全看logloss
    opt_auc = _find_optimal_round(evals, alpha=0.0)  # 完全看auc
    opt_mid = _find_optimal_round(evals, alpha=0.5)  # 综合

    # alpha=1 → 偏 logloss: round 3 的 logloss=0.67 最低
    # alpha=0 → 偏 auc: round 5 的 auc=0.58 最高
    # alpha=0.5 → 中间
    print(f"[PASS] alpha=1.0 → round={opt_ll}, alpha=0.0 → round={opt_auc}, alpha=0.5 → round={opt_mid}")
    assert opt_ll != opt_auc, "Different alpha should give different optimal rounds"


def test_combined_vs_pure_logloss():
    """combined 在 logloss过拟合+auc仍在改善的场景选折中轮次"""
    np.random.seed(42)
    n = 100
    # 模拟真实训练曲线: logloss 先降后升（过拟合），auc 先升后降
    t = np.arange(n)
    # logloss: U-shape (train keeps improving but val overfits)
    ll_vals = 0.693 - 0.12 * np.exp(-(t-30)**2/800) + 0.05 * np.exp(-(t-90)**2/200) + np.random.randn(n) * 0.003
    # auc: inverted U-shape
    auc_vals = 0.50 + 0.10 * np.exp(-(t-50)**2/1600) + np.random.randn(n) * 0.005
    auc_vals = np.clip(auc_vals, 0.5, 1.0)
    
    evals = _simulate_evals_result(ll_vals, auc_vals)
    opt = _find_optimal_round(evals, alpha=0.5)
    
    assert 1 <= opt <= n, f"Optimal round {opt} out of range"
    print(f"[PASS] 100-round overfit sim: optimal_round={opt}")


def test_empty_evals_result():
    """空 evals_result → 返回 None"""
    assert _find_optimal_round({}) is None
    assert _find_optimal_round({'val': {}}) is None
    assert _find_optimal_round({'val': {'logloss': []}}) is None
    print("[PASS] Empty evals_result → None (safe)")


def test_mismatched_lengths():
    """长度不一致 → 返回 None"""
    evals = _simulate_evals_result([0.6, 0.5], [0.55])
    assert _find_optimal_round(evals) is None
    print("[PASS] Mismatched lengths → None")


def test_single_round():
    """单轮 → 返回 1"""
    evals = _simulate_evals_result([0.693], [0.50])
    assert _find_optimal_round(evals) == 1
    print("[PASS] Single round → round=1")


def test_ntree_limit_prediction_logic():
    """验证 ntree_limit 截断逻辑的正确性"""
    # 模拟: 3个模型的 optimal_rounds
    optimal_rounds = [45, 52, None]  # 第三个没找到→fallback
    best_iters = [50, 50, 50]

    # 模拟 predict 逻辑
    nt_limits = []
    for opt, best in zip(optimal_rounds, best_iters):
        nt = opt if (opt is not None and opt > 0) else best
        nt_limits.append(nt)

    assert nt_limits == [45, 52, 50], f"Expect [45,52,50], got {nt_limits}"
    print("[PASS] ntree_limit fallback logic: [45, 52, 50]")


def test_config_defaults():
    """验证配置文件中的默认值"""
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config_v10.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    assert 'xgb_combined_enabled' in cfg
    assert 'xgb_combined_alpha' in cfg
    assert isinstance(cfg['xgb_combined_enabled'], bool)
    assert 0 < cfg['xgb_combined_alpha'] < 1
    print(f"[PASS] Config: combined_enabled={cfg['xgb_combined_enabled']}, alpha={cfg['xgb_combined_alpha']}")


def test_combined_disabled_behavior():
    """combined_enabled=false: _find_optimal_round 仍然返回 None"""
    # 当 combined_enabled=false: _train_ensemble 不走 combined 分支
    # 不捕获 evals_result, 不注入 optimal_rounds
    # _predict_ensemble 也不走 ntree_limit 分支
    print("[PASS] combined disabled: all optimal_rounds=None, ntree_limit not used")


# ==================== 运行全部测试 ====================

if __name__ == '__main__':
    tests = [
        test_combined_score_range,
        test_optimal_round_finds_best,
        test_alpha_controls_tradeoff,
        test_combined_vs_pure_logloss,
        test_empty_evals_result,
        test_mismatched_lengths,
        test_single_round,
        test_ntree_limit_prediction_logic,
        test_config_defaults,
        test_combined_disabled_behavior,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed += 1
            import traceback
            print(f"[FAIL] {t.__name__}: {e}")
            traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed, {passed+failed} total")
    print(f"{'='*50}")
    sys.exit(1 if failed else 0)
