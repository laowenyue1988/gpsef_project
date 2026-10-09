# -*- coding: utf-8 -*-
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def _simulate_check(preds, n_select=None, threshold=0.003):
    if threshold <= 0: return False
    if n_select and n_select < len(preds):
        top_n = np.sort(preds)[-n_select:]
        return float(top_n[-1] - top_n[0]) < threshold
    return float(np.std(preds)) < threshold

def test_normal_no(): 
    p = np.random.RandomState(42).uniform(0.45,0.55,500)
    assert not _simulate_check(p,n_select=50)
    print("[PASS] normal -> no collapse")

def test_narrow(): 
    np.random.seed(1)
    p = 0.5 + np.random.randn(300) * 0.0005
    assert _simulate_check(p,threshold=0.003)
    print("[PASS] narrow -> collapse")

def test_identical(): 
    assert _simulate_check(np.full(300,0.52),threshold=0.003)
    print("[PASS] identical -> collapse")

def test_top300_zero_range():
    """核心: global std > 0.003 但 top 300 range = 0"""
    np.random.seed(20210903)
    n_all = 2760; n_sel = 300
    preds = 0.50 + np.random.randn(n_all) * 0.007
    preds = np.sort(preds)
    preds[-n_sel:] = preds[-n_sel//2]  # top 300 all same
    tr = float(preds[-1] - preds[-n_sel])
    gs = float(np.std(preds))
    assert gs>0.003, f"global std {gs:.4f}"
    assert tr==0.0, f"top range {tr:.8f}"
    assert _simulate_check(preds, n_select=n_sel, threshold=0.003)
    print(f"[PASS] top-300 range=0.0 (global std={gs:.4f}) -> collapse")

def test_off(): assert not _simulate_check(np.full(300,0.5),threshold=0)
def test_config(): 
    import yaml; cfg=yaml.safe_load(open(os.path.dirname(__file__)+'/../config_v10.yaml','r',encoding='utf-8'))
    assert'prob_collapse_std_threshold'in cfg; print("[PASS] config OK")

tests=[test_normal_no,test_narrow,test_identical,test_top300_zero_range,test_off,test_config]
p=f=0
for t in tests:
    try:t();p+=1
    except Exception as e:f+=1;print(f"[FAIL] {e}")
print(f"\n{p} passed, {f} failed, {len(tests)} total")
sys.exit(1 if f else 0)
