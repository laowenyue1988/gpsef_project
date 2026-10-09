# -*- coding: utf-8 -*-
"""
SmallAlpha191Test2 12组格栅全自动排程器
=======================================
- 批0(R0,R1) 已手动启动(重跑), 本脚本等待其完成
- 之后按批推进: 每批最多 2 组并行, 全部完成后进入下一批
- 断点续跑: 状态文件 results/batch_all_status.txt 记录已完成组, 重启跳过
- 完成检测: 轮询日志出现"绩效报告已保存"(run_backtest.py 异步, 主进程提前返回)
用法: .venv/Scripts/python.exe scripts/auto_runner.py  (后台运行)
"""
import os
import re
import subprocess
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJ)
PY = os.path.join(PROJ, '.venv', 'Scripts', 'python.exe')
CFG_DIR = os.path.join(PROJ, 'strategies', 'small_alpha191_test2')
LOG_DIR = os.path.join(PROJ, 'results')
STATUS = os.path.join(LOG_DIR, 'batch_all_status.txt')

# 批次: (批次名, [(组名, config文件名)])
BATCHES = [
    ('B0', [('R0', 'config_R0.yaml'), ('R1', 'config_R1.yaml')]),   # 重跑(已启动, 等待)
    ('B1', [('R7', 'config_R7.yaml'), ('R8', 'config_R8.yaml')]),   # S1 σ 2.2 / 3.0
    ('B2', [('R9', 'config_R9.yaml'), ('R10', 'config_R10.yaml')]),  # AUC 门控 0.52 / 0.56
    ('B3', [('T1', 'config_T1.yaml'), ('T2', 'config_T2.yaml')]),    # 趋势过滤 ma 10 / 30
    ('B4', [('R11', 'config_R11.yaml'), ('R12', 'config_R12.yaml')]),  # 组合(激进/次优)
]
# [2026-08-30 修正] 裁剪 R2-R6: 本区间最大回撤仅-9.25%, 熔断阈值12%+不触发,
#   恢复参数/熔断MA无区分度(见批0分析: R1 8%触发但作用在选股空的调仓日);
#   新增 T1/T2 趋势过滤 ma 灵敏度(74%触发率→仓位25.6% 元凶; 原 R5/R6 误改熔断 index_ma)
BATCHES = {b[0]: b[1] for b in BATCHES}
HEARTBEAT = os.path.join(LOG_DIR, 'auto_runner_heartbeat.txt')


def log(msg):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}'
    try:
        print(line, flush=True)
    except Exception:
        pass  # stdout 管道关闭时容忍 (BrokenPipe 曾是排程器死亡根因)
    try:
        with open(os.path.join(LOG_DIR, 'auto_runner.log'), 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass


def load_status():
    done = set()
    if os.path.exists(STATUS):
        for line in open(STATUS, encoding='utf-8'):
            line = line.strip()
            if line and '=' not in line and '完成' not in line and 'BATCH' not in line:
                done.add(line)
    return done


def mark_done(g):
    with open(STATUS, 'a', encoding='utf-8') as f:
        f.write(g + '\n')


def wait_done(log_path, tag, timeout_h=14):
    """轮询日志直到出现完成标志 (run_backtest 异步, 不能等进程退出)。"""
    end = time.time() + timeout_h * 3600
    while time.time() < end:
        try:
            with open(HEARTBEAT, 'w', encoding='utf-8') as f:
                f.write(f'{tag} {time.strftime("%H:%M:%S")}\n')
        except Exception:
            pass
        if os.path.exists(log_path):
            try:
                txt = open(log_path, encoding='utf-8', errors='ignore').read()
                if '绩效报告已保存' in txt:
                    return True
            except Exception:
                pass
        time.sleep(120)
    return False


def run_group(g, cfg, log_name):
    """启动一组回测 (run_backtest 主进程提前返回, 回测子进程后台继续)。"""
    logp = os.path.join(LOG_DIR, log_name)
    cmd = [PY, '-u', 'run_backtest.py', '--config', os.path.join(CFG_DIR, cfg)]
    with open(logp, 'a', encoding='utf-8') as f:
        f.write(f'\n===== [{time.strftime("%H:%M:%S")}] {g} 启动 =====\n')
    subprocess.Popen(cmd, stdout=open(logp, 'a', encoding='utf-8', errors='ignore'),
                     stderr=subprocess.STDOUT)
    log(f'{g} 已启动 (config={cfg})')


def analyze_group(g):
    """组完成后自动分析: 解析结果目录 → 运行分析脚本 → 追加到 grid_summary.txt。"""
    logp = os.path.join(LOG_DIR, f'{g}.log')
    try:
        txt = open(logp, encoding='utf-8', errors='ignore').read()
        m = re.search(r'结果保存在: \./(results/[A-Za-z0-9_]+)', txt)
        if not m:
            m = re.search(r'绩效报告已保存: \./(results/[A-Za-z0-9_]+)', txt)
        if m:
            d = m.group(1)
            cmd = [PY, os.path.join('scripts', 'analyze_small_test2_result.py'),
                   '--dir', d, '--label', g]
            r = subprocess.run(cmd, capture_output=True, text=True,
                               encoding='utf-8', errors='ignore')
            with open(os.path.join(LOG_DIR, 'grid_summary.txt'), 'a', encoding='utf-8') as f:
                f.write(r.stdout + '\n')
            log(f'{g} 分析已写入 grid_summary.txt (dir={d})')
        else:
            log(f'{g} 未从日志解析到结果目录, 跳过自动分析')
    except Exception as e:
        log(f'{g} 自动分析失败: {e}')


def main():
    # [v4 鲁棒化] 整循环 try/except: 任何单组异常不退出, 记录后继续
    try:
        done = load_status()
        log('=' * 60)
        log(f'AUTO_RUNNER 启动. 已完成组: {sorted(done) or "无"}')
        # 批0: 等待 R0/R1 重跑完成 (日志已被重跑覆盖)
        for g in ['R0', 'R1']:
            if g in done:
                continue
            log(f'等待 {g} 完成 (批0 重跑)...')
            ok = wait_done(os.path.join(LOG_DIR, f'{g}.log'), g)
            if ok:
                mark_done(g)
                log(f'{g} 完成')
                analyze_group(g)
            else:
                log(f'{g} 超时未完成! 继续后续 (将视为未完成)')
        # 后续批次
        order = ['B1', 'B2', 'B3', 'B4']
        for bname in order:
            try:
                groups = BATCHES[bname]
                pending = [(g, cfg) for g, cfg in groups if g not in done]
                if not pending:
                    log(f'{bname} 全部已完成, 跳过')
                    continue
                log(f'{bname} 启动: {[g for g, _ in pending]}')
                for g, cfg in pending:
                    run_group(g, cfg, f'{g}.log')
                    time.sleep(3)   # [v4] 错开启动秒, 防输出目录同名互覆
                for g, cfg in pending:
                    try:
                        ok = wait_done(os.path.join(LOG_DIR, f'{g}.log'), g)
                        mark_done(g)
                        log(f'{g} {"完成" if ok else "超时"}')
                        if ok:
                            analyze_group(g)
                    except Exception as e:
                        log(f'{bname}/{g} 组内异常(继续): {e}')
            except Exception as e:
                log(f'{bname} 批次异常(继续): {e}')
        log('=' * 60)
        log('ALL BATCHES COMPLETE - DONE_ALL')
        with open(os.path.join(LOG_DIR, 'DONE_ALL.txt'), 'w', encoding='utf-8') as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
    except Exception as e:
        log(f'MAIN 异常(记录后等待重启): {e}')
        time.sleep(60)


if __name__ == '__main__':
    # [v4] 异常兜底: 永不静默退出 (watchdog 心跳超时会重启, 重启后可断点续跑)
    while True:
        try:
            main()
            # main 正常返回 = 已写 DONE_ALL → 退出
            if os.path.exists(os.path.join(LOG_DIR, 'DONE_ALL.txt')):
                break
        except Exception as e:
            log(f'RUN 异常(60s后重试): {e}')
            time.sleep(60)
