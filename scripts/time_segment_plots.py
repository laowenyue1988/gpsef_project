# -*- coding: utf-8 -*-
"""时间段特征-市值收益分布图 (2026-08-11)

对目标区间 (20220801~20260522) 按 120 交易日分段, 每段全量样本按坐标空间
簇聚类 (1500 点/簇) + 簇内投票染色 (盈利>60%红 / 亏损>60%绿 / 其余黑),
所有段子图同图依次展示, 底部为段间相似性矩阵 (红占比网格 Pearson 相关 + 红单元 IoU)。

用法:
  python scripts/time_segment_plots.py --src results/20260809_010953_compare \
      --neu results/20260808_131106/samples_neutral.parquet \
      --out results/20260811_time_seg
"""
import os
import sys
import argparse
import glob
import time
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'feat_analysis'))
from run_feat_analysis import _cluster_and_vote  # noqa: E402

START_DATE = '20220801'
END_DATE = '20260522'
SEG_DAYS = 120          # 每段交易日数
POINTS_PER_CLUSTER = 1500
POS_THR, NEG_THR = 0.60, 0.60
G = 30                  # 相似性网格粒度
MIN_SAMPLES = 50        # 网格单元最小样本
COL_RED, COL_GREEN, COL_BLACK = '#D62728', '#2CA02C', '#333333'

logger = None


def _log(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)


def extract_features(src_dir):
    """从对比图目录提取特征名 (去 _compare.png 后缀)。"""
    feats = []
    for f in sorted(glob.glob(os.path.join(src_dir, 'plots', '*_compare.png'))):
        feats.append(os.path.basename(f)[:-len('_compare.png')])
    return feats


def build_segments(neu, start, end):
    """按 120 交易日分段 → [(seg_idx, dates, sub_df)]。"""
    sub = neu[(neu['snapshot_date'] >= pd.Timestamp(start)) &
              (neu['snapshot_date'] <= pd.Timestamp(end))].copy()
    days = sorted(sub['snapshot_date'].unique())
    segs = []
    for i in range(0, len(days), SEG_DAYS):
        seg_days = days[i:i + SEG_DAYS]
        seg_df = sub[sub['snapshot_date'].isin(seg_days)]
        segs.append((i // SEG_DAYS, seg_days[0], seg_days[-1], seg_df))
    return segs


def seg_red_ratio_grid(seg_df, feat, xmin, xmax, ymin, ymax, G=G, min_samples=MIN_SAMPLES):
    """段内红占比网格 (30×30): 返回 (r 矩阵, 有效掩码)。"""
    x = seg_df['log_mv'].values
    y = seg_df[feat].values
    lab = seg_df['label'].values
    xi = np.clip((x - xmin) / (xmax - xmin), 0, 1) * (G - 1)
    yi = np.clip((y - ymin) / (ymax - ymin), 0, 1) * (G - 1)
    cell = (yi.astype(np.int64) * G + xi.astype(np.int64))
    n_all = np.bincount(cell, minlength=G * G)
    n_pos = np.bincount(cell[lab == 1], minlength=G * G)
    r = np.zeros(G * G)
    valid = n_all >= min_samples
    r[valid] = n_pos[valid] / n_all[valid]
    return r.reshape(G, G), valid.reshape(G, G)


def segment_similarity(ratios, masks):
    """段间相似性: Pearson 相关 (有效单元交集) + IoU (红单元 r>0.6)。"""
    n = len(ratios)
    corr_m = np.zeros((n, n))
    iou_m = np.zeros((n, n))
    for i in range(n):
        corr_m[i, i] = 1.0
        iou_m[i, i] = 1.0
        for j in range(i + 1, n):
            vm = masks[i] & masks[j]
            if vm.sum() >= 10:
                ri, rj = ratios[i][vm], ratios[j][vm]
                if ri.std() > 0 and rj.std() > 0:
                    corr_m[i, j] = corr_m[j, i] = float(np.corrcoef(ri, rj)[0, 1])
                else:
                    corr_m[i, j] = corr_m[j, i] = 0.0
            else:
                corr_m[i, j] = corr_m[j, i] = 0.0
            Ai = ratios[i] > 0.6
            Bj = ratios[j] > 0.6
            inter = (Ai & Bj).sum()
            union = (Ai | Bj).sum()
            iou_m[i, j] = iou_m[j, i] = inter / union if union else 0.0
    return corr_m, iou_m


def render_seg_axis(ax, seg_df, feat, xmin, xmax, ymin, ymax):
    """像素级渲染一段: 全量点逐点映射, 簇投票色 (与 plot_feature 全量模式一致)。"""
    sub, cluster_color = _cluster_and_vote(seg_df, feat, POINTS_PER_CLUSTER,
                                           POS_THR, NEG_THR)
    H, W = 240, 960
    xi = np.clip((sub['log_mv'].values - xmin) / (xmax - xmin), 0, 1) * (W - 1)
    yi = np.clip((sub[feat].values - ymin) / (ymax - ymin), 0, 1) * (H - 1)
    flat = (yi.astype(np.int64) * W + xi.astype(np.int64))
    col_map = {COL_RED: 0, COL_GREEN: 1, COL_BLACK: 2}
    cid = sub['_cl'].map(cluster_color).map(col_map).values.astype(np.int64)
    cnts = [np.bincount(flat[cid == k], minlength=H * W) for k in range(3)]
    img_c = np.argmax(np.stack(cnts, axis=1), axis=1)
    img_c[(cnts[0] + cnts[1] + cnts[2]) == 0] = 3
    rgb = np.array([[214, 39, 40], [44, 160, 44], [51, 51, 51], [255, 255, 255]],
                   dtype=np.uint8)
    img = rgb[img_c].reshape(H, W, 3)
    ax.imshow(img, extent=[xmin, xmax, ymin, ymax], origin='lower',
              aspect='auto', interpolation='nearest')
    n_cl = sub['_cl'].nunique()
    return len(sub), n_cl


def plot_time_segments(feat, segs, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    # 全区间合并坐标范围 (统一 x/y, 便于跨段比较)
    all_x = pd.concat([s[3]['log_mv'] for s in segs])
    all_y = pd.concat([s[3][feat] for s in segs])
    xmin, xmax = float(all_x.min()), float(all_x.max())
    ymin, ymax = float(all_y.min()), float(all_y.max())
    padx = (xmax - xmin) * 0.02 + 1e-12
    pady = (ymax - ymin) * 0.02 + 1e-12
    xmin -= padx; xmax += padx; ymin -= pady; ymax += pady

    n_seg = len(segs)
    # 前 n_seg 行为时间段子图; 底部矩阵用 add_axes 手动放置 (避免 tight_layout 覆盖)
    fig, axes = plt.subplots(n_seg, 1, figsize=(10, 2.5 * n_seg + 4.2))
    if n_seg == 1:
        axes = [axes]
    seg_infos = []
    for k, (idx, d0, d1, seg_df) in enumerate(segs):
        ax = axes[k]
        n_points, n_cl = render_seg_axis(ax, seg_df, feat, xmin, xmax, ymin, ymax)
        seg_infos.append((idx, d0, d1, n_points, n_cl))
        ax.set_title(f'段{idx + 1}  {d0:%Y-%m-%d} ~ {d1:%Y-%m-%d}  '
                     f'(n={n_points:,}, {n_cl}簇, ~{POINTS_PER_CLUSTER:,}点/簇)',
                     fontsize=9)
        ax.set_xticks([])
        if k < n_seg - 1:
            ax.tick_params(labelleft=False)
        else:
            ax.set_xlabel('log(市值)')
        ax.set_ylabel(feat, fontsize=9)
        ax.grid(alpha=0.2)

    # 段间相似性矩阵
    ratios, masks = [], []
    for idx, d0, d1, seg_df in segs:
        r, m = seg_red_ratio_grid(seg_df, feat, xmin, xmax, ymin, ymax)
        ratios.append(r)
        masks.append(m)
    corr_m, iou_m = segment_similarity(ratios, masks)
    seg_labels = [f'{d0:%m%d}-{d1:%m%d}' for _, d0, d1, _, _ in seg_infos]
    # 底部两个独立矩阵轴 (手动绝对定位, 不参与 tight_layout)
    ax_iou = fig.add_axes([0.07, 0.10, 0.40, 0.30])
    ax_corr = fig.add_axes([0.55, 0.10, 0.40, 0.30])
    im_iou = ax_iou.imshow(iou_m, cmap='YlOrRd', vmin=0, vmax=1)
    for i in range(n_seg):
        for j in range(n_seg):
            ax_iou.text(j, i, f'{iou_m[i, j]:.2f}', ha='center', va='center',
                        fontsize=7, color='#222')
    ax_iou.set_xticks(range(n_seg))
    ax_iou.set_yticks(range(n_seg))
    ax_iou.set_xticklabels(seg_labels, rotation=45, fontsize=7)
    ax_iou.set_yticklabels(seg_labels, fontsize=7)
    ax_iou.set_title(f'段间红单元 IoU', fontsize=9)
    im_corr = ax_corr.imshow(corr_m, cmap='RdYlBu', vmin=-1, vmax=1)
    for i in range(n_seg):
        for j in range(n_seg):
            ax_corr.text(j, i, f'{corr_m[i, j]:.2f}', ha='center', va='center',
                         fontsize=7, color='#222')
    ax_corr.set_xticks(range(n_seg))
    ax_corr.set_yticks(range(n_seg))
    ax_corr.set_xticklabels(seg_labels, rotation=45, fontsize=7)
    ax_corr.set_yticklabels(seg_labels, fontsize=7)
    ax_corr.set_title(f'段间红占比 Pearson 相关', fontsize=9)
    fig.text(0.5, 0.045,
             f'相似性口径: {G}×{G} 网格 (单元≥{MIN_SAMPLES}样本, 范围=全区间合并值域); '
             f'IoU=红单元(r>60%) Jaccard, Corr=红占比 Pearson',
             ha='center', va='bottom', fontsize=8)

    fig.suptitle(f'{feat}  vs  log(市值): 按120交易日分段  (全量样本, {POINTS_PER_CLUSTER:,}点/簇, '
                 f'簇内投票: 盈利>{int(POS_THR*100)}%红/亏损>{int(NEG_THR*100)}%绿/其余黑)',
                 fontsize=11)
    # 手动布局: 上段子图区 + 底部矩阵区 (不用 tight_layout, 避免覆盖 add_axes 位置)
    fig.subplots_adjust(top=0.955, bottom=0.42, hspace=0.40)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return corr_m, iou_m, seg_infos


def main():
    global logger
    ap = argparse.ArgumentParser(description='时间段特征分布图 (120交易日分段)')
    ap.add_argument('--src', required=True, help='对比图目录 (提取特征列表)')
    ap.add_argument('--neu', default=None, help='全量中性化数据 parquet (默认 results/20260808_131106)')
    ap.add_argument('--out', default='results/20260811_time_seg', help='输出目录')
    ap.add_argument('--points-per-cluster', type=int, default=1500,
                    help='每簇点数 (默认 1500)')
    ap.add_argument('--limit', type=int, default=0, help='测试用: 只处理前 N 个特征')
    args = ap.parse_args()

    global POINTS_PER_CLUSTER
    POINTS_PER_CLUSTER = args.points_per_cluster

    neu_path = args.neu or os.path.join(PROJECT_ROOT, 'results', '20260808_131106',
                                        'samples_neutral.parquet')
    feats = extract_features(args.src)
    if args.limit > 0:
        feats = feats[:args.limit]
    _log(f"特征数: {len(feats)} | 数据源: {neu_path} | 分段: {SEG_DAYS}交易日/段 | "
         f"每簇: {POINTS_PER_CLUSTER:,} 点")

    out_dir = args.out if os.path.isabs(args.out) else os.path.join(PROJECT_ROOT, args.out)
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # 加载全区间 neu (只读需要列; score 特征由基础列现算, 不读 score 列本身)
    needed = set()
    for f in feats:
        if f.endswith('_score'):
            needed.add(f[:f.rfind('_')])   # score → 基础列 (现算 tanh)
        else:
            needed.add(f)
    cols = ['ts_code', 'snapshot_date', 'log_mv', 'label'] + sorted(needed - {'log_mv', 'label'})
    t0 = time.time()
    neu = pd.read_parquet(neu_path, columns=cols)
    _log(f"neu 加载: {len(neu):,} 行 x {len(cols)} 列 ({time.time()-t0:.0f}s)")

    # score 特征现算 (与 compute_score_conf 同语义) — 必须先于 build_segments,
    # 否则段切片(布尔掩码)列集固定看不到新列
    for f in feats:
        if f.endswith('_score') and f not in neu.columns:
            neu[f] = np.tanh(neu[f[:f.rfind('_')]].fillna(0))

    segs = build_segments(neu, START_DATE, END_DATE)
    _log(f"分段: {len(segs)} 段")

    summary_rows = []
    for i, feat in enumerate(feats):
        t1 = time.time()
        segs_f = [(idx, d0, d1, df[['log_mv', feat, 'label']].dropna(subset=[feat]))
                  for idx, d0, d1, df in segs]
        out_path = os.path.join(plots_dir, f'{feat}_timeseg.png')
        corr_m, iou_m, infos = plot_time_segments(feat, segs_f, out_path)
        dt = time.time() - t1
        # 汇总行: 每段对
        n = len(infos)
        for a in range(n):
            for b in range(a + 1, n):
                summary_rows.append({
                    'feature': feat,
                    'seg_a': a + 1, 'seg_b': b + 1,
                    'iou': iou_m[a, b], 'corr': corr_m[a, b],
                })
        _log(f"[{i + 1}/{len(feats)}] {feat} 完成 ({dt:.0f}s), 剩余约 "
             f"{dt * (len(feats) - i - 1) / 60:.0f} 分钟")
        if i % 5 == 4:
            _log(f"  进度: {i + 1}/{len(feats)}")

    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, 'segment_similarity.csv'),
                                      index=False)
    # 报告
    with open(os.path.join(out_dir, 'time_seg_report.md'), 'w', encoding='utf-8') as fp:
        fp.write('\n'.join([
            '# 时间段特征分布图报告', '',
            f'- 区间: {START_DATE} ~ {END_DATE}',
            f'- 分段: 每 {SEG_DAYS} 交易日一段 ({len(segs)} 段)',
            f'- 每段: 全量样本, {POINTS_PER_CLUSTER:,} 点/簇 坐标空间等频网格聚类',
            f'- 染色: 簇内盈利>{int(POS_THR*100)}%红 / 亏损>{int(NEG_THR*100)}%绿 / 其余黑',
            f'- 底部: 段间相似性矩阵 (红占比 {G}×{G} 网格 Pearson 相关 + 红单元 IoU)',
            f'- 特征数: {len(feats)} | 输出: {out_dir}/plots/',
            f'- 明细: segment_similarity.csv (每特征段对 IoU/Corr)',
            '',
            f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        ]))
    _log(f"全部完成: {len(feats)} 特征 -> {out_dir} (总耗时 {(time.time()-t0)/60:.0f} 分钟)")


if __name__ == '__main__':
    main()
