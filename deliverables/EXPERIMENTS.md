# 实验登记表（自动生成，勿手改）

> 由 `_exp_version.py` 维护。**INVALID 臂的读数禁止进入任何判据与绩效结论。**

## 冻结基线

- note: `Z23 标签置换泄漏检验：新增 meta_labeling.label_permutation_seed 钩子（默认 None 零影响闭合）；G17 -> G18`
- at: 2026-10-06 10:58:51

- `labels.py` md5 `d10de39112025cbe`（2026-09-19 17:43:19）
- `strategy.py` md5 `b07b160720eca210`（2026-10-06 10:57:14）
- `ensemble.py` md5 `db51860ff1b9ad27`（2026-09-27 18:58:44）
- `target_transforms.py` md5 `0db37b72f8581b87`（2026-09-19 08:58:11）
- `snapshot_fast.py` md5 `a5181167f3540876`（2026-09-16 16:51:09）
- `gpsef/data/local_provider.py` md5 `2eb22b31f26c97c3`（2026-10-05 13:28:56）

## 臂登记

| tag | config | 状态 | 登记时间 | 目的 | 假设 |
|---|---|---|---|---|---|
| `D331_SMOKE` | `config_exp_smoke_D331_o3.yaml` | **registered** | 2026-10-05 13:32:01 | D-331 修复验证：local_provider 缓存淘汰 del->pop(k,None)（5 处同构）后的短窗烟测（唯一变量=窗口 20240401~20240801） | H1: 修复后链路可跑通且四硬全过（ERROR/缺列/训练失败=0、见 meta_IC、调仓完成>0） |
| `N1_orthvol` | `config_exp_Q5N_N1_orthvol_long_o3.yaml` | **valid** | 2026-09-19 18:27:57 | Q5N-N1：目标 ⊥ rank(vol) | H1 剥离 vol 后兑现率优于 N0 |
| `N1_smoke2` | `config_exp_Q5N_N1_orthvol_smoke2_o3.yaml` | **valid** | 2026-09-19 18:27:56 | R-127 烟测（覆盖 2024-05 缺陷区间） | 修复后标签链路可跑通 |
| `N2_fwdret` | `config_exp_Q5N_N2_fwdret_long_o3.yaml` | **valid** | 2026-09-19 18:27:58 | Q5N-N2：同周期前瞻收益 ⊥ {vol,市值} | H2 真选股能力优于 N0 |
| `N2_smoke2` | `config_exp_Q5N_N2_fwdret_smoke2_o3.yaml` | **valid** | 2026-09-19 18:27:57 | R-127 烟测（覆盖 2024-05 缺陷区间） | 修复后标签链路可跑通 |
| `PROBE` | `config_exp_Q5TbC_W4_sham_long_o3.yaml` | **registered** | 2026-09-28 16:32:37 |  |  |
| `Q5B35_P1_PT175` | `config_exp_Q5B35_P1_PT175_long_o3.yaml` | **incomplete** | 2026-09-26 21:43:38 | 波3.5 出场档位定向探针（重启） |  |
| `Q5B35_P2_PT200` | `config_exp_Q5B35_P2_PT200_long_o3.yaml` | **incomplete** | 2026-09-26 21:43:38 | 波3.5 出场档位定向探针（重启） |  |
| `Q5B35_P3_PT125` | `config_exp_Q5B35_P3_PT125_long_o3.yaml` | **incomplete** | 2026-09-26 21:43:38 | 波3.5 出场档位定向探针（重启） |  |
| `Q5B35_P4_SL080` | `config_exp_Q5B35_P4_SL080_long_o3.yaml` | **valid** | 2026-09-26 21:43:38 | 波3.5 出场档位定向探针（重启） |  |
| `Q5C1b_probe_l2_1p5_sh932000` | `config_exp_Q5C1b_probe_l2_1p5_sh932000_long_o3.yaml` | **registered** | 2026-09-20 06:02:08 | Q5C-1b：BARRA 混淆清除 + 风险层探针（λ_l2=1.5） | H3 非集中 BARRA 风险层可压回 |MaxDD| 并守住兑现率 |
| `Q5DT_T1` | `config_exp_Q5DT_trend_ma60_f030_long_o3.yaml` | **registered** | 2026-09-21 07:42:33 | Q5DT_T1 重新登记（配置修正后） | H7 |
| `Q5DT_T2` | `config_exp_Q5DT_trend_ma60_f050_long_o3.yaml` | **registered** | 2026-09-21 07:42:34 | Q5DT_T2 重新登记（配置修正后） | H7 |
| `Q5DT_T3` | `config_exp_Q5DT_trend_ma20_f050_long_o3.yaml` | **registered** | 2026-09-21 07:42:34 | Q5DT_T3 重新登记（配置修正后） | H7 |
| `Q5DT_smoke` | `config_exp_Q5DT_trend_ma60_f050_smoke_o3.yaml` | **valid** | 2026-09-21 07:42:33 | Q5D-T 烟测 v2（修正窗口类型后） | R-126/R-127 四硬指标 |
| `Q5D_vol_tgt_018` | `config_exp_Q5D_vol_tgt_018_long_o3.yaml` | **valid** | 2026-09-20 13:36:08 | Q5D-D2：日频波动率目标敞口（vol_target 0.18，触发率71.5%） | H4（强档）剂量-反应 |
| `Q5D_vol_tgt_024` | `config_exp_Q5D_vol_tgt_024_long_o3.yaml` | **valid** | 2026-09-20 13:36:08 | Q5D-D1：日频波动率目标敞口（vol_target 0.24，触发率42.8%） | H4 波动率目标敞口可改善风险调整后收益（vs PT150） |
| `Q5LAD_LAD_FULL` | `config_exp_Q5LAD_LAD_FULL_long_o3.yaml` | **INVALID** | 2026-10-02 22:39:28 | 波4.9 Q5LAD 阶梯分辨率轴（LAD_FULL）：2×2 因子 阶梯{折叠/拆分}×4.5σ档保护线{+1σ/0σ} | H1: 阶梯拆分本身带来可辨识改善（保护线固定时）；H0: 无差异 ⇒ 不采纳、不冻结 |
| `Q5LAD_LAD_PRICE` | `config_exp_Q5LAD_LAD_PRICE_long_o3.yaml` | **valid** | 2026-10-02 22:39:28 | 波4.9 Q5LAD 阶梯分辨率轴（LAD_PRICE）：2×2 因子 阶梯{折叠/拆分}×4.5σ档保护线{+1σ/0σ} | H1: 阶梯拆分本身带来可辨识改善（保护线固定时）；H0: 无差异 ⇒ 不采纳、不冻结 |
| `Q5LAD_LAD_SL` | `config_exp_Q5LAD_LAD_SL_long_o3.yaml` | **valid** | 2026-10-02 22:39:29 | 波4.9 Q5LAD 阶梯分辨率轴（LAD_SL）：2×2 因子 阶梯{折叠/拆分}×4.5σ档保护线{+1σ/0σ} | H1: 阶梯拆分本身带来可辨识改善（保护线固定时）；H0: 无差异 ⇒ 不采纳、不冻结 |
| `Q5NR_n2_vol015` | `config_exp_Q5NR_n2_vol015_long_o3.yaml` | **registered** | 2026-09-20 21:46:08 | Q5N-R NR-B：N2 目标 × 波动率目标敞口 0.15（强档） | H6b 剂量-反应：强档可把 |MaxDD| 压回 20% 内 |
| `Q5NR_n2_vol021` | `config_exp_Q5NR_n2_vol021_long_o3.yaml` | **valid** | 2026-09-20 22:56:55 | Q5N-R NR-A：N2 目标 × 波动率目标敞口 0.21（敞口对齐 N0）；烟测四硬指标已过 | H6a 敞口对齐后 N2 仍在风险调整后占优（★Sharpe≥1.1587） |
| `Q5NR_n2_vol021_smoke` | `config_exp_Q5NR_n2_vol021_smoke_o3.yaml` | **registered** | 2026-09-20 21:46:09 | Q5N-R 烟测（R-127 缺陷区间 2024-04~08） | R-126/R-127：组合链路四硬指标全过 |
| `Q5PL2_HALF` | `config_exp_Q5PL2_HALF_long_o3.yaml` | **incomplete** | 2026-10-02 22:45:25 | 波4.10 Q5PL2 保护线续扫：s2_sl 1.0->0.5（补 0.0~1.0 之间，验证单调连续性） | H1: eff_sl 越低越好（延续 D-325 三通道严格单调）⇒ 负侧仍改善或见底；H0: 无差异 ⇒ 不采纳、不冻结 |
| `Q5PL2_NEG10` | `config_exp_Q5PL2_NEG10_long_o3.yaml` | **incomplete** | 2026-10-02 22:45:25 | 波4.10 Q5PL2 保护线续扫：s2_sl 1.0->-1.0（单调方向明确延伸） | H1: eff_sl 越低越好（延续 D-325 三通道严格单调）⇒ 负侧仍改善或见底；H0: 无差异 ⇒ 不采纳、不冻结 |
| `Q5PL2_NEG14` | `config_exp_Q5PL2_NEG14_long_o3.yaml` | **incomplete** | 2026-10-02 22:45:24 | 波4.10 Q5PL2 保护线续扫：s2_sl 1.0->-1.4（逼近退化边界 -1.8，余量0.4σ；eff_sl=max(sl,s2_sl)） | H1: eff_sl 越低越好（延续 D-325 三通道严格单调）⇒ 负侧仍改善或见底；H0: 无差异 ⇒ 不采纳、不冻结 |
| `Q5PL3_NEG08` | `config_exp_Q5PL3_NEG08_long_o3.yaml` | **incomplete** | 2026-10-04 11:05:24 | 波4.12 Q5PL3 峰值细扫 右邻域：s2_sl 1.0->-0.8（夹逼峰值下界） | H1: 峰值在 s2_sl≈-1.0 邻域（R-187 单峰加密）⇒ 两侧应低于 -1.0；H0: 平坦/无峰 ⇒ 不采纳 |
| `Q5PL3_NEG12` | `config_exp_Q5PL3_NEG12_long_o3.yaml` | **incomplete** | 2026-10-04 11:05:23 | 波4.12 Q5PL3 峰值细扫 左邻域：s2_sl 1.0->-1.2（夹逼峰值上界） | H1: 峰值在 s2_sl≈-1.0 邻域（R-187 单峰加密）⇒ 两侧应低于 -1.0；H0: 平坦/无峰 ⇒ 不采纳 |
| `Q5PL3_NEG16` | `config_exp_Q5PL3_NEG16_long_o3.yaml` | **INVALID** | 2026-10-04 11:05:24 | 波4.12 Q5PL3 负侧下降段闭合：s2_sl 1.0->-1.6（退化边界-1.8，余量0.2σ） | H1: 峰值在 s2_sl≈-1.0 邻域（R-187 单峰加密）⇒ 两侧应低于 -1.0；H0: 平坦/无峰 ⇒ 不采纳 |
| `Q5PTL_L100` | `config_exp_S17_SL060PT150_long_o3.yaml` | **valid** | 2026-09-27 22:00:05 | PT阶梯s=1.0锚重跑_兼全窗零影响自证_兼dump零行为声明验证 |  |
| `Q5PTL_L125` | `config_exp_Q5PTL_L125_long_o3.yaml` | **valid** | 2026-09-29 00:08:33 | PT阶梯s=1.25低段邻域_早止盈5.625晚7.5/11.25 |  |
| `Q5PTL_L133` | `config_exp_Q5PTL_L133_long_o3.yaml` | **valid** | 2026-09-27 22:00:06 | PT阶梯s=4/3_早止盈6.0晚止盈8.0_12.0 |  |
| `Q5PTL_L150` | `config_exp_Q5PTL_L150_long_o3.yaml` | **valid** | 2026-09-29 00:08:34 | PT阶梯s=1.5低段邻域_早止盈6.75晚9.0/13.5 |  |
| `Q5PTL_L167` | `config_exp_Q5PTL_L167_long_o3.yaml` | **valid** | 2026-09-28 11:21:18 | PT阶梯s=5/3续档_早止盈7.5晚10.0/15.0 |  |
| `Q5PTL_L200` | `config_exp_Q5PTL_L200_long_o3.yaml` | **valid** | 2026-09-28 11:21:19 | PT阶梯s=2.0网格上限_早止盈9.0晚12.0/18.0 |  |
| `Q5P_lam025` | `config_exp_Q5P_lam025_long_o3.yaml` | **valid** | 2026-09-26 02:58:30 |  |  |
| `Q5P_lam050` | `config_exp_Q5P_lam050_long_o3.yaml` | **valid** | 2026-09-26 02:58:30 |  |  |
| `Q5P_lam100` | `config_exp_Q5P_lam100_long_o3.yaml` | **valid** | 2026-09-26 02:58:30 |  |  |
| `Q5P_smoke_lam050` | `config_exp_Q5P_smoke_lam050_o3.yaml` | **incomplete** | 2026-09-24 13:01:39 | 波3 Q5P' 短窗烟测（R-126/R-127；窗 20240401~20240801）：验证越过首个调仓不崩 + 机制生效 | 四硬指标全过 + 出现 [Q5P] vol_penalty λ= 行 + 无 AttributeError |
| `Q5R_D281_A_oldcal_sm` | `config_exp_Q5R_D281_A_oldcal_o3.yaml` | **valid** | 2026-09-23 16:19:55 | 波2.5A Step3 D-281短窗敏感性·对照臂(修复前口径) | 若快照口径修复不改变决策，则A/B的Top-20逐日重合度>=95%、逐轮meta_IC与日收益差可忽略 |
| `Q5R_D281_B_fixedcal_sm` | `config_exp_Q5R_D281_B_fixedcal_o3.yaml` | **valid** | 2026-09-23 16:19:55 | 波2.5A Step3 D-281短窗敏感性·处理臂(修复后口径) | 同A臂假设；两臂差异唯一来源=data.factor_snapshot_dir |
| `Q5R_D281_C_seednull_sm` | `config_exp_Q5R_D281_C_seednull_o3.yaml` | **valid** | 2026-09-23 17:44:53 | 波2.5A Step3b 噪声地板臂：同数据(旧口径)仅换ensemble_seeds，测量Top-20身份固有稳定性 | 若Top-20身份本身不稳，则seed-null的∩/20也应显著低于0.95 ⇒ A/B的30%换股不可辨识于噪声 |
| `Q5R_OPT0_anchor_newcal` | `config_exp_S17_SL060PT150_long_o3.yaml` | **registered** | 2026-09-23 18:49:12 | 波2.5B Step1：修复口径（G9c）采纳包基线新对照锚（全窗 2021-04-01~2026-09-01） | Step4 已判修复改变选股≈30% ⇒ 修复前口径读数不可比 ⇒ 需在修复口径上重建采纳包基线作为波3~6 的比较锚 |
| `Q5R_OPT_dump_full` | `config_exp_Q5R_OPT_dump_full_o3.yaml` | **valid** | 2026-09-23 21:34:36 | 波2.5B：全窗 dump 臂 = 修复口径新对照锚 + 21 臂缓存源 | dump 零行为改变；缓存可被 21 个 replay 臂消费 |
| `Q5R_OPT_random_sm_s0` | `config_exp_Q5R_OPT_random_sm_s0_o3.yaml` | **valid** | 2026-09-23 20:28:11 | 短窗验证-对照端 | replay 必须逐位相同 |
| `Q5R_OPT_replay_full_s00` | `config_exp_Q5R_OPT_replay_full_s00_o3.yaml` | **valid** | 2026-09-24 05:55:41 | 波2.5C：免训练 replay 全窗 s00（N=21 第1臂） | 长窗跑通；审计4/4 |
| `Q5R_OPT_replay_full_s01` | `config_exp_Q5R_OPT_replay_full_s01_o3.yaml` | **valid** | 2026-09-24 06:04:18 | 波2.5C N=21 免训练 replay 分布（s01） | 免训练 replay：审计 A1~A4 全过；与 s00 同框架 |
| `Q5R_OPT_replay_full_s02` | `config_exp_Q5R_OPT_replay_full_s02_o3.yaml` | **valid** | 2026-09-24 06:04:20 | 波2.5C N=21 免训练 replay 分布（s02） | 免训练 replay：审计 A1~A4 全过；与 s00 同框架 |
| `Q5R_OPT_replay_full_s03` | `config_exp_Q5R_OPT_replay_full_s03_o3.yaml` | **valid** | 2026-09-24 06:04:23 | 波2.5C N=21 免训练 replay 分布（s03） | 免训练 replay：审计 A1~A4 全过；与 s00 同框架 |
| `Q5R_OPT_replay_full_s04` | `config_exp_Q5R_OPT_replay_full_s04_o3.yaml` | **valid** | 2026-09-24 06:14:22 | 波2.5C N=21 免训练 replay 分布（s04） | 免训练 replay：审计 A1~A4 全过；与 s00 同框架 |
| `Q5R_OPT_replay_full_s05` | `config_exp_Q5R_OPT_replay_full_s05_o3.yaml` | **valid** | 2026-09-24 06:14:24 | 波2.5C N=21 免训练 replay 分布（s05） | 免训练 replay：审计 A1~A4 全过；与 s00 同框架 |
| `Q5R_OPT_replay_full_s06` | `config_exp_Q5R_OPT_replay_full_s06_o3.yaml` | **valid** | 2026-09-24 06:14:26 | 波2.5C N=21 免训练 replay 分布（s06） | 免训练 replay：审计 A1~A4 全过；与 s00 同框架 |
| `Q5R_OPT_replay_full_s07` | `config_exp_Q5R_OPT_replay_full_s07_o3.yaml` | **valid** | 2026-09-24 06:23:53 | 波2.5C N=21 免训练 replay 分布（s07） | 免训练 replay：审计 A1~A4 全过；与 s00 同框架 |
| `Q5R_OPT_replay_full_s08` | `config_exp_Q5R_OPT_replay_full_s08_o3.yaml` | **valid** | 2026-09-24 06:23:55 | 波2.5C N=21 免训练 replay 分布（s08） | 免训练 replay：审计 A1~A4 全过；与 s00 同框架 |
| `Q5R_OPT_replay_full_s09` | `config_exp_Q5R_OPT_replay_full_s09_o3.yaml` | **valid** | 2026-09-24 06:23:57 | 波2.5C N=21 免训练 replay 分布（s09） | 免训练 replay：审计 A1~A4 全过；与 s00 同框架 |
| `Q5R_OPT_replay_full_s10` | `config_exp_Q5R_OPT_replay_full_s10_o3.yaml` | **valid** | 2026-09-24 06:33:21 | 波2.5C N=21 免训练 replay 分布（s10） | 免训练 replay：审计 A1~A4 全过；与 s00 同框架 |
| `Q5R_OPT_replay_full_s11` | `config_exp_Q5R_OPT_replay_full_s11_o3.yaml` | **valid** | 2026-09-24 06:33:23 | 波2.5C N=21 免训练 replay 分布（s11） | 免训练 replay：审计 A1~A4 全过；与 s00 同框架 |
| `Q5R_OPT_replay_full_s12` | `config_exp_Q5R_OPT_replay_full_s12_o3.yaml` | **valid** | 2026-09-24 06:33:25 | 波2.5C N=21 免训练 replay 分布（s12） | 免训练 replay：审计 A1~A4 全过；与 s00 同框架 |
| `Q5R_OPT_replay_full_s13` | `config_exp_Q5R_OPT_replay_full_s13_o3.yaml` | **valid** | 2026-09-24 06:43:48 | 波2.5C N=21 免训练 replay 分布（s13） | 免训练 replay：审计 A1~A4 全过；与 s00 同框架 |
| `Q5R_OPT_replay_full_s14` | `config_exp_Q5R_OPT_replay_full_s14_o3.yaml` | **valid** | 2026-09-24 06:43:50 | 波2.5C N=21 免训练 replay 分布（s14） | 免训练 replay：审计 A1~A4 全过；与 s00 同框架 |
| `Q5R_OPT_replay_full_s15` | `config_exp_Q5R_OPT_replay_full_s15_o3.yaml` | **valid** | 2026-09-24 06:43:52 | 波2.5C N=21 免训练 replay 分布（s15） | 免训练 replay：审计 A1~A4 全过；与 s00 同框架 |
| `Q5R_OPT_replay_full_s16` | `config_exp_Q5R_OPT_replay_full_s16_o3.yaml` | **valid** | 2026-09-24 06:53:29 | 波2.5C N=21 免训练 replay 分布（s16） | 免训练 replay：审计 A1~A4 全过；与 s00 同框架 |
| `Q5R_OPT_replay_full_s17` | `config_exp_Q5R_OPT_replay_full_s17_o3.yaml` | **valid** | 2026-09-24 06:53:31 | 波2.5C N=21 免训练 replay 分布（s17） | 免训练 replay：审计 A1~A4 全过；与 s00 同框架 |
| `Q5R_OPT_replay_full_s18` | `config_exp_Q5R_OPT_replay_full_s18_o3.yaml` | **valid** | 2026-09-24 06:53:33 | 波2.5C N=21 免训练 replay 分布（s18） | 免训练 replay：审计 A1~A4 全过；与 s00 同框架 |
| `Q5R_OPT_replay_full_s19` | `config_exp_Q5R_OPT_replay_full_s19_o3.yaml` | **valid** | 2026-09-24 07:03:02 | 波2.5C N=21 免训练 replay 分布（s19） | 免训练 replay：审计 A1~A4 全过；与 s00 同框架 |
| `Q5R_OPT_replay_full_s20` | `config_exp_Q5R_OPT_replay_full_s20_o3.yaml` | **valid** | 2026-09-24 07:03:05 | 波2.5C N=21 免训练 replay 分布（s20） | 免训练 replay：审计 A1~A4 全过；与 s00 同框架 |
| `Q5R_OPT_replay_sm_s0` | `config_exp_Q5R_OPT_replay_sm_s0_o3.yaml` | **valid** | 2026-09-23 20:28:12 | 短窗验证-处理端 | 免训练重放等价 |
| `Q5R_OPT_replay_sm_v2` | `config_exp_Q5R_OPT_replay_sm_s0_o3.yaml` | **INVALID** | 2026-09-24 05:39:46 | 波2.5B v2 验证：真免训练 replay（窗 20240401~20240801，缓存 _q5r_cache_short） | 与 v1 replay 逐位等价；且训练类日志归零（R-158） |
| `Q5R_OPT_replay_sm_v3` | `config_exp_Q5R_OPT_replay_sm_s0_o3.yaml` | **INVALID** | 2026-09-24 05:45:28 | 波2.5B v3 验证：真免训练+选股短路生效（窗 20240401~20240801） | 与 v1 replay 逐位等价；训练日志归零；replay 命中=5 |
| `Q5R_OPT_replay_sm_v4` | `config_exp_Q5R_OPT_replay_sm_s0_o3.yaml` | **valid** | 2026-09-24 05:49:54 | 波2.5B v4 验证：免训练+选股短路+meta_auc恢复 | 与 v1 replay 逐位等价；训练日志归零；replay命中=5；无空仓 |
| `Q5R_R0b_random` | `config_exp_Q5R_R0b_random_long_o3.yaml` | **valid** | 2026-09-21 19:46:31 | Q5R 零信息对照：置换预测值↔股票，唯一变量=选股是否含信息 | R-V1 钩子生效；R-5 可反驳：基线若落入随机分布内⇒191+XGBoost 无可检测贡献 |
| `Q5R_R1_random` | `config_exp_Q5R_R1_random_long_o3.yaml` | **valid** | 2026-09-21 19:46:31 | Q5R 零信息对照：置换预测值↔股票，唯一变量=选股是否含信息 | R-V1 钩子生效；R-5 可反驳：基线若落入随机分布内⇒191+XGBoost 无可检测贡献 |
| `Q5R_R2_random` | `config_exp_Q5R_R2_random_long_o3.yaml` | **valid** | 2026-09-21 19:46:32 | Q5R 零信息对照：置换预测值↔股票，唯一变量=选股是否含信息 | R-V1 钩子生效；R-5 可反驳：基线若落入随机分布内⇒191+XGBoost 无可检测贡献 |
| `Q5R_R3_random` | `config_exp_Q5R_R3_random_long_o3.yaml` | **registered** | 2026-09-22 10:21:02 | Q5R 波2：零信息对照第 3 号随机臂（补至 N=6 做 gate 判读） | R-3 正向强证据 / R-4 反向终止 / R-5 可反驳 |
| `Q5R_R4_random` | `config_exp_Q5R_R4_random_long_o3.yaml` | **registered** | 2026-09-22 10:21:03 | Q5R 波2：零信息对照第 4 号随机臂（补至 N=6 做 gate 判读） | R-3 正向强证据 / R-4 反向终止 / R-5 可反驳 |
| `Q5R_R5_random` | `config_exp_Q5R_R5_random_long_o3.yaml` | **registered** | 2026-09-22 10:21:03 | Q5R 波2：零信息对照第 5 号随机臂（补至 N=6 做 gate 判读） | R-3 正向强证据 / R-4 反向终止 / R-5 可反驳 |
| `Q5S2_HIGH2` | `config_exp_Q5S2_HIGH2_long_o3.yaml` | **valid** | 2026-10-01 12:05:13 | Q5S2剂量外推_4.5σ份额0.60到0.40_反向 |  |
| `Q5S2_LOW2` | `config_exp_Q5S2_LOW2_long_o3.yaml` | **valid** | 2026-10-01 12:05:13 | Q5S2剂量外推_4.5σ份额0.60到0.80_单键对转移上界 |  |
| `Q5S2_S2LOOSE` | `config_exp_Q5S2_S2LOOSE_long_o3.yaml` | **valid** | 2026-10-01 12:05:14 | Q5S2保护线轴_s2_sl由1.0到0.0_R196可达键 |  |
| `Q5S2_S2TIGHT` | `config_exp_Q5S2_S2TIGHT_long_o3.yaml` | **valid** | 2026-10-01 12:05:14 | Q5S2保护线轴_s2_sl由1.0到2.0_对称对 |  |
| `Q5S_BEBUF` | `config_exp_Q5S_BEBUF_long_o3.yaml` | **valid** | 2026-09-30 22:33:45 | Q5S出场形状_保本缓冲_s1_sl由0.0到-0.5 |  |
| `Q5S_SHIFT_HIGH` | `config_exp_Q5S_SHIFT_HIGH_long_o3.yaml` | **valid** | 2026-09-30 22:33:44 | Q5S出场形状_份额对_低档份额0.60到0.50_延长持仓时长 |  |
| `Q5S_SHIFT_LOW` | `config_exp_Q5S_SHIFT_LOW_long_o3.yaml` | **valid** | 2026-09-30 22:33:44 | Q5S出场形状_份额对_低档份额0.60到0.70_收缩持仓时长 |  |
| `Q5TbC_W1` | `config_exp_Q5TbC_W1_pt1sl1_long_o3.yaml` | **valid** | 2026-09-27 21:15:07 | 波C'_子轴A关闭_pt1sl1_权重近似均匀 |  |
| `Q5TbC_W1_run309` | `config_exp_Q5TbC_W1_pt1sl1_long_o3.yaml` | **registered** | 2026-09-28 03:22:11 |  |  |
| `Q5TbC_W2` | `config_exp_Q5TbC_W2_pt3sl2_long_o3.yaml` | **valid** | 2026-09-27 21:15:07 | 波C'_用户原假设_止盈重止损略重 |  |
| `Q5TbC_W3` | `config_exp_Q5TbC_W3_pt5sl5_long_o3.yaml` | **valid** | 2026-09-27 21:15:07 | 波C'_子轴A强剂量_pt5sl5 |  |
| `Q5TbC_W4` | `config_exp_Q5TbC_W4_sham_long_o3.yaml` | **valid** | 2026-09-27 21:15:07 | 波C'_sham对照_W2同分布随机置换 |  |
| `Q5TbC_W4_sham` | `config_exp_Q5TbC_W4_sham_long_o3.yaml` | **valid** | 2026-09-28 14:25:04 |  |  |
| `Q5TbX_X05` | `config_exp_Q5TbX_X05_long_o3.yaml` | **valid** | 2026-09-28 13:18:57 |  |  |
| `Q5TbX_X10` | `config_exp_Q5TbX_X10_long_o3.yaml` | **valid** | 2026-09-29 12:04:22 |  |  |
| `Q5TbX_X20` | `config_exp_Q5TbX_X20_long_o3.yaml` | **valid** | 2026-09-29 17:29:58 |  |  |
| `Q5TbX_X50` | `config_exp_Q5TbX_X50_long_o3.yaml` | **valid** | 2026-09-28 11:21:21 | 子轴A饱和探针_pt50sl50_A=25.0 |  |
| `Q5TbX_XS` | `config_exp_Q5TbX_XS_long_o3.yaml` | **valid** | 2026-09-28 11:21:22 | R146 sham_X10同分布随机置换_seed20260929 |  |
| `Z20c_m1` | `config_exp_Z20c_m1_short_o3.yaml` | **valid** | 2026-09-26 03:25:34 | 波3.9 Z20-c 零臂判死闸（探针）: 同窗 20240101~20250131 只换 sample_weight_pt_sl_multiplier=1.0，测 Top-20 重合度 | H1: 权重倍数改变模型打分 => Top-20 变动; H0: 重合>=0.90 且 秩相关>=0.99 => 低于决策分辨率 => 关闭波4波C' |
| `Z20c_m2` | `config_exp_Z20c_m2_short_o3.yaml` | **valid** | 2026-09-26 03:25:35 | 波3.9 Z20-c 零臂判死闸（探针）: 同窗 20240101~20250131 只换 sample_weight_pt_sl_multiplier=2.0，测 Top-20 重合度 | H1: 权重倍数改变模型打分 => Top-20 变动; H0: 重合>=0.90 且 秩相关>=0.99 => 低于决策分辨率 => 关闭波4波C' |
| `Z20c_m5` | `config_exp_Z20c_m5_short_o3.yaml` | **valid** | 2026-09-26 03:25:35 | 波3.9 Z20-c 零臂判死闸（探针）: 同窗 20240101~20250131 只换 sample_weight_pt_sl_multiplier=5.0，测 Top-20 重合度 | H1: 权重倍数改变模型打分 => Top-20 变动; H0: 重合>=0.90 且 秩相关>=0.99 => 低于决策分辨率 => 关闭波4波C' |
| `Z23_SMOKE_S1` | `config_exp_Z23_smokeS1_long_o3.yaml` | **registered** | 2026-10-06 10:58:52 | Z23 泄漏检验烟测：同日标签置换钩子（seed=1）短窗真回测；窗 20240401~20240801（唯一变量=窗口+置换键） | H1: 置换钩子可达且链跑通、四硬全过，且 meta_IC 显著塌向 0（预期 <=0.03）；H0: 未塌 ⇒ 疑似泄漏 |

## 有效性审计（最近一次）

| 臂 | 结论 | ERROR | 训练失败 | 越界 | 零持仓 | 调仓 | 原因 |
|---|---|---|---|---|---|---|---|
| `D331_SMOKE` | - | - | - | - | - | - |  |
| `N1_orthvol` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `N1_smoke2` | ✅VALID | 0 | 0 | 0 | 0 | 5 |  |
| `N2_fwdret` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `N2_smoke2` | ✅VALID | 0 | 0 | 0 | 0 | 5 |  |
| `PROBE` | - | - | - | - | - | - |  |
| `Q5B35_P1_PT175` | - | 0 | 0 | 0 | 0 | 25 |  |
| `Q5B35_P2_PT200` | - | 0 | 0 | 0 | 0 | 25 |  |
| `Q5B35_P3_PT125` | - | 0 | 0 | 0 | 0 | 26 |  |
| `Q5B35_P4_SL080` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5C1b_probe_l2_1p5_sh932000` | - | - | - | - | - | - |  |
| `Q5DT_T1` | - | - | - | - | - | - |  |
| `Q5DT_T2` | - | - | - | - | - | - |  |
| `Q5DT_T3` | - | - | - | - | - | - |  |
| `Q5DT_smoke` | ✅VALID | 0 | 0 | 0 | 0 | 5 |  |
| `Q5D_vol_tgt_018` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5D_vol_tgt_024` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5LAD_LAD_FULL` | ❌INVALID | 1 | 0 | 0 | 0 | 66 | ERROR=1 |
| `Q5LAD_LAD_PRICE` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5LAD_LAD_SL` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5NR_n2_vol015` | - | - | - | - | - | - |  |
| `Q5NR_n2_vol021` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5NR_n2_vol021_smoke` | - | - | - | - | - | - |  |
| `Q5PL2_HALF` | - | 0 | 0 | 0 | 0 | 15 |  |
| `Q5PL2_NEG10` | - | 0 | 0 | 0 | 0 | 15 |  |
| `Q5PL2_NEG14` | - | 0 | 0 | 0 | 0 | 15 |  |
| `Q5PL3_NEG08` | - | 0 | 0 | 0 | 0 | 24 |  |
| `Q5PL3_NEG12` | - | 0 | 0 | 0 | 0 | 25 |  |
| `Q5PL3_NEG16` | ❌INVALID | 1 | 0 | 0 | 0 | 24 | ERROR=1 |
| `Q5PTL_L100` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5PTL_L125` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5PTL_L133` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5PTL_L150` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5PTL_L167` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5PTL_L200` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5P_lam025` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5P_lam050` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5P_lam100` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5P_smoke_lam050` | - | 0 | 0 | 0 | 0 | 1 |  |
| `Q5R_D281_A_oldcal_sm` | ✅VALID | 0 | 0 | 0 | 0 | 5 |  |
| `Q5R_D281_B_fixedcal_sm` | ✅VALID | 0 | 0 | 0 | 0 | 5 |  |
| `Q5R_D281_C_seednull_sm` | ✅VALID | 0 | 0 | 0 | 0 | 5 |  |
| `Q5R_OPT0_anchor_newcal` | - | - | - | - | - | - |  |
| `Q5R_OPT_dump_full` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5R_OPT_random_sm_s0` | ✅VALID | 0 | 0 | 0 | 0 | 5 |  |
| `Q5R_OPT_replay_full_s00` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5R_OPT_replay_full_s01` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5R_OPT_replay_full_s02` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5R_OPT_replay_full_s03` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5R_OPT_replay_full_s04` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5R_OPT_replay_full_s05` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5R_OPT_replay_full_s06` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5R_OPT_replay_full_s07` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5R_OPT_replay_full_s08` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5R_OPT_replay_full_s09` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5R_OPT_replay_full_s10` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5R_OPT_replay_full_s11` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5R_OPT_replay_full_s12` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5R_OPT_replay_full_s13` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5R_OPT_replay_full_s14` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5R_OPT_replay_full_s15` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5R_OPT_replay_full_s16` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5R_OPT_replay_full_s17` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5R_OPT_replay_full_s18` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5R_OPT_replay_full_s19` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5R_OPT_replay_full_s20` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5R_OPT_replay_sm_s0` | ✅VALID | 0 | 0 | 0 | 0 | 5 |  |
| `Q5R_OPT_replay_sm_v2` | ❌INVALID | 0 | 0 | 0 | 88 | 5 | 零持仓轮=88 |
| `Q5R_OPT_replay_sm_v3` | ❌INVALID | 0 | 0 | 0 | 88 | 5 | 零持仓轮=88 |
| `Q5R_OPT_replay_sm_v4` | ✅VALID | 0 | 0 | 0 | 0 | 5 |  |
| `Q5R_R0b_random` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5R_R1_random` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5R_R2_random` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5R_R3_random` | - | - | - | - | - | - |  |
| `Q5R_R4_random` | - | - | - | - | - | - |  |
| `Q5R_R5_random` | - | - | - | - | - | - |  |
| `Q5S2_HIGH2` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5S2_LOW2` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5S2_S2LOOSE` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5S2_S2TIGHT` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5S_BEBUF` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5S_SHIFT_HIGH` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5S_SHIFT_LOW` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5TbC_W1` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5TbC_W1_run309` | - | - | - | - | - | - |  |
| `Q5TbC_W2` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5TbC_W3` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5TbC_W4` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5TbC_W4_sham` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5TbX_X05` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5TbX_X10` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5TbX_X20` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5TbX_X50` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Q5TbX_XS` | ✅VALID | 0 | 0 | 0 | 0 | 66 |  |
| `Z20c_m1` | ✅VALID | 0 | 0 | 0 | 0 | 13 |  |
| `Z20c_m2` | ✅VALID | 0 | 0 | 0 | 0 | 13 |  |
| `Z20c_m5` | ✅VALID | 0 | 0 | 0 | 0 | 13 |  |
| `Z23_SMOKE_S1` | - | - | - | - | - | - |  |

_生成时间：2026-10-06 10:58:52_
