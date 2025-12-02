'''
此代码的目的是从trials中提取部分的支配点,并评估其NARMA10和TI46任务性能。
'''

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Sequence
import os

def _parse_values_to_array(values_series: pd.Series) -> np.ndarray:
    """
    将字符串形式的 '[x, y, ...]' 解析为二维 numpy 数组。
    对非法/缺失值行予以丢弃（返回时保持与原索引对齐的过滤）。
    """
    parsed = []
    valid_idx = []
    for idx, s in values_series.items():
        try:
            # 允许 s 已经是 list/tuple 的情况
            v = s if isinstance(s, (list, tuple)) else json.loads(str(s))
            v = np.array(v, dtype=float).ravel()
            parsed.append(v)
            valid_idx.append(idx)
        except Exception:
            # 跳过解析失败的行
            pass
    if not parsed:
        return np.empty((0, 0)), pd.Index([])
    # 保证各行维度一致
    lens = [len(v) for v in parsed]
    mode_dim = max(set(lens), key=lens.count)
    keep = [i for i, v in enumerate(parsed) if len(v) == mode_dim]
    arr = np.vstack([parsed[i] for i in keep])
    kept_idx = pd.Index([valid_idx[i] for i in keep])
    return arr, kept_idx

def _min_dist_to_set(X: np.ndarray, Y: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """
    对于 X 中每个点，计算到 Y 集合的最小欧氏距离（支持较大规模，分块计算）。
    X: (n, d), Y: (m, d)
    返回: (n,)
    """
    if X.size == 0 or Y.size == 0:
        return np.full((X.shape[0],), np.inf)
    n = X.shape[0]
    out = np.empty(n, dtype=float)
    # 分块以避免内存爆炸
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        Xi = X[start:end]  # (b, d)
        # (b, m, d) -> (b, m) 距离，取 min
        # 使用 (x - y)^2 = x^2 + y^2 - 2xy 的展开，避免大中间张量
        X2 = np.sum(Xi**2, axis=1, keepdims=True)   # (b,1)
        Y2 = np.sum(Y**2, axis=1, keepdims=True).T  # (1,m)
        d2 = X2 + Y2 - 2.0 * Xi @ Y.T               # (b,m)
        d2 = np.maximum(d2, 0.0)
        out[start:end] = np.sqrt(np.min(d2, axis=1))
    return out

def _robust_scale(arr: np.ndarray, method: str = "mad", eps: float = 1e-9) -> np.ndarray:
    """
    对每一维做稳健尺度缩放，减弱量纲差异的影响。
    method: 'mad' or 'iqr' or 'none'
    """
    if method == "none":
        return arr
    Q1 = np.percentile(arr, 25, axis=0)
    Q2 = np.percentile(arr, 50, axis=0)
    Q3 = np.percentile(arr, 75, axis=0)
    if method.lower() == "iqr":
        scale = np.maximum(Q3 - Q1, eps)
        return (arr - Q2) / scale
    # MAD
    mad = np.median(np.abs(arr - Q2), axis=0)
    scale = np.maximum(1.4826 * mad, eps)
    return (arr - Q2) / scale

def select_trials_near_pareto(
    trials_csv: str,
    pareto_csv: str,
    out_csv: str,
    n_select: int = 100,
    distance_scale: str = "iqr",        # 'mad' | 'iqr' | 'none'
    temperature: Optional[float] = None,# 若为 None，使用距离的 p50 作为 τ
    hard_quantile: Optional[float] = 0.95, # 先丢弃距离 > 该分位数 的点；None 表示不加硬阈值
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """
    从 trials 中挑选靠近 pareto 的一部分 trial，且不与 pareto 的 number 重合；
    并按 pareto 的列结构保存到 out_csv。返回保存的 DataFrame。
    """
    rng = np.random.default_rng(random_state)

    # 读取（支持外部文件夹的文件路径），明确指定 saved_studies 文件夹地址
    SAVED_STUDIES_DIR = r"C:\Users\Chen\Desktop\Repository\spnc_taskindependent_metrics\saved_studies"
    def _resolve_path(p):
        if os.path.isabs(p) and os.path.exists(p):
            return p
        # 1) 绝对路径且存在直接用
        # 2) 相对路径尝试当前工作目录
        if os.path.exists(p):
            return os.path.abspath(p)
        # 3) 否则尝试 saved_studies 文件夹
        possible = os.path.join(SAVED_STUDIES_DIR, p)
        if os.path.exists(possible):
            return possible
        raise FileNotFoundError(f"无法找到文件: {p}")

    trials = pd.read_csv(_resolve_path(trials_csv), low_memory=False)
    pareto = pd.read_csv(_resolve_path(pareto_csv), low_memory=False)

    # 1) 列顺序按 pareto 文件对齐
    target_cols = list(pareto.columns)

    # 2) 去重：number 不重合（必要条件）
    pareto_ids = set(pareto["number"].tolist())
    candidates = trials[~trials["number"].isin(pareto_ids)].copy()

    # 3) 解析 values 为向量（候选 & pareto）
    Y_vals, y_idx = _parse_values_to_array(pareto["values"])
    if Y_vals.size == 0:
        raise ValueError("Pareto 文件的 'values' 无法解析，请检查格式。")
    # 注意：对候选的 values 也要解析
    X_vals_all, x_idx_all = _parse_values_to_array(candidates["values"])
    # 需要把 candidates 过滤到解析成功的行
    candidates = candidates.loc[x_idx_all].copy()
    X_vals = X_vals_all

    # 4) 可选的稳健缩放（避免某一维主导）
    X_scaled = _robust_scale(X_vals, method=distance_scale)
    Y_scaled = _robust_scale(Y_vals, method=distance_scale)

    # 5) 计算到 pareto 集合的最小距离
    d = _min_dist_to_set(X_scaled, Y_scaled)  # (len(candidates),)
    candidates["dist_to_pareto"] = d

    # 6) 可选：硬阈值过滤（剔除过远的点）
    if hard_quantile is not None:
        thr = float(np.quantile(d[~np.isinf(d)], hard_quantile))
        candidates = candidates[candidates["dist_to_pareto"] <= thr].copy()

    if candidates.empty:
        raise ValueError("靠近 pareto 的候选为空，请放宽阈值或检查数据。")

    # 7) 软选择权重（越近权重越高）
    d_now = candidates["dist_to_pareto"].to_numpy()
    if temperature is None:
        # 用距离中位数作为温度，避免数值过尖或过平
        finite_d = d_now[np.isfinite(d_now)]
        tau = np.median(finite_d) if finite_d.size else 1.0
        tau = tau if tau > 1e-12 else 1.0
    else:
        tau = float(temperature)

    weights = np.exp(-d_now / tau) + 1e-9
    weights = np.nan_to_num(weights, nan=1e-9, posinf=1e-9, neginf=1e-9)
    weights = weights / weights.sum()

    # 8) 采样 n_select 个（无放回）
    k = min(int(n_select), len(candidates))
    chosen_idx = rng.choice(len(candidates), size=k, replace=False, p=weights)
    chosen = candidates.iloc[chosen_idx].copy()

    # 9) 按 pareto 的列结构组织并保存
    #    注意：values 保持原字符串格式；只拣选 pareto 文件中出现的列
    missing_cols = [c for c in target_cols if c not in chosen.columns]
    if missing_cols:
        # 从 trials -> pareto 的列映射里，pareto 的列 trials 都有；如果缺，说明 trials 数据缺失
        raise ValueError(f"Trials 缺少以下列，无法对齐到 Pareto 格式：{missing_cols}")

    out_df = chosen[target_cols].copy()
    # 保证 'values' 为字符串（若中途被解析过/修改）
    out_df["values"] = out_df["values"].astype(str)

    # 明确文件保存的文件夹
    fixed_folder = r"C:\Users\Chen\Desktop\Repository\spnc_taskindependent_metrics\saved_studies"
    out_path = Path(fixed_folder) / Path(out_csv).name
    Path(fixed_folder).mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return out_df, str(out_path)


def evaluate_nearpareto_with_existing_evaluator(
    nearpareto_csv_path: str,
    *,
    narma_config: dict | None = None,
    ti46_config: dict | list | None = None,
    ti46_nvirt: int | None = 150,
    output_filename: str | None = None
):
    """
    调用 CQ_MC_ParetofrontPoints.py 的现有评估流程，评估 nearpareto.csv 中的点。
    不修改对方文件，仅通过 import & 方法调用。
    """
    import os
    from pathlib import Path

    # 延迟导入，避免文件顶层 import 时报路径问题
    from CQ_MC_ParetofrontPoints import ParetoPointEvaluator  # 不改动该文件

    nearpareto_path = Path(nearpareto_csv_path).resolve()
    work_dir = nearpareto_path.parent
    csv_basename = nearpareto_path.name  # 只传文件名，配合对方的 "**/*{filename}*" 搜索

    cwd_backup = Path.cwd()
    try:
        os.chdir(work_dir)  # 关键：让对方的 Path(".").glob() 能找到这个 csv
        evaluator = ParetoPointEvaluator()

        evaluator.evaluate_all_points(
            source=csv_basename,
            narma_config=narma_config or {'Ntrain': 2000, 'Ntest': 1000},
            ti46_config=ti46_config or ['f1', 'f2', 'f3', 'f4', 'f5'],
            ti46_nvirt=ti46_nvirt,
            output_filename=output_filename or Path(csv_basename).stem
        )
    finally:
        os.chdir(cwd_backup)

if __name__ == "__main__":
    out_df, out_path = select_trials_near_pareto(
        trials_csv="CQ_MC_Pareto_beta50_20250825_121711_trials.csv",
        pareto_csv="CQ_MC_Pareto_beta50_20250825_121711_pareto.csv",
        out_csv="CQ_MC_Pareto_beta50_20250825_121711_dominatedpoints.csv",
        n_select=40,
        distance_scale="iqr",
        temperature=None,
        hard_quantile=0.95,
        random_state=2024,
    )

    # 串联调用现有的 Paretofront 评估器（不修改其源码）
    evaluate_nearpareto_with_existing_evaluator(
        out_path,
        ti46_nvirt=150,  # 与你在对方脚本 main 中的示例保持一致
        # 也可以定制 output_filename，默认用 nearpareto 的文件名（去 .csv）
        # output_filename="NearPareto_TaskResults"
    )
