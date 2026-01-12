# 这个文档中函数作用是辅助数据提取等任务
import sys
import os
import glob
import pandas as pd
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


def load_data(folder_path, file_pattern, columns_of_interest=None, file_types=("csv", "pkl")):
    """
    参数:
    folder_path: 数据文件夹路径 (可以是相对路径)
    file_pattern: 文件名前缀匹配模式 (例如 'result_MC_heterogeneous')
    columns_of_interest: 需要保留的列名列表 (例如 ['temp', 'MC']), None则保留所有
    file_types: 允许的文件类型，默认同时支持 ("csv","pkl")

    返回:
    merged_df: 合并后的完整 DataFrame，包含一列 'number'
    """
    # 1. 设置路径
    base_dir = Path.cwd()
    target_dir = (base_dir / folder_path).resolve()

    # 2. 查找文件（同时支持 csv/pkl）
    files = []
    for ext in file_types:
        ext = ext.lstrip(".").lower()
        files.extend(list(target_dir.glob(f"{file_pattern}*.{ext}")))

    # 去重 + 排序（按文件名）
    files = sorted(set(files), key=lambda p: p.name)

    if not files:
        exts = ",".join([f".{e.lstrip('.').lower()}" for e in file_types])
        print(f"警告: 在 {target_dir} 中没有找到匹配 '{file_pattern}*({exts})' 的文件")
        return pd.DataFrame()

    data_list = []
    print(f"找到 {len(files)} 个文件，开始处理...")

    for file in files:
        # 3. 解析文件名提取 Legend
        try:
            filename_parts = file.stem.split("_")
            legend_val = filename_parts[3]
        except IndexError:
            legend_val = "Unknown"

        # 4. 读取数据（csv / pkl）
        suffix = file.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(file)
        elif suffix == ".pkl":
            import pickle
            with open(file, "rb") as f:
                df = pickle.load(f)
            if not isinstance(df, pd.DataFrame):
                print(f"警告: 跳过 {file.name}，原因: pkl 解包后不是 DataFrame，而是 {type(df)}")
                continue
        else:
            print(f"警告: 跳过 {file.name}，原因: 不支持的后缀 {suffix}")
            continue

        # 5. 筛选列（如果指定了列名）
        if columns_of_interest:
            valid_cols = [c for c in columns_of_interest if c in df.columns]
            df = df[valid_cols]

        # 6. 添加标签列
        df["number"] = legend_val
        data_list.append(df)

    # 7. 合并所有数据
    if not data_list:
        return pd.DataFrame()

    merged_df = pd.concat(data_list, ignore_index=True)

    # 可选：如果 number 是数字，转为数字类型以便排序
    try:
        merged_df["number"] = pd.to_numeric(merged_df["number"])
        merged_df = merged_df.sort_values("number")
    except ValueError:
        pass

    return merged_df



def analyze_correlations(
    analysis_results_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_metric: str = "averagenrmse",
    features=None,
    plot: bool = True,
    plot_style_func=None,  # 传入类似 plot_style_config.set_pub_style 的函数；不传则不调用
):
    """
    计算指定特征与目标指标的 Pearson / Spearman 相关性，并可选择绘图。
    
    Parameters
    ----------
    analysis_results_df : pd.DataFrame
        行为指标/特征名，列为不同实验条件的值。
    features : list[str] | None
        要分析的特征行名列表；None 时默认使用 TIMs 相关特征。
    target_metric : str
        目标行名（例如 averagenrmse）；需存在于 analysis_results_df.index。
    plot : bool
        是否绘制热图。
    plot_style_func : callable | None
        若提供，将在绘图前调用（用于统一出版风格等）。
    
    Returns
    -------
    correlation_results : dict
        {feature: {pearson_corr, pearson_p, spearman_corr, spearman_p}}
    correlation_matrix_pearson : pd.DataFrame
    correlation_matrix_spearman : pd.DataFrame
    correlation_matrix_combined : pd.DataFrame
    fig : matplotlib.figure.Figure | None
        绘图对象；plot=False 时为 None。
    """
    default_features = [
         "mean", "median", "std", "var", "amplitude",
        "first_order_sensitivity", "second_order_sensitivity",
    ]
    tims_features = features if features is not None else default_features

    if target_metric not in target_df.columns:
        raise ValueError(f"'{target_metric}' not found in target_df.columns")
    target_values = target_df[target_metric]

    correlation_results = {}

    for feature in tims_features:
        if feature not in analysis_results_df.columns:
            continue
        feature_values = analysis_results_df[feature]
        valid_mask = ~(np.isnan(feature_values) | np.isnan(target_values))
        if valid_mask.sum() < 2:
            correlation_results[feature] = {
                "pearson_corr": np.nan, "pearson_p": np.nan,
                "spearman_corr": np.nan, "spearman_p": np.nan,
            }
            continue

        f_valid = feature_values[valid_mask]
        t_valid = target_values[valid_mask]

        corr_p, p_p = pearsonr(f_valid, t_valid)
        corr_s, p_s = spearmanr(f_valid, t_valid)
        correlation_results[feature] = {
            "pearson_corr": corr_p, "pearson_p": p_p,
            "spearman_corr": corr_s, "spearman_p": p_s,
        }

    # 构建矩阵
    correlation_matrix_pearson = pd.DataFrame(index=[target_metric], columns=tims_features, dtype=float)
    correlation_matrix_spearman = pd.DataFrame(index=[target_metric], columns=tims_features, dtype=float)
    correlation_matrix_combined = pd.DataFrame(index=["Pearson", "Spearman"], columns=tims_features, dtype=float)

    for feature in tims_features:
        if feature in correlation_results:
            correlation_matrix_pearson.loc[target_metric, feature] = correlation_results[feature]["pearson_corr"]
            correlation_matrix_spearman.loc[target_metric, feature] = correlation_results[feature]["spearman_corr"]
            correlation_matrix_combined.loc["Pearson", feature] = correlation_results[feature]["pearson_corr"]
            correlation_matrix_combined.loc["Spearman", feature] = correlation_results[feature]["spearman_corr"]

    # 打印结果
    print(f"特征与 {target_metric} 的相关性分析结果")
    print("=" * 80)
    header = f"{'Feature':<25s} {'Pearson r':<12s} {'Pearson p':<12s} {'Spearman ρ':<12s} {'Spearman p':<12s}"
    print(header)
    print("-" * 80)
    for feature in tims_features:
        if feature not in correlation_results:
            continue
        pearson_corr = correlation_results[feature]["pearson_corr"]
        pearson_p = correlation_results[feature]["pearson_p"]
        spearman_corr = correlation_results[feature]["spearman_corr"]
        spearman_p = correlation_results[feature]["spearman_p"]
        pearson_sig = "***" if pearson_p < 0.001 else "**" if pearson_p < 0.01 else "*" if pearson_p < 0.05 else ""
        spearman_sig = "***" if spearman_p < 0.001 else "**" if spearman_p < 0.01 else "*" if spearman_p < 0.05 else ""
        print(f"{feature:<25s} {pearson_corr:7.4f} {pearson_sig:<3s} {pearson_p:8.4f}   "
              f"{spearman_corr:7.4f} {spearman_sig:<3s} {spearman_p:8.4f}")
    print("=" * 80)
    print("显著性水平: *** p<0.001, ** p<0.01, * p<0.05\n")

    fig = None
    if plot:
        if plot_style_func is not None:
            plot_style_func()
        fig, axes = plt.subplots(1, 2, figsize=(16, 3))

        sns.heatmap(
            correlation_matrix_pearson, annot=True, fmt=".3f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            cbar_kws={"label": "Correlation Coefficient"},
            linewidths=0.5, linecolor="gray", ax=axes[0],
        )
        axes[0].set_xlabel("Features", fontsize=12)
        axes[0].set_ylabel("Target Metric", fontsize=12)
        axes[0].set_title(f"Pearson: Features vs {target_metric}", fontsize=14)

        sns.heatmap(
            correlation_matrix_spearman, annot=True, fmt=".3f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            cbar_kws={"label": "Correlation Coefficient"},
            linewidths=0.5, linecolor="gray", ax=axes[1],
        )
        axes[1].set_xlabel("Features", fontsize=12)
        axes[1].set_ylabel("Target Metric", fontsize=12)
        axes[1].set_title(f"Spearman: Features vs {target_metric}", fontsize=14)

        plt.tight_layout()
        plt.show()

def beta_to_temp(data):
    data = data/20 *293.15-273.15

    return data

