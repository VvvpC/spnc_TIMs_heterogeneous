# -*- coding: utf-8 -*-
"""
24/06/25 Chen (修改于 25/09/09)
基于Optuna和新颖性搜索(Novelty Search)对超顺磁性纳米点储层的行为空间进行探索
-------------------------------------------------------------------------------
此脚本旨在绘制出以下两个指标构成的二维行为空间：

    • CQ  = KR - GR   (核秩减去泛化秩)
    • MC               (线性记忆容量)

与原始版本寻找帕累托前沿不同，此版本采用了新颖性搜索算法，其灵感来源
于 Dale et al., 2019 的论文 "A substrate-independent framework to 
characterize reservoir computers"。

工作原理
==========
1.  **搜索空间** - `HYPERSPACE` 字典定义了每个可调超参数的边界。
2.  **目标函数 (objective)** - 优化的目标不再是最大化CQ或MC，而是最大化
    每个新产生的 (CQ, MC) 点的“新颖性分数”。
        • 对于每次试验(trial)，脚本会采样超参数并计算出 (CQ, MC) 值。
        • 然后，它会访问研究(study)中所有过去已完成的试验结果。
        • 通过计算当前点与历史点在归一化后的 (CQ, MC) 空间中的k-近邻
          平均距离，来得到一个“新颖性分数”。
        • Optuna的任务就是最大化这个分数，从而驱使搜索过程去探索行为
          空间中尚未被发现的、稀疏的区域。
3.  **研究 (Study)** - `optuna.create_study(direction="maximize")` 被
    用于最大化单一目标——新颖性分数。
4.  **结果** - 优化结束后，脚本不再打印最优试验，而是会收集所有探索过
    的点，并使用 Plotly 绘制出一个交互式的散点图，直观地展示出模型所能
    覆盖的整个 (CQ, MC) 行为空间。

如何运行:

bash
python <your_script_name>.py
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import plotly.express as px
import optuna
from optuna.trial import TrialState
import optunahub

# 新增：从scikit-learn导入用于计算距离和数据归一化的工具
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# 导入您自己的评估工具包
from formal_Parameter_Dynamics_Preformance import (
    ReservoirParams,
    evaluate_MC,
    evaluate_KRandGR,
)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Search‑space definition
# ──────────────────────────────────────────────────────────────────────────────
HYPERSPACE = {
    "gamma": (0.045, 0.053),       
    "theta": (0.2, 0.2),       
    "m0": (0.03, 0.055),                   
    "beta_prime": (50, 50),              
}

#──────────────────────────────────────────────────────────────────────────────
# 2. Set the relative parameters
# ──────────────────────────────────────────────────────────────────────────────
# 定义k近邻的邻居数量
k_neighbor = 15

# 定义新颖性搜索的距离阈值
scaler = MinMaxScaler()

# ──────────────────────────────────────────────────────────────────────────────
# 3. Objective function – returns (CQ, MC)
# ──────────────────────────────────────────────────────────────────────────────
def objective(trial: optuna.Trial):
    """Single Optuna trial evaluating CQ and MC."""
    # Sample hyper‑parameters
    gamma = trial.suggest_float("gamma", *HYPERSPACE["gamma"])
    theta = trial.suggest_float("theta", *HYPERSPACE["theta"])
    m0 = trial.suggest_float("m0", *HYPERSPACE["m0"])
    beta_prime = trial.suggest_float("beta_prime", *HYPERSPACE["beta_prime"])

    # Build a ReservoirParams instance with the sampled values
    rparams = ReservoirParams(
        h=0.4,
        m0=m0,
        Nvirt=200,
        beta_prime=beta_prime,
        params={
            "gamma": gamma,
            "theta": theta,
            "Nvirt": 200,
        },
    )

    # Evaluate task‑independent metrics
    mc_dict = evaluate_MC(rparams)            
    kgr_dict = evaluate_KRandGR(rparams)      # Nwash = 7, Nequal = 7

    MC = float(mc_dict.get("MC", 0.0))
    KR = float(kgr_dict.get("KR", 0.0))
    GR = float(kgr_dict.get("GR", 0.0))
    CQ = KR - GR

    # Store detailed results in trial
    trial.set_user_attr("MC", MC)
    trial.set_user_attr("KR", KR)
    trial.set_user_attr("GR", GR)
    trial.set_user_attr("CQ", CQ)

    
    past_trials = [
        t for t in trial.study.get_trials(deepcopy=False) 
        if t.state == TrialState.COMPLETE and t.number != trial.number
    ]

    # give a higher score to ealier trials
    if len(past_trials) < k_neighbor:
        return 10

    # pick all CQ and MC values from past trials
    past_behaviors = np.array([
        [t.user_attrs["CQ"], t.user_attrs["MC"]] for t in past_trials
    ])
    current_behavior = np.array([[CQ, MC]])

    all_behaviors = np.vstack([past_behaviors, current_behavior])

    # 将CQ和MC归一化，是为了避免一个过大数值的指标主导距离计算
    normalized_behaviors = scaler.fit_transform(all_behaviors)

    normalized_current = normalized_behaviors[-1].reshape(1, -1)

    normalized_past = normalized_behaviors[:-1]

    nn_search = NearestNeighbors(n_neighbors=k_neighbor)
    nn_search.fit(normalized_past)
    
    distances, _ = nn_search.kneighbors(normalized_current, k_neighbor)

    novelty_score = np.mean(distances)

    return novelty_score

# ──────────────────────────────────────────────────────────────────────────────
# 4. Study setup
# ──────────────────────────────────────────────────────────────────────────────
def create_study():
    # Create a study, and add a suffix to the study if the study name already exists
    suffix = 0

    # set up the storage and study name
    storage = "sqlite:///db.sqlite3" 
    study_name = "CQ_MC_Novelty"
    new_study_name = study_name

    while True:
        try:
            # try to load the study
            optuna.load_study(study_name=new_study_name, storage=storage)
            # if the study exists, add a suffix to the study name
            suffix += 1
            new_study_name = f"{study_name}_{suffix}"
        except KeyError:
            # if the study does not exist, break the loop
            print(f"Study '{new_study_name}' doesn't exist, Create it。")
            break
    
    module = optunahub.load_module(package="samplers/auto_sampler")


    # set up the object of the study
    study = optuna.create_study(
        # set the samplers
        sampler=module.AutoSampler(),
        # set the direction of the objectives
        direction="maximize",  
        storage=storage,
        study_name=new_study_name,
    )
    
    return study



    