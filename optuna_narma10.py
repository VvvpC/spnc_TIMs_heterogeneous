import os
import sys
import numpy as np
import optuna
from optuna.samplers import NSGAIISampler # 适合多目标优化的采样器

# ------------------------------------------------------------------------------
# 路径配置：确保能导入项目中的模块
# ------------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from res_configs import Params, ResConfigs, TempConfigs, TIMsConfigs, TaskConfigs
from res_manager import ResManager
from res_task_evaluation import TaskEvaluation


HYPERSPACE = {
    "gamma": (0.0, 0.5),
    "theta": (0.01, 0.6),
    "m0": (0.001, 0.008),
    "n_instances": (2, 7),        # 异质点数量范围
    "size_range": (15.0, 40.0),   # 单个纳米点尺寸的搜索范围
    "beta_size_ref": 20.0,  
    "weights": (0.0, 1.0),   # 权重范围
}

# 温度扫描配置
TEMP_CONFIG = {
    "temp_mode": "temp_sweep",
    "beta_temp_ref": 20.0,        # 训练温度 (也是参考温度)
    "temp_range": (19, 21, 0.25)     # 测试温度范围 (Start, Stop, Step)
}

# ------------------------------------------------------------------------------
# 2. Objective Function
# ------------------------------------------------------------------------------
def objective(trial: optuna.Trial):
    # --- A. 采样通用参数 ---
    gamma = trial.suggest_float("gamma", *HYPERSPACE["gamma"])
    theta = trial.suggest_float("theta", *HYPERSPACE["theta"])
    m0 = trial.suggest_float("m0", *HYPERSPACE["m0"])
    
    # --- B. 采样异质储层参数 (关键部分) ---
    n_instances = trial.suggest_int("n_instances", *HYPERSPACE["n_instances"])
    
    custom_sizes = []
    weights = []

    for i in range(n_instances):
            s = trial.suggest_float(f"size_inst_{i}", *HYPERSPACE["size_range"])
            custom_sizes.append(s)

            w = trial.suggest_float(f"weight_inst_{i}", *HYPERSPACE["weights"])
            weights.append(w)

    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    
    # 2. 修正浮点误差 (更安全的做法)
    # 不要去修改最小值，而是修改最大值，或者直接修改最后一个值
    total_weight = sum(weights)
    
    # 只有当误差确实存在时才处理
    if not np.isclose(total_weight, 1.0, atol=1e-9):
        diff = 1.0 - total_weight
        # 找到最大值的索引，将误差加给最大值，这样相对影响最小
        max_index = weights.index(max(weights)) 
        weights[max_index] += diff
    
    # 3. 双重保险：防止因为 diff 为负且权重极小导致的负数
    # 虽然在加给最大值的情况下几乎不可能发生，但作为防御性编程是个好习惯
    weights = [max(0.0, w) for w in weights] 

    # 4. 断言检查
    assert np.isclose(sum(weights), 1.0, atol=1e-5)

    # --- C. 构建配置对象 ---
    params_configs = Params(
        m0=m0,
        beta_prime=HYPERSPACE["beta_size_ref"], # 默认基准
        Nvirt=200,
        params={
            'gamma': gamma,
            'theta': theta,
            'Nvirt': 200 
        }
    )

    # 2. ResConfigs (Heterogeneous Setup)
    res_configs = ResConfigs(
        morph_type='heterogeneous',
        n_instances=n_instances,
        custom_sizes=custom_sizes,  
        weights=weights,
        beta_size_ref=HYPERSPACE["beta_size_ref"]
    )

    # 3. TempConfigs (Training & Sweep)
    temp_configs = TempConfigs(
        temp_mode=TEMP_CONFIG["temp_mode"],
        beta_temp_ref=TEMP_CONFIG["beta_temp_ref"],
        temp_range=TEMP_CONFIG["temp_range"]
    )

    # 4. Other Configs
    tims_configs = TIMsConfigs()
    task_configs = TaskConfigs(
        # NARMA
        narma_len=2000,
        narma_train_len=1000,
        narma_test_len=1000,
        narma_seed=1234,
        spacer_NRMSE=0
    ) 

    manager = ResManager(
            params_configs, res_configs, 
            temp_configs, tims_configs, task_configs, verbose=False
        )

    evaluation = TaskEvaluation(manager)

    # --- E. 执行任务 ---
    try:
        # 注意：testing_narma_at_varying_temperatures 内部会自动：
        # 1. 在 beta_temp_ref (20度) 下训练并固定权重
        # 2. 在 temp_range 下进行测试
        # 设置 n_jobs=1 避免嵌套并行导致的资源死锁，让 Optuna 层面去并行 Trials
        results = evaluation.testing_narma_at_varying_temperatures(n_jobs=-1)

        trial.set_user_attr("weights", weights)
        trial.set_user_attr("custom_sizes", custom_sizes)
        trial.set_user_attr("n_instances", n_instances)
        trial.set_user_attr("gamma", gamma)
        trial.set_user_attr("theta", theta)
        trial.set_user_attr("m0", m0)


    except Exception as e:
        # 如果训练发散或报错，剪枝该 Trial
        print(f"Trial failed with error: {e}")
        raise optuna.exceptions.TrialPruned()


    if not results:
        raise optuna.exceptions.TrialPruned()
        
    nrmse_list = [r['NRMSE'] for r in results]
    
    # 定义目标
    min_nrmse = min(nrmse_list)  # 越小越好（Peak Performance）
    avg_nrmse = np.mean(nrmse_list) # 越小越好 (Average Performance)
    
    # 返回 Tuple (float, float)
    return min_nrmse, avg_nrmse

def create_study():
    suffix = 0
    storage = "sqlite:///db.sqlite3" 
    study_name = f"TIMs_Hetero_tempsweep_Task_Pareto"
    new_study_name = study_name
    
    # 检查研究是否已存在，如存在则添加后缀
    while True:
        try:
            optuna.load_study(study_name=new_study_name, storage=storage)
            suffix += 1
            new_study_name = f"{study_name}_{suffix}"
        except KeyError:
            print(f"create new study: '{new_study_name}'")
            break
    
    sampler = NSGAIISampler()
    
    study = optuna.create_study(
        sampler=sampler,
        directions=["minimize", "minimize"],  # 最小化 NARMA10 NRMSE 和 average NRMSE
        storage=storage,
        study_name=new_study_name,
    )
    
    return study

def run_study(study: optuna.Study, n_trials: int = 300):
    study.optimize(
        lambda trial: objective(trial),
        n_trials=n_trials,
        catch=(ValueError, FloatingPointError),
    )

    print("\nPareto front (all non-dominated trials):")
    for t in study.best_trials:
        print('  Values: ', t.values)
        print('  Params:')
        for key, value in t.params.items():
            print(f'    {key}: {value}')
        print('-------------------')

if __name__ == "__main__":
    study = create_study()
    run_study(study, n_trials=300)



