# 这个文件是从ParetoFront.csv中读取数据，然后运行储层评估

import pandas as pd
import numpy as np
import ast
import os
import sys
import traceback
from tqdm import tqdm

# 导入你的现有模块
# 确保 res_configs.py, res_manager.py, res_run.py 等在同一目录下
try:
    from res_configs import Params, ResConfigs, TempConfigs, TIMsConfigs, TaskConfigs
    from res_manager import ResManager
    from res_run import run_reservoir_evaluation
except ImportError as e:
    print("【错误】无法导入必要模块。请确保 res_configs.py, res_manager.py, res_run.py 在当前目录下。")
    raise e

# =========================================================
#  A. 全局配置 (Global Settings)
# =========================================================
# 输入数据文件路径
CSV_FILE_PATH = 'saved_studies/TIMs_Hetero_tempsweep_Task_Pareto_20251208_105813_pareto.csv'


# 运行控制标志 (根据你的 res_run.py 修改)
RUN_FLAGS = {
    'ENABLE_HETERO':  True,        # 必须为 True
    
    # --- 评估模块开关 ---
    'RUN_TIMS':       True,        # 是否跑 TIMs (MC, KR/GR)
    'RUN_TASKS':      True,        # 是否跑 Tasks (TI46, NARMA)

    # --- 细分指标 (TIMs) ---
    'RUN_MC':         True,
    'RUN_KRGR':       True,

    # --- 细分任务 (Tasks) ---
    'RUN_NARMA':      False,       # 示例：关闭 NARMA 以节省时间
    'RUN_TI46':       True,        # 示例：开启 TI46
    
    'N_JOBS':         -1           # 并行核心数
}

# =========================================================
#  B. 数据加载与解析 (Data Loading & Parsing)
# =========================================================
def load_and_parse_csv(file_path):
    """读取 CSV 文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到 CSV 文件: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f">>> 已加载数据文件: {file_path}, 共 {len(df)} 条记录")
    return df

def create_configs_from_row(row):
    """
    核心工厂函数：将 CSV 的一行数据转换为 Manager 所需的配置对象
    """
    try:
        # 1. 提取并解析列表类型的参数 (使用 ast.literal_eval 安全解析字符串列表)
        # 例如: "[10, 20]" -> [10, 20]
        custom_sizes = ast.literal_eval(row['attr_custom_sizes'])
        weights = ast.literal_eval(row['attr_weights'])
        
        # 2. 提取标量参数
        m0 = float(row['param_m0'])
        gamma = float(row['param_gamma'])
        theta = float(row['param_theta'])
        n_instances = int(row['param_n_instances'])
        trial_id = row.get('number', None)

        print(f"custom_sizes: {custom_sizes}")
        print(f"weights: {weights}")
        print(f"m0: {m0}")
        print(f"gamma: {gamma}")
        print(f"theta: {theta}")
        print(f"n_instances: {n_instances}")
        
    except (ValueError, SyntaxError, KeyError) as e:
        print(f"【解析错误】Trial ID {row.get('number', 'Unknown')}: {e}")
        return None

    # 3. 构建 Params 对象
    params_config = Params(
        m0=m0,
        Nvirt=200,          # 固定为 200
        beta_prime=20,# 固定为 20.0
        params={
            'theta': theta,
            'gamma': gamma,
            'Nvirt': 200    # 确保字典中也更新
        }
    )

    # 4. 构建 ResConfigs 对象 (异质储层)
    res_config = ResConfigs(
        morph_type='heterogeneous',
        beta_size_ref=20,
        n_instances=n_instances,
        weights=weights,
        custom_sizes=custom_sizes
    )

    # 5. 构建 TempConfigs (保持原 res_run.py 的温度扫描配置)
    temp_config = TempConfigs(
        temp_mode='temp_sweep',
        beta_temp_ref=20.0,
        temp_range=(19.03, 21.08, 0.5) 
    )

    # 6. 构建 TIMsConfigs (保持默认)
    tims_config = TIMsConfigs(
        mc_signal_len=550,
        mc_seed=1234,
        mc_splits=[0.2, 0.6],
        mc_delays=10,
        krgr_nwash=7,
        krgr_seed=1234,
        krgr_threshold=0.003
    )

    # 7. 构建 TaskConfigs (TI-46 配置)
    task_config = TaskConfigs(
        narma_len=2000,
        narma_train_len=1000,
        narma_test_len=1000,
        narma_seed=1234,
        spacer_NRMSE=0,
        # TI-46 参数
        ti46_speakers=['f1', 'f2', 'f3', 'f4', 'f5'], # 根据需要调整
        ti46_nfft=512,
        ti46_nblocks=4,
        ti46_seed=1234,
        train_temp=20.0
    )

    return params_config, res_config, temp_config, tims_config, task_config

# =========================================================
#  C. 主执行循环 (Main Execution)
# =========================================================
def main():
    # 1. 加载数据
    try:
        df = load_and_parse_csv(CSV_FILE_PATH)
    except Exception as e:
        print(f"Error: {e}")
        return

    success_count = 0
    fail_count = 0
    
    # 2. 遍历 CSV 中的每一行 (使用 tqdm 显示进度条)
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Trials"):
        trial_id = row['number']
        
        # 为每个 Trial 生成唯一的任务名称，防止文件名冲突
        # 例如: Hetero_Pareto_ID_9
        task_name = f"Hetero_Pareto_ID_{trial_id}"
        
        print(f"\n\n>>> [开始处理] Trial ID: {trial_id}")
        
        try:
            # 2.1 创建配置
            configs = create_configs_from_row(row)
            if configs is None:
                fail_count += 1
                continue
            
            p_cfg, r_cfg, t_cfg, tim_cfg, tsk_cfg = configs

            # 2.2 创建管理器 (ResManager)
            # verbose=False 以减少控制台输出，避免刷屏
            manager = ResManager(
                p_cfg, r_cfg, t_cfg, tim_cfg, tsk_cfg, verbose=True
            )

            # 2.3 运行评估
            # 直接调用 res_run.py 中的通用评估函数
            run_reservoir_evaluation(task_name, manager, RUN_FLAGS)
            
            success_count += 1
            print(f">>> [成功完成] Trial ID: {trial_id}")

        except Exception as e:
            print(f">>> [处理失败] Trial ID: {trial_id}")
            print(f"错误信息: {e}")
            traceback.print_exc() # 打印完整堆栈以便调试
            fail_count += 1

    # 3. 总结
    print("\n" + "="*60)
    print(f"批量评估结束")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print("="*60)

if __name__ == '__main__':
    main()