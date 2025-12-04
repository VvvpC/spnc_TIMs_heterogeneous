import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入所有必要的配置和模块
from res_configs import Params, ResConfigs, TempConfigs, TIMsConfigs, TaskConfigs
from res_manager import ResManager
from res_TIMs_evaluation import TIMsEvaluation
from res_task_evaluation import TaskEvaluation

# 文件名配置函数
def get_smart_filename(manager, task_name,ext = ".csv"):

    res_cfg = manager.res_configs
    temp_cfg = manager.temp_configs
    
    # 1. 基础部分
    filename_parts = [f"result_{task_name}", res_cfg.morph_type]

    # 2. 异质储层特有部分 (n_instances, size_range, weights)
    if res_cfg.morph_type == 'heterogeneous':
        # n_instances
        filename_parts.append(f"n{res_cfg.n_instances}")
        
        # size_range (格式化为 "sr-5to5")
        s_range = res_cfg.size_range
        if isinstance(s_range, (list, tuple)) and len(s_range) == 2:
            filename_parts.append(f"sr{s_range[0]}to{s_range[1]}")
        else:
            filename_parts.append(f"sr{s_range}")

        # weights (格式化为 "w0.2-0.2..." 或 "wCustom")
        weights = res_cfg.weights
        if weights:
            if len(weights) > 7:
                filename_parts.append(f"wCustom{len(weights)}")
            else:
                # 保留2位有效数字，用短横线连接
                w_str = "-".join([f"{w:.2g}" for w in weights]) 
                filename_parts.append(f"w{w_str}")
        else:
            filename_parts.append("wU") # Uniform weights

    # 3. 温度范围 (Temp Range)
    t_range = temp_cfg.temp_range
    if isinstance(t_range, (list, tuple)) and len(t_range) == 3:
        filename_parts.append(f"tr{t_range[0]}to{t_range[1]}s{t_range[2]}")
    else:
        filename_parts.append(f"tr{t_range}")

    # 4. 通用参数 (Beta Size, Beta Temp, Temp Mode)
    # 使用 getattr 防止属性不存在
    beta_size = getattr(res_cfg, 'beta_size_ref', 'NA')
    filename_parts.append(f"bs{beta_size}")

    beta_temp = getattr(temp_cfg, 'beta_temp_ref', getattr(temp_cfg, 'temp_ref', 'NA'))
    filename_parts.append(f"bt{beta_temp}")

    temp_mode = getattr(temp_cfg, 'temp_mode', 'NA')
    filename_parts.append(f"{temp_mode}")

    # 组合文件名
    filename = "_".join(map(str, filename_parts)) + ext
    return filename

# =========================================================
#  核心工作流函数 (Modular Workflow Engine)
# =========================================================
def run_reservoir_evaluation(manager_name, manager, flags):
    """
    通用评估引擎：根据 flags 决定对当前 manager 执行哪些测试
    """
    print(f"\n{'='*20} Start Evaluating: {manager_name} {'='*20}")

    # ---------------------------
    # 1. TIMs 评估模块
    # ---------------------------
    if flags.get('RUN_TIMS', False):
        print(f"[{manager_name}] -> Running TIMs Evaluation...")
        tims_eval = TIMsEvaluation(manager)
        
        # 1.1 MC
        if flags.get('RUN_MC', False):
            tims_eval.evaluate_MC()
            
        # 1.2 KR & GR
        if flags.get('RUN_KRGR', False):
            tims_eval.evaluate_KRandGR()

    # ---------------------------
    # 2. Tasks 评估模块
    # ---------------------------
    if flags.get('RUN_TASKS', False):
            print(f"[{manager_name}] -> Running Tasks Evaluation...")
            task_eval = TaskEvaluation(manager)
            
            # 并行核心数配置
            n_jobs = flags.get('N_JOBS', -1)

            # 2.1 NARMA-10
            if flags.get('RUN_NARMA', False):

                task_eval.generate_narma_data()
                # 变温测试
                results_narma = task_eval.testing_narma_at_varying_temperatures(n_jobs=n_jobs)
                
                # 保存结果
                df_narma = pd.DataFrame(results_narma)
                save_name = get_smart_filename(manager, "NARMA10", ext=".pkl")
                df_narma.to_pickle(save_name)

                print(f"[{manager_name}] NARMA-10 Results Saved to: {save_name}")
                print(df_narma.head()) # 打印前几行预览

            # 2.2 TI-46
            if flags.get('RUN_TI46', False):
                task_eval.prepare_ti46_data()
                results_ti46 = task_eval.testing_ti46_at_varying_temperatures(n_jobs=n_jobs)
                
                # 保存结果
                df_ti46 = pd.DataFrame(results_ti46)
                save_name = get_smart_filename(manager, "TI46", ext=".pkl")
                df_ti46.to_pickle(save_name)
                
                print(f"[{manager_name}] TI-46 Results Saved to: {save_name}")
                print(df_ti46.head())

    print(f"{'='*20} Completed: {manager_name} {'='*20}\n")


def main():
    # =========================================================
    #  A. 控制中心 (Control Flags)
    #  在这里灵活开关你想要运行的内容
    # =========================================================
    RUN_FLAGS = {
        # --- 储层选择 ---
        'ENABLE_UNIFORM': True,       # 是否运行均质储层
        'ENABLE_HETERO':  True,       # 是否运行异质储层

        # --- 评估大类选择 ---
        'RUN_TIMS':       True,       # 是否运行记忆能力评估 (TIMs)
        'RUN_TASKS':      True,       # 是否运行实际任务评估 (Tasks)

        # --- 细分指标选择 (TIMs) ---
        'RUN_MC':         True,       # Memory Capacity
        'RUN_KRGR':       True,       # Kernel/Generalization Rank

        # --- 细分任务选择 (Tasks) ---
        'RUN_NARMA':      True,       # NARMA-10 预测
        'RUN_TI46':       False,      # TI-46 语音识别 (耗时较长，建议单独开)
        
        # --- 系统配置 ---
        'N_JOBS':         -1          # 并行计算核心数 (-1 为全部)
    }

    print(">>> Current Configuration Flags:")
    for k, v in RUN_FLAGS.items():
        print(f"  {k}: {v}")
    print("-" * 30)

    # =========================================================
    #  B. 通用配置 (General Configs)
    # =========================================================
    
    # 1. 温度配置
    temp_configs = TempConfigs(
        temp_mode='temp_sweep',
        beta_temp_ref=20.0,
        temp_range=(19.03, 21.08, 0.25) # 对应物理温度 35℃ 到 5℃
    )
    
    # 2. TIMs 配置
    tims_configs = TIMsConfigs(
        mc_signal_len=550,
        mc_seed=1234,
        mc_splits=[0.2, 0.6],
        mc_delays=10,
        krgr_nwash=7,
        krgr_seed=1234,
        krgr_threshold=0.003
    )

    # 3. Tasks 配置 (新增)
    task_configs = TaskConfigs(
        # NARMA
        narma_len=2000,
        narma_train_len=1000,
        narma_test_len=1000,
        narma_seed=1234,
        spacer_NRMSE=0,
        
        # TI-46
        ti46_speakers=['f1', 'f2', 'f3', 'f4', 'f5'], # 示例：仅使用部分说话人加速测试
        ti46_nfft=512,
        ti46_nblocks=4,
        ti46_seed=1234,
        
        # Common Training
        train_temp=20.0
    )

    # =========================================================
    #  C. 均质储层执行块 (Uniform Reservoir)
    # =========================================================
    if RUN_FLAGS['ENABLE_UNIFORM']:
        # C.1 参数定义
        params_uniform = Params(
            m0=0.007, 
            Nvirt=461, 
            beta_prime=20,
            params={'theta': 0.65, 'gamma': 0.146, 'Nvirt': 461}
        )
        
        res_configs_uniform = ResConfigs(
            morph_type='uniform',
            beta_size_ref=20
        )

        # C.2 构建管理器
        manager_uniform = ResManager(
            params_uniform, res_configs_uniform, 
            temp_configs, tims_configs, task_configs, verbose=True
        )

        # C.3 执行评估工作流
        run_reservoir_evaluation("Uniform_Res", manager_uniform, RUN_FLAGS)


    # =========================================================
    #  D. 异质储层执行块 (Heterogeneous Reservoir)
    # =========================================================
    if RUN_FLAGS['ENABLE_HETERO']:
        # D.1 参数定义
        params_hetero = Params(
            m0=0.00214942083634672, 
            Nvirt=358, 
            beta_prime=20.0,
            params={'theta': 0.379, 'gamma': 0.119, 'Nvirt': 358}
        )

        custom_sizes = [20.2, 22.2, 22.4, 24, 24.6, 24.8]

        res_configs_hetero = ResConfigs(
            morph_type='heterogeneous',
            beta_size_ref=20,
            n_instances=6,
            weights=[0.16, 0.2, 0.2, 0.15, 0.1, 0.18],
            custom_sizes=custom_sizes 
        )

        # D.2 构建管理器
        manager_hetero = ResManager(
            params_hetero, res_configs_hetero, 
            temp_configs, tims_configs, task_configs, verbose=True
        )
        manager_hetero.task_configs = task_configs

        # D.3 执行评估工作流
        run_reservoir_evaluation("Hetero_Res", manager_hetero, RUN_FLAGS)

    # =========================================================
    #  E. 异质储层执行块2 (Heterogeneous Reservoir)
    # =========================================================
    if RUN_FLAGS['ENABLE_HETERO']:
        # D.1 参数定义
        params_hetero2 = Params(
            m0=0.0018142837362546912, 
            Nvirt=149, 
            beta_prime=20.0,
            params={
                'theta': 0.20572464739048968, 
                'gamma': 0.155581477996096, 
                'Nvirt': 149
                })

        custom_size2=[16.299, 17.212, 17.413, 22.812, 23.686]

        res_configs_hetero2 = ResConfigs(
            morph_type='heterogeneous',
            beta_size_ref=20,
            n_instances=5,
            weights=[0.20, 0.28, 0.31, 0.18, 0.02],
            custom_sizes=custom_size2 
        )

        # D.2 构建管理器
        manager_hetero2 = ResManager(
            params_hetero2, res_configs_hetero2, 
            temp_configs, tims_configs, task_configs, verbose=True
        )
        manager_hetero2.task_configs = task_configs

        # D.3 执行评估工作流
        run_reservoir_evaluation("Hetero_Res2", manager_hetero2, RUN_FLAGS)
if __name__ == '__main__':
    main()