import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from joblib import Parallel, delayed
import contextlib
import io

from formal_Parameter_Dynamics_Preformance import (
    generate_signal,
    linear_MC,
    gen_KR_GR_input,
    Evaluate_KR_GR,
)

from res_manager import ResManager
from res_configs import Params, ResConfigs, TempConfigs, TIMsConfigs

def _evaluate_MC_single_temp(temp, manager, signal, splits, delays):
    with contextlib.redirect_stdout(io.StringIO()):
        S, J = manager.transform(signal, env_temp=temp)
        MC = linear_MC(signal, S, splits = splits, delays = delays)
        return {
            'temp': temp,
            'MC': MC
        }

def _evaluate_KRandGR_single_temp(temp, manager, input, Nvirt, threshold):
    with contextlib.redirect_stdout(io.StringIO()):
        outputs = []
        for input_row in input:
            input_row = input_row.reshape(-1, 1)
            S, J = manager.transform(input_row, env_temp=temp)
            outputs.append(S)
        States = np.stack(outputs, axis=0)
        States_min = np.amin(States)
        States_max = np.amax(States)
        States = (States - States_min) / (States_max - States_min)

        KR, GR = Evaluate_KR_GR(States, Nvirt, threshold=threshold)
        CQ = KR - GR

        return {
            'temp': temp,
            'KR': KR,
            'GR': GR,
            'CQ': CQ
        }


class TIMsEvaluation:
    def __init__(self, res_manager: ResManager):
        self.res_manager = res_manager
        self.params = res_manager.params_configs
        self.res_configs = res_manager.res_configs
        self.temp_configs = res_manager.temp_configs
        self.tims_configs = res_manager.tims_configs

    # 获得待测试的温度列表
    def _get_temp_list(self):
        return self.temp_configs.gen_temp_list()

    def _get_save_path(self, task_name: str) -> str:
            """
            获取或生成保存路径
            文件名格式: result_{task}_{morph}_[hetero_params]_{common_params}.csv
            """
            # 1. 优先尝试从配置中获取固定路径
            path_attr = f"{task_name.lower()}_result_path"
            if hasattr(self.tims_configs, path_attr):
                custom_path = getattr(self.tims_configs, path_attr)
                if custom_path: # 确保路径不为空字符串
                    return custom_path

            # 2. 构建动态文件名
            # 基础部分
            filename_parts = [f"result_{task_name}", self.res_configs.morph_type]

            # 可选的 trial number（若无则跳过）
            trial_id = getattr(self.res_configs, 'trial_number', None)
            if trial_id is not None and not (isinstance(trial_id, float) and np.isnan(trial_id)) and str(trial_id) != "":
                filename_parts.append(str(trial_id))


            # 异质储层特有部分 (n_instance, deltalis(size_range), weight)
            if self.res_configs.morph_type == 'heterogeneous':
                # n_instances
                filename_parts.append(f"n{self.res_configs.n_instances}")
                
                # size_range (deltalis) -> 格式化为 "range-5to5"
                s_range = self.res_configs.size_range
                if isinstance(s_range, (list, tuple)) and len(s_range) == 2:
                    filename_parts.append(f"sr{s_range[0]}to{s_range[1]}")
                else:
                    filename_parts.append(f"sr{s_range}")

                # weights -> 格式化为 "w0.2-0.2-0.6" 或 "wUniform"
                weights = self.res_configs.weights
                if weights:
                    # 简化权重字符串，避免过长
                    if len(weights) > 7:
                        # 如果权重太多，只显示前两个和长度
                        w_str = f"custom{len(weights)}"
                    else:
                        w_str = "-".join([f"{w:.2g}" for w in weights]) # 保留有效数字，用短横线连接
                    filename_parts.append(f"w{w_str}")
                else:
                    filename_parts.append("wU")

            # 通用部分 (beta_size_ref, beta_temp_ref, temp_mode)
            # 使用 getattr 提供默认值 'NA'，防止属性未定义报错
            beta_size = getattr(self.res_configs, 'beta_size_ref', 'NA')
            # 兼容 temp_ref 和 beta_temp_ref 两种命名
            beta_temp = getattr(self.temp_configs, 'beta_temp_ref', getattr(self.temp_configs, 'temp_ref', 'NA'))
            temp_mode = getattr(self.temp_configs, 'temp_mode', 'NA')
            t_range = self.temp_configs.temp_range
            if isinstance(t_range, (list, tuple)) and len(t_range) == 3:
                    filename_parts.append(f"tr{t_range[0]}to{t_range[1]}s{t_range[2]}")
            else:
                    filename_parts.append(f"tr{t_range}")



            filename_parts.append(f"bs{beta_size}")
            filename_parts.append(f"bt{beta_temp}")
            filename_parts.append(f"{temp_mode}")


            # 组合文件名
            filename = "_".join(map(str, filename_parts)) + ".csv"
            
            return filename

    # 评估MC
    def evaluate_MC(self, n_jobs=-1):
        temp_list = self._get_temp_list()
        results = []

        # 1. 生成信号
        signal_len = self.tims_configs.mc_signal_len
        seed = self.tims_configs.mc_seed
        signal = generate_signal(signal_len, seed=seed)

        # 2. 温度扫描
        results = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_MC_single_temp)(
                temp, 
                self.res_manager, 
                signal, 
                self.tims_configs.mc_splits, 
                self.tims_configs.mc_delays
            )
            for temp in tqdm(temp_list, desc="temp_sweep_evaluate_MC")
        )
        results.sort(key=lambda x: x['temp'])
        df = pd.DataFrame(results)
        save_path = self._get_save_path('MC')
        df.to_csv(save_path, index=False)
        print(f"MC results saved to {save_path}")
        return df

        

    # 评估KR和GR
    def evaluate_KRandGR(self, n_jobs=-1):
        temp_list = self._get_temp_list()
        results = []

        # 1. 生成信号
        input = gen_KR_GR_input(self.params.Nvirt, Nwash=self.tims_configs.krgr_nwash, seed=self.tims_configs.krgr_seed)

        
        # 2. 温度扫描
        results = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_KRandGR_single_temp)(
                temp, 
                self.res_manager, 
                input, 
                self.params.Nvirt, 
                self.tims_configs.krgr_threshold
            )
            for temp in tqdm(temp_list, desc="temp_sweep_evaluate_KRandGR")
        )
        results.sort(key=lambda x: x['temp'])

        df = pd.DataFrame(results)

        save_path = self._get_save_path('KRandGR')
        df.to_csv(save_path, index=False)
        print(f"KR&GR results saved to {save_path}")

        return df


    def evaluate_MC_KRandGR(self):

        # print(f'Start to evaluate {self.res_configs.morph_type} Reservoir')
        # print(f'temp_mode: {self.temp_configs.temp_mode}')
        # if self.res_configs.morph_type == 'heterogeneous':
        #     print(f'morph_type: heterogeneous')
        #     print(f'n_instances: {self.res_configs.n_instances}')
        #     print(f'size_range: {self.res_configs.size_range}')
        #     print(f'weights: {self.res_configs.weights}')
        #     print(f'base_size: {self.res_configs.beta_size_ref}')
        # else:
        #     print(f'morph_type: homogeneous')
        #     print(f'base_size: {self.res_configs.beta_size_ref}')
        # print(f'base_temp: {self.temp_configs.beta_temp_ref}')
        
        
        df_MC = self.evaluate_MC(n_jobs=-1)
        df_KRandGR = self.evaluate_KRandGR(n_jobs=-1)

        return {
            'df_MC': df_MC,
            'df_KRandGR': df_KRandGR
        }

# ---- 测试------#

# if __name__ == '__main__':
#     params = Params(Nvirt=20, m0=0.03)

#     # 形貌配置

#     res_configs = ResConfigs(
#         morph_type='uniform',
#         # n_instances=3,
#         # size_range=(-3, 3), 
#         # weights=[0.2, 0.2, 0.8]
#         )
#     res_configs.beta_size_ref = 20

#     # 温度配置
#     temp_configs = TempConfigs(
#         temp_mode='temp_sweep', 
#         beta_temp_ref=20,
#         temp_range=(18, 23,2),
#         )
#     temp_configs.beta_temp_ref = 20

#     # TIMs配置
#     tims_configs = TIMsConfigs()

#     # 储层管理器
#     res_manager = ResManager(params, res_configs, temp_configs, tims_configs)

#     # 评估管理器
#     tims_evaluator = TIMsEvaluation(res_manager)

#     # 运行评估
#     results = tims_evaluator.evaluate_MC_KRandGR()

            


