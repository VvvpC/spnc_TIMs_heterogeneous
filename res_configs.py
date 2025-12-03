# 这个文件是用来配置储层的形貌、环境温度条件、TIMs的参数

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Literal
import numpy as np


'''
强调:有两个容易混淆的参数：
1. 纳米点基准尺寸:beta_size_ref
2. 基准温度: beta_temp_ref
'''

@dataclass
class ResConfigs: # 这个类是用来配置储层的形貌
    morph_type: str # 'uniform', 'heterogeneous'
    n_instances: int = 1 
    size_range: tuple[float, float] = (20, 30) # 异质点尺寸范围，注意与温度变化范围区别
    weights: list[float] | None = None # 权重范围
    random_seed: int = 1234 # 随机种子
    beta_size_ref: Optional[float] = None # 纳米点基准尺寸

@dataclass
class TempConfigs: # 这个类是用来配置环境温度条件
    temp_mode: str # 'stable', 'temp_sweep'
    beta_temp_ref: Optional[float] = None # 参考温度
    temp_range: tuple[float, float, float] = (-10, 10, 1) # 温度范围

    def gen_temp_list(self):
        if self.temp_mode == 'stable':
            return [self.beta_temp_ref]
        elif self.temp_mode == 'temp_sweep':
            return np.arange(self.temp_range[0], self.temp_range[1], self.temp_range[2]).tolist()

@dataclass
class TIMsConfigs: # 这个类是用来配置TIMs的参数
    # MC
    mc_signal_len: int = 550 # MC信号长度
    mc_seed: int = 1234 # MC信号种子
    mc_splits: list[float] = field(default_factory=lambda: [0.2, 0.6])
    mc_delays: int = 10 # MC信号延迟数量

    # KR&GR
    krgr_nwash: int = 7 # KR&GR冲洗数量
    krgr_seed: int = 1234 # KR&GR信号种子
    krgr_threshold: float = 0.003 # KR&GR信号阈值

@dataclass
class Params:
    def __init__(self, **kwargs):
            # Reservoir parameters 
            self.h = 0.4
            self.theta_H = 90
            self.k_s_0 = 0
            self.phi = 45
            self.beta_prime = 20

            # Network parameters 
            self.Nvirt = 30
            self.m0 = 0.003
            self.bias = True
            self.Nwarmup = 0
            self.verbose_repr = False

            self.params = {
                'theta': 0.5,
                'gamma': 0.1,
                'delay_feedback': 0,
                'Nvirt': self.Nvirt,
                'length_warmup': self.Nwarmup,
                'warmup_sample': self.Nwarmup * self.Nvirt,
                'voltage_noise': False,
                'seed_voltage_noise': 1234,
                'delta_V': 0.1,
                'johnson_noise': False,
                'seed_johnson_noise': 1234,
                'mean_johnson_noise': 0.0000,
                'std_johnson_noise': 0.00001,
                'thermal_noise': False,
                'seed_thermal_noise': 1234,
                'lambda_ou': 1.0,
                'sigma_ou': 0.1
        }

            for key in ['h', 'theta_H', 'k_s_0', 'phi', 'beta_prime', 'Nvirt', 'm0', 'bias', 'Nwarmup']:
                if key in kwargs:
                    setattr(self, key, kwargs[key])

            
            if 'params' in kwargs and isinstance(kwargs['params'], dict):
                self.params.update(kwargs['params'])

    
    def update_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            if key in self.params:
                self.params[key] = value
            if not hasattr(self, key) and key not in self.params:
                raise AttributeError(f"ReservoirParams has no attribute or param key '{key}'")
            


# 整合三个类，
@dataclass
class Configs:
    params_configs: Params
    res_configs: ResConfigs
    temp_configs: TempConfigs
    tims_configs: TIMsConfigs

    def summary(self):
        info =[]

        # 1. 形貌
        info.append(f"Morphology: {self.res_configs.morph_type}")
        if self.res_configs.morph_type == 'heterogeneous':
            info.append(f"Number of instances: {self.res_configs.n_instances}")
            info.append(f"Size range: {self.res_configs.size_range}")
            info.append(f"Weights: {self.res_configs.weights}")

        # 2. 温度
        info.append(f"Temperature mode: {self.temp_configs.temp_mode}")
        if self.temp_configs.temp_mode == 'stable':
            info.append(f"Temperature: {self.temp_configs.beta_temp_ref}")
        elif self.temp_configs.temp_mode == 'temp_sweep':
            info.append(f"Temperature list: {self.temp_configs.gen_temp_list()}")

        # 3. Params
        # 获取 Params 对象中的用户定义属性
        param_obj_attrs = {
            k: v for k, v in vars(self.params_configs).items()
            if not k.startswith("_") and k != "params"  # 排除 params 字典自身
        }

        # 添加 Params 显式属性
        info.append("Params (object attributes):")
        for k, v in param_obj_attrs.items():
            info.append(f"  {k}: {v}")

        # 添加 params 字典内容
        info.append("Params (dictionary):")
        for k, v in self.params_configs.params.items():
            info.append(f"  {k}: {v}")

        return "\n".join(info)



#  ----------------------------- 测试代码 -----------------------------

if __name__ == "__main__":
    # 1. 参数配置
    params_configs = Params(Nvirt=20, m0 = 0.01)

    # 2. 形貌配置
    weights = [0.1, 0.2, 0.3, 0.4, 0.5]
    res_configs = ResConfigs(morph_type='heterogeneous', n_instances=5, size_range=(20, 30), weights=weights)
    
    # 3. 温度配置
    temp_configs = TempConfigs(temp_mode='temp_sweep', temp_range=(15, 50, 1))

    # 4. TIMs配置（默认配置）
    tims_configs = TIMsConfigs()

    # 5. 整合配置
    configs = Configs(params_configs=params_configs, res_configs=res_configs, temp_configs=temp_configs, tims_configs=tims_configs)

    # 6. 打印配置
    print(configs.summary())


        