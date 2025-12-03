# 这个文件将根据Configs类来实例化储层
import configparser
from res_configs import Params, ResConfigs, TempConfigs, TIMsConfigs
import numpy as np
from spnc import spnc_anisotropy
from single_node_heteroreservoir import single_node_heteroreservoir
from deterministic_mask import fixed_seed_mask
from single_node_uniformreservoir import single_node_uniformreservoir
import matplotlib.pyplot as plt

class ResManager:
    def __init__(self, params_configs: Params, res_configs: ResConfigs, temp_configs: TempConfigs, tims_configs: TIMsConfigs):
        self.params_configs = params_configs
        self.res_configs = res_configs
        self.temp_configs = temp_configs
        self.tims_configs = tims_configs
        self.mask_object = None

    def _gen_hetero_size(self):
        if self.res_configs.morph_type == 'uniform':
            return [0.0]
        elif self.res_configs.morph_type == 'heterogeneous':
            if self.res_configs.size_range is None or self.res_configs.n_instances is None:
                raise ValueError("size_range and n_instances are required for heterogeneous reservoir")
            if self.res_configs.random_seed is not None:
                np.random.seed(self.res_configs.random_seed)
            size_min = self.res_configs.size_range[0]
            size_max = self.res_configs.size_range[1] 

            return np.random.uniform(size_min, size_max, self.res_configs.n_instances).tolist()

    def _gen_weights(self):
        if self.res_configs.morph_type == 'uniform':
            return []
        if self.res_configs.weights is not None:
            if len(self.res_configs.weights) != self.res_configs.n_instances:
                raise ValueError("The number of weights must be equal to the number of instances")
            return self.res_configs.weights
        else:
            return [1.0/self.res_configs.n_instances] * self.res_configs.n_instances

    # 构建储层实例
    def build_res(self, env_temp: float):
    # 这里有两个容易混淆的参数:
    # 1. 纳米点基准尺寸:beta_size_ref
    # 2. 基准温度: beta_temp_ref
    # 纳米点基准尺寸变化使用deltabeta_list±beta_size_ref来体现
    # 而基准温度变化使用temp_range缩放beta_temp_ref来体现

        if self.res_configs.beta_size_ref is None:
            self.res_configs.beta_size_ref = self.params_configs.beta_prime # 默认使用params_configs.beta_prime
        if self.temp_configs.beta_temp_ref is None:
            self.temp_configs.beta_temp_ref = self.params_configs.beta_prime # 默认使用params_configs.beta_prime

        if self.mask_object is None:
            seed = 1234
            self.mask_object = fixed_seed_mask(1, self.params_configs.Nvirt, self.params_configs.m0, seed)

        if self.res_configs.morph_type == 'uniform':
            uni = spnc_anisotropy(
                h=self.params_configs.h,
                theta_H=self.params_configs.theta_H,
                k_s=self.params_configs.k_s_0,
                phi=self.params_configs.phi,
                beta_prime=self.res_configs.beta_size_ref,
                restart=True
            )
            return uni

        elif self.res_configs.morph_type == 'heterogeneous':

            # 0. 计算纳米点基准尺寸变化
            size_list = self._gen_hetero_size()

            # 1. 计算温度缩放比例：当前temp除以beta_temp_ref
            beta_temp_ref = self.temp_configs.beta_temp_ref
            temp_scale = env_temp / beta_temp_ref

            # 2. 对所有子储层进行温度缩放
            size_list_temp = [size * temp_scale for size in size_list]

            # 3. 构建子储层
            hetero = single_node_heteroreservoir(
                Nin=1,
                Nvirt=self.params_configs.Nvirt,
                Nout=1,
                m0=self.params_configs.m0,
                beta_prime=self.params_configs.beta_prime,
                beta_size_ref=self.res_configs.beta_size_ref,
                size_list=size_list_temp
            )

            return hetero

    def transform(self, signal, env_temp = None):

        # 1. 确定温度模式
        if env_temp is None: # 默认为静态温度模式
            env_temp = self.temp_configs.beta_temp_ref
        
        # 2. 获取基准温度，来锁定input-rate
        beta_temp_ref = self.temp_configs.beta_temp_ref

        # 3. 构建储层
        core = self.build_res(env_temp)

        if self.res_configs.morph_type == 'uniform':
            spn = single_node_uniformreservoir(
                Nin=1,
                Nout=1,
                Nvirt=self.params_configs.Nvirt,
                m0=self.params_configs.m0,
                dilution=1.0,
                identity=False,
                res = core.gen_signal_slow_delayed_feedback_omegaref,
                mask_object=self.mask_object
            )
            S, J = spn.transform(signal, self.params_configs.params, beta_size_ref=self.res_configs.beta_size_ref)
            return S, J

        elif self.res_configs.morph_type == 'heterogeneous':
            weights = self._gen_weights()
            S, J = core.transform(signal, self.params_configs.params, *weights)
            return S, J




#----测试代码----#
if __name__ == "__main__":
    from res_configs import ResConfigs, TempConfigs, TIMsConfigs, Params

    # 1. Configs
    params_configs = Params(Nvirt=1, m0=0.03)

    # 2. ResConfigs
    weights = [0.2, 0.2, 0.8, 0.2, 0.2]
    res_configs_hetero = ResConfigs(morph_type='heterogeneous', n_instances=5, size_range=(-3, 3), weights=weights)
    res_configs_hetero.beta_size_ref = 20 

    # 2.1 ResConfigs for uniform
    res_configs_uniform = ResConfigs(morph_type='uniform')
    res_configs_uniform.beta_size_ref = 20 
    
    # TempConfigs 中应该包含 beta_temp_ref，注意检查定义
    temp_configs = TempConfigs(temp_mode='temp_sweep', beta_temp_ref=20, temp_range=(15, 25, 1))

    tims_configs = TIMsConfigs()

    # 3. build the Manager
    spn_hetero = ResManager(params_configs, res_configs_hetero, temp_configs, tims_configs)
    spn_uniform = ResManager(params_configs, res_configs_uniform, temp_configs, tims_configs)

    # 4. create the signal
    input_signal = np.random.rand(100, 1) - 0.5

    # 5. transform the signal
    env_temp = 25
    S_hetero, J_hetero = spn_hetero.transform(input_signal, env_temp)
    S_uniform, J_uniform = spn_uniform.transform(input_signal, env_temp)

    # 6. plot the result
    plt.plot(S_uniform)
    plt.plot(S_hetero)
    plt.legend(['Uniform', 'Heterogeneous'])
    plt.show()
    



