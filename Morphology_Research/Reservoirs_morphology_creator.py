"""
储层形貌创造模块 (Reservoir Morphology Creator)
===============================================

这个模块实现了对不同形貌储层的配置和创建：
1. MorphologyConfig - 储层形貌配置类
2. ReservoirMorphologyManager - 储层形貌管理器

Author: Chen
Date: 2025-01-XX
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

# 导入现有模块
from spnc import spnc_anisotropy
from single_node_heterogenous_reservoir import single_node_heterogenous_reservoir
from formal_Parameter_Dynamics_Preformance import ReservoirParams


@dataclass 
class MorphologyConfig:
    # This class is to define the configuration of the reservoir morphology
    morph_type: str  # 'uniform', 'gradient', 'normaldistribution','random'
    
    # 异质储层相关参数
    beta_range: Optional[Tuple[float, float]] = None  # beta变化范围
    distribution_type: Optional[str] = None           # 'gradient', 'normaldistribution', 'random'
    random_seed: Optional[int] = None                 # 随机种子
    n_instances: Optional[int] = None                 # 异质储层中的实例数量（仅用于'gradient', 'normaldistribution', 'random'）
    
    def __post_init__(self):
        if self.morph_type in ['gradient', 'normaldistribution', 'random']:
            if self.beta_range is None:
                raise ValueError(f"{self.morph_type} morphology requires beta_range to be specified")
            if self.n_instances is None:
                self.n_instances = 5  # 默认实例数量


class ReservoirMorphologyManager:
    # This class to generate the reservoir with different morphology
    
    def __init__(self):
        self.supported_morphologies = ['uniform', 'gradient', 'normaldistribution','random']
    
    def generate_deltabeta_list(self, config: MorphologyConfig, base_beta: float) -> List[float]:
        """
        this function is to generate subreservoirs with different beta based on the beta_range
        base_beta is the reference beta of the reservoir
        delta is the difference between the beta of the subreservoir and the base_beta
        """
        # uniform reservoir
        if config.morph_type == 'uniform':
            return [0.0]  # uniform reservoir has only one instance, delta=0
        
        # gradient reservoir
        elif config.morph_type == 'gradient':
            if config.beta_range is None or config.n_instances is None:
                raise ValueError("Gradient morphology requires beta_range and n_instances")
            delta_min = config.beta_range[0] - base_beta
            delta_max = config.beta_range[1] - base_beta
            # set the seed to control the reproducibility
            if config.random_seed is not None:
                np.random.seed(config.random_seed)
            # generate the deltabeta list (the gradient distribution with same step size)
            return np.linspace(delta_min, delta_max, config.n_instances).tolist()

        # normal distribution reservoir
        elif config.morph_type == 'normaldistribution':
            if config.beta_range is None or config.n_instances is None:
                raise ValueError("Normal distribution morphology requires beta_range and n_instances")
            if config.random_seed is not None:
                np.random.seed(config.random_seed)
            return np.random.normal(config.beta_range[0], config.beta_range[1], config.n_instances).tolist()
        
        # random reservoir
        elif config.morph_type == 'random':
            if config.beta_range is None or config.n_instances is None:
                raise ValueError("Random morphology requires beta_range and n_instances")
            if config.random_seed is not None:
                np.random.seed(config.random_seed)
            
            delta_min = config.beta_range[0] - base_beta
            delta_max = config.beta_range[1] - base_beta
            return np.random.uniform(delta_min, delta_max, config.n_instances).tolist()
        
        else:
            raise ValueError(f"Unsupported morphology type: {config.morph_type}")

    # 不在ReservoirMorphologyManager中创建储层. 将这一步后移到Reservoirs_morphology_evaluation.py中
    
    # def create_reservoir(self, config: MorphologyConfig, reservoir_params: ReservoirParams):
    #     """Create the reservoir with different morphology"""
    #     if config.morph_type == 'uniform':
    #         # create the uniform reservoir
    #         return spnc_anisotropy(
    #             h=reservoir_params.h,
    #             theta_H=reservoir_params.theta_H,
    #             k_s=reservoir_params.k_s_0,
    #             phi=reservoir_params.phi,
    #             beta_prime=reservoir_params.beta_prime,
    #             restart=True
    #         )
        
    #     elif config.morph_type in ['gradient', 'normaldistribution', 'random']:
    #         # generate the heterogenous reservoir
    #         # first, generate the deltabeta list, according to the config(the design of the reservoir morphology)
    #         deltabeta_list = self.generate_deltabeta_list(config, reservoir_params.beta_prime)
            
    #         # copy the params of the uniform reservoir, to generate a params for the heterogenous reservoir
    #         # because I use the 'single_node_heterogenous_reservoir' function to generate the heterogenous reservoir instead of 'spnc_anisotropy'
    #         temp_params = {
    #             'beta_prime': reservoir_params.beta_prime,
    #             'beta_ref': reservoir_params.beta_prime
    #         }
            
    #         res_params = {
    #             'h': reservoir_params.h,
    #             'm0': reservoir_params.m0,
    #             'deltabeta_list': deltabeta_list
    #         }
            
    #         return single_node_heterogenous_reservoir(
    #             Nin=1,
    #             Nvirt=reservoir_params.Nvirt, 
    #             Nout=1,  
    #             temp_params=temp_params,
    #             res_params=res_params,
    #             dilution=1.0,
    #             identity=False

    #         )
        
    #     else:
    #         raise ValueError(f"Unsupported morphology type: {config.morph_type}")
    
    # def get_transform_function(self, reservoir, config: MorphologyConfig):
    #     """set the transform function for the reservoir"""
    #     if config.morph_type == 'uniform':
    #         return reservoir.gen_signal_fast_delayed_feedback_omegacons
    #     else:
    #         # 需要再次核实下
    #         return reservoir.transform
    
    def generate_weights(self, reservoir, config: MorphologyConfig):
        """
        generate the weights for the heterogenous reservoir
        """
        if config.morph_type == 'uniform':
            return []  # uniform reservoir doesn't need weights
        
        n_instances = len(reservoir.anisotropy_instances)
        
        weights = [1.0/n_instances] * n_instances

        return weights


# 便捷函数
def create_standard_morphology_configs(n_instances: int = 5, 
                                     beta_range: Tuple[float, float] = (20, 30)) -> List[MorphologyConfig]:
    """创建标准的三种形貌配置"""
    return [
        MorphologyConfig(
            morph_type='uniform'
        ),
        MorphologyConfig(
            morph_type='gradient',
            n_instances=n_instances,
            beta_range=beta_range,
            distribution_type='linear'
        ),
        MorphologyConfig(
            morph_type='normaldistribution',
            n_instances=n_instances,
            beta_range=beta_range,
            distribution_type='normaldistribution'
        ),
        MorphologyConfig(
            morph_type='random',
            n_instances=n_instances,
            beta_range=beta_range,
            random_seed=1234
        )
    ] 