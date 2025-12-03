import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from res_configs import Params, ResConfigs, TempConfigs, TIMsConfigs
from res_manager import ResManager
from res_TIMs_evaluation import TIMsEvaluation

def main():

    # 1. 参数配置
    params_uniform = Params(
        m0=0.007, 
        Nvirt=461, 
        beta_prime=16.78,
        params={
            'theta': 0.65, 
            'gamma': 0.146,
            'Nvirt': 461
            })

    params_heterogeneous = Params(
        m0=0.00214942083634672, 
        Nvirt=358, 
        beta_prime=20.0,
        params={
            'theta': 0.37916131466641434, 
            'gamma': 0.11899425227193927, 
            'Nvirt': 358
            })

    # 2. 温度配置
    temp_configs = TempConfigs(
        temp_mode='temp_sweep',
        beta_temp_ref=20.0,
        temp_range=(19.03, 21.08, 0.25) # 等价于温度从5℃到35℃
        )
    
    # 3. TIMs配置
    tims_configs = TIMsConfigs(
        mc_signal_len=550,
        mc_seed=1234,
        mc_splits=[0.2, 0.6],
        mc_delays=10,
        krgr_nwash=7,
        krgr_seed=1234,
        krgr_threshold=0.003
    )

    ####### uniform reservoir #######
    print('###### uniform reservoir #######')
    res_configs_uniform = ResConfigs(
        morph_type='uniform',
        beta_size_ref=16.78
        )

    res_manager_uniform = ResManager(params_uniform, res_configs_uniform, temp_configs, tims_configs, verbose=True)

    evaluation_uniform = TIMsEvaluation(res_manager_uniform)
    results_uniform = evaluation_uniform.evaluate_MC_KRandGR()
    print('###### uniform reservoir evaluation completed #######')

    ####### heterogeneous reservoir #######
    print('###### heterogeneous reservoir #######')
    res_configs_heterogeneous = ResConfigs(
        morph_type='heterogeneous',
        beta_size_ref=20.0,
        n_instances=6,
        size_range=(15, 25),
        weights=[0.16, 0.2, 0.2, 0.15, 0.1, 0.18],
        custom_sizes=[20.2, 22.2, 22.4, 24, 24.6, 24.8]
        )
    res_manager_heterogeneous = ResManager(params_heterogeneous, res_configs_heterogeneous, temp_configs, tims_configs, verbose=True)
    evaluation_heterogeneous = TIMsEvaluation(res_manager_heterogeneous)
    results_heterogeneous = evaluation_heterogeneous.evaluate_MC_KRandGR()
    print('###### heterogeneous reservoir evaluation completed #######')


if __name__ == '__main__':
    main()
