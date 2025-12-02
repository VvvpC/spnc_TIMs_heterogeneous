"""
这个test的目的是从pareto前沿文件中提取参数,创建储层,遍历所有均一化方法执行KR和GR评估,并返回其sKR和sGR
"""

import ast
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union
from enum import Enum
import numpy as np
import pandas as pd

parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# 只保留真正需要的导入
from spnc import spnc_anisotropy
from formal_Parameter_Dynamics_Preformance import ReservoirParams
from Morphology_Research.Reservoirs_morphology_creator import ReservoirMorphologyManager
from single_node_res import single_node_reservoir
from deterministic_mask import fixed_seed_mask, max_sequences_mask

class NormalizationMethod(Enum):
    """定义所有可用的归一化方法"""
    nmax = "divide_max"
    n1to1 = "range_-1_to_1"
    n0to1 = "range_0_to_1"
    nwithout = "no_normalization"


def normalize_states(states: np.ndarray, method: NormalizationMethod) -> np.ndarray:
    """根据指定方法归一化states"""
    if method == NormalizationMethod.nmax:
        return states / np.amax(states)
    elif method == NormalizationMethod.n1to1:
        states_min, states_max = np.amin(states), np.amax(states)
        return 2 * (states - states_min) / (states_max - states_min) - 1
    elif method == NormalizationMethod.n0to1:
        states_min, states_max = np.amin(states), np.amax(states)
        return (states - states_min) / (states_max - states_min)
    else:  # NO_NORMALIZATION
        return states


def gen_input(Nreadouts, Nwash=10, seed=1234):
    """生成KR和GR的输入数据"""
    np.random.seed(seed)
    KR_inputs = np.random.ranf((Nreadouts, Nwash))
    GR_inputs = np.tile(np.random.ranf((10)), (Nreadouts, 1))
    return np.concatenate((KR_inputs, GR_inputs), axis=1)


def Evaluate_sKR_sGR(states):
    """计算KR和GR的奇异值"""
    GR_states = states[:, -1, :]
    KR_states = states[:, -11, :]
    _, sGR, _ = np.linalg.svd(GR_states)
    _, sKR, _ = np.linalg.svd(KR_states)
    return sKR, sGR

def RunSpnc(signal,Nin,Nout,Nvirt,m0,transform, params,**kwargs):
    '''
    Run a reservoir computer with the signal sequence
    '''
    snr = single_node_reservoir(Nin, Nout, Nvirt, m0, res=transform)

    fixed_mask = kwargs.get('fixed_mask', False)
    if fixed_mask==True:
        # print("Deterministic mask will be used")
        seed_mask = kwargs.get('seed_mask', 1234)
        if seed_mask>=0:
            # print(seed_mask)
            snr.M = fixed_seed_mask(Nin, Nvirt, m0, seed=seed_mask)
        else:
            # print("Max_sequences mask will be used")
            snr.M = max_sequences_mask(Nin, Nvirt, m0)
    
    # Run
    S,_ = snr.transform(signal,params)
    
    return S

def evaluate_KRandGR_returndata(reservoir_params, normalization_method: NormalizationMethod):
    """评估KR和GR，返回奇异值数据"""
    Nreadouts = reservoir_params.Nvirt
    inputs = gen_input(Nreadouts, Nwash=10, seed=1234)
    outputs = []
    
    for input_row in inputs:
        input_row = input_row.reshape(-1, 1)
        spn = spnc_anisotropy(reservoir_params.h, reservoir_params.theta_H,
                              reservoir_params.k_s_0, reservoir_params.phi,
                              reservoir_params.beta_prime, restart=True)
        transforms = spn.gen_signal_slow_delayed_feedback
        output = RunSpnc(input_row, 1, 1, reservoir_params.Nvirt,
                         reservoir_params.m0, transforms, reservoir_params.params, 
                         fixed_mask=True, seed_mask=1234)
        outputs.append(output)
    
    States = np.stack(outputs, axis=0)
    # 打印出当前的归一化方法
    print(f"Normalization method: {normalization_method}")
    Normalized_States = normalize_states(States, normalization_method)
    sKR, sGR = Evaluate_sKR_sGR(Normalized_States)
    
    return {'sKR': sKR, 'sGR': sGR}


@dataclass
class ParetoPointParams:
    """Container for Pareto point parameters."""
    trial_number: int
    gamma: float
    theta: float
    m0: float
    beta_prime: float
    cq_value: float
    mc_value: float



@dataclass
class TaskResults:
    """Stores KR/GR evaluation results for a trial with normalization method."""
    trial_number: int
    gamma: float
    theta: float
    m0: float
    beta_prime: float
    normalization_method: str
    sKR: np.ndarray
    sGR: np.ndarray


@dataclass
class ParameterSource:
    """Describes how Pareto point parameters are provided."""
    source_type: str
    data: str


class ParetoPointEvaluator:
    """Evaluates sKR/sGR metrics for Pareto points."""

    def __init__(self):
        self.manager = ReservoirMorphologyManager()

    def load_pareto_csv(self, filename: str) -> List[ParetoPointParams]:
        """Load Pareto points from a CSV file, searching by partial filename."""
        search_pattern = f"**/*{filename}*"
        matching_files = list(Path(".").glob(search_pattern))

        if not matching_files:
            raise FileNotFoundError(f"No file found containing '{filename}'")

        if len(matching_files) > 1:
            print(f"Multiple files matched: {matching_files}")
            print(f"Using first match: {matching_files[0]}")

        csv_path = matching_files[0]
        print(f"Reading file: {csv_path}")

        df = pd.read_csv(csv_path)

        pareto_points: List[ParetoPointParams] = []
        for _, row in df.iterrows():
            values_data = row["values"]
            if isinstance(values_data, str):
                values_data = ast.literal_eval(values_data)
            if isinstance(values_data, (list, tuple)):
                cq_value = float(values_data[0])
                mc_value = float(values_data[1])
            elif isinstance(values_data, (float, int)):
                cq_value = float(values_data)
                mc_value = float("nan")
            else:
                raise ValueError(f"Unsupported values column type: {type(values_data)}")

            pareto_points.append(
                ParetoPointParams(
                    trial_number=int(row["number"]),
                    gamma=float(row["param_gamma"]),
                    theta=float(row["param_theta"]),
                    m0=float(row["param_m0"]),
                    beta_prime=float(row["param_beta_prime"]),
                    cq_value=cq_value,
                    mc_value=mc_value,
                )
            )

        print(f"Loaded {len(pareto_points)} Pareto points")
        if pareto_points:
            n_show = min(2, len(pareto_points))
            print(f"Preview first {n_show} points:")
            for point in pareto_points[:n_show]:
                print(
                    f" trial_number={point.trial_number}, CQ={point.cq_value}, MC={point.mc_value}, "
                    f"gamma={point.gamma}, theta={point.theta}, m0={point.m0}, beta_prime={point.beta_prime}"
                )
        return pareto_points

    def load_parameters(self, source: Union[str, ParameterSource]) -> List[ParetoPointParams]:
        """Load Pareto point parameters from a CSV path or ParameterSource."""
        if isinstance(source, str):
            return self.load_pareto_csv(source)

        if isinstance(source, ParameterSource):
            if source.source_type == "csv":
                return self.load_pareto_csv(source.data)
            raise ValueError(f"Unsupported parameter source type: {source.source_type}")

        raise TypeError(f"Expected str or ParameterSource, got {type(source)}")

    def create_reservoir_params(self, pareto_point: ParetoPointParams) -> ReservoirParams:
        """Build a ReservoirParams instance from Pareto point data."""
        return ReservoirParams(
            h=0.4,
            m0=pareto_point.m0,
            Nvirt=200,
            beta_prime=pareto_point.beta_prime,
            params={"theta": pareto_point.theta, "gamma": pareto_point.gamma, "Nvirt": 200},
        )

    def _evaluate_single_point(self, pareto_point: ParetoPointParams, 
                              normalization_methods: List[NormalizationMethod]) -> List[TaskResults]:
        """Evaluate KR and GR for a single Pareto point with all normalization methods."""
        print(
            f"Evaluating trial {pareto_point.trial_number}: gamma={pareto_point.gamma}, "
            f"theta={pareto_point.theta}, m0={pareto_point.m0}, beta_prime={pareto_point.beta_prime}"
        )
        reservoir_params = self.create_reservoir_params(pareto_point)
        results = []

        for norm_method in normalization_methods:
            kr_gr_result = evaluate_KRandGR_returndata(reservoir_params, norm_method)
            results.append(TaskResults(
                trial_number=pareto_point.trial_number,
                gamma=pareto_point.gamma,
                theta=pareto_point.theta,
                m0=pareto_point.m0,
                beta_prime=pareto_point.beta_prime,
                normalization_method=norm_method.value,
                sKR=kr_gr_result["sKR"],
                sGR=kr_gr_result["sGR"],
            ))
        
        return results

    def evaluate_all_points(self, source: Union[str, ParameterSource], output_filename: str = None,
                          trial_numbers: Union[int, List[int]] = None,
                          normalization_methods: List[NormalizationMethod] = None) -> List[TaskResults]:
        """Evaluate KR/GR for all or subset of Pareto points with all normalization methods."""
        pareto_points = self.load_parameters(source)
        
        # 筛选trials
        target_trials = [trial_numbers] if isinstance(trial_numbers, int) else trial_numbers
        pareto_points = [p for p in pareto_points if p.trial_number in target_trials] if target_trials else pareto_points
        
        normalization_methods = normalization_methods or list(NormalizationMethod)
        print(f"Evaluating {len(pareto_points)} trials × {len(normalization_methods)} methods")

        # 评估所有点
        results = []
        for idx, point in enumerate(pareto_points, 1):
            print(f"Progress: {idx}/{len(pareto_points)} - Trial {point.trial_number}")
            results.extend(self._evaluate_single_point(point, normalization_methods))

        # 生成文件名
        base_name = Path(source if isinstance(source, str) else source.data).stem
        trial_suffix = f"_trials_{'_'.join(map(str, target_trials[:5]))}" if target_trials and len(target_trials) <= 5 else f"_{len(target_trials)}trials" if target_trials else ""
        output_filename = output_filename or f"ParetoFront_{base_name}{trial_suffix}"

        self.save_results(results, output_filename)
        return results

    def save_results(self, results: List[TaskResults], output_filename: str) -> None:
        """Persist evaluation outputs to pickle file."""
        output_dir = Path("ParetoFront_CQandMC")
        output_dir.mkdir(exist_ok=True)

        pickle_file = output_dir / f"{output_filename}_returnsKRsGR.pkl"
        with open(pickle_file, "wb") as handle:
            pickle.dump(results, handle)
        
        norm_counts = {}
        for result in results:
            norm_counts[result.normalization_method] = norm_counts.get(result.normalization_method, 0) + 1
        
        print(f"\n✓ Saved {len(results)} records to: {pickle_file}")
        for method, count in sorted(norm_counts.items()):
            print(f"  {method}: {count}")


def main(filename: str) -> None:
    """Entry point for standalone execution."""
    evaluator = ParetoPointEvaluator()
    results = evaluator.evaluate_all_points(source=filename)
    n_trials = len(results) // len(list(NormalizationMethod))
    n_methods = len(list(NormalizationMethod))
    print(f"\n✓ Complete: {n_trials} trials × {n_methods} methods = {len(results)} results")


if __name__ == "__main__":
    main("CQ_MC_Pareto_beta50_20250825_121711_dominatedpoints.csv")
