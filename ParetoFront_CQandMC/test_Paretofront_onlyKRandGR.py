"""
这个test的目的是从pareto前沿文件中提取参数,创建储层,遍历所有储层仅计算KR和GR
"""




import ast
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

# Add parent directory to Python path so shared modules are importable.
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from spnc import spnc_anisotropy  # noqa: F401, imported for side effects
from formal_Parameter_Dynamics_Preformance import ReservoirParams, evaluate_KRandGR
from Morphology_Research.Reservoirs_morphology_creator import ReservoirMorphologyManager


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
    """Stores KR/GR evaluation results for a trial."""

    trial_number: int
    gamma: float
    theta: float
    m0: float
    beta_prime: float
    kr_threshold: float
    gr_threshold: float


@dataclass
class ParameterSource:
    """Describes how Pareto point parameters are provided."""

    source_type: str  # currently only "csv"
    data: str  # CSV file name or path fragment


class ParetoPointEvaluator:
    """Evaluates KR/GR metrics for Pareto points."""

    def __init__(self):
        # Hold the manager reference; downstream evaluation code relies on it.
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

    def _evaluate_single_point(
        self,
        pareto_point: ParetoPointParams,
        kr_gr_config: Dict[str, float] = None,
    ) -> TaskResults:
        """Evaluate KR and GR thresholds for a single Pareto point."""
        config = {"threshold": 0.001}
        if kr_gr_config:
            config.update(kr_gr_config)

        print(
            f"Evaluating trial {pareto_point.trial_number}: gamma={pareto_point.gamma}, "
            f"theta={pareto_point.theta}, m0={pareto_point.m0}, beta_prime={pareto_point.beta_prime}"
        )

        reservoir_params = self.create_reservoir_params(pareto_point)

        kr_gr_result = evaluate_KRandGR(reservoir_params, **config)
        kr = kr_gr_result.get("KR")
        gr = kr_gr_result.get("GR")
        print(f"  KR: {kr}, GR: {gr}")

        return TaskResults(
            trial_number=pareto_point.trial_number,
            gamma=pareto_point.gamma,
            theta=pareto_point.theta,
            m0=pareto_point.m0,
            beta_prime=pareto_point.beta_prime,
            kr_threshold=kr,
            gr_threshold=gr,
        )

    def evaluate_all_points(
        self,
        source: Union[str, ParameterSource],
        output_filename: str = None,
        trial_numbers: Union[int, List[int]] = None,
        kr_gr_config: Dict[str, float] = None,
    ) -> List[TaskResults]:
        """Evaluate KR/GR metrics for all or a subset of Pareto points."""
        pareto_points = self.load_parameters(source)

        if trial_numbers is not None:
            target_trials = [trial_numbers] if isinstance(trial_numbers, int) else list(trial_numbers)
            filtered_points = [point for point in pareto_points if point.trial_number in target_trials]

            found_trials = [point.trial_number for point in filtered_points]
            missing_trials = [trial for trial in target_trials if trial not in found_trials]
            if missing_trials:
                print(f"Warning: trials not found {missing_trials}")

            if not filtered_points:
                print(f"No matching trials found for {target_trials}")
                return []

            pareto_points = filtered_points
            print(f"Evaluating selected trials: {found_trials}")
        else:
            print(f"Evaluating all {len(pareto_points)} trials")

        results: List[TaskResults] = []
        for idx, point in enumerate(pareto_points, start=1):
            print(f"\nProgress: {idx}/{len(pareto_points)} (Trial {point.trial_number})")
            try:
                results.append(self._evaluate_single_point(point, kr_gr_config))
            except Exception as exc:
                print(f"  Error evaluating trial {point.trial_number}: {exc}")

        if output_filename is None:
            if isinstance(source, str):
                base_name = Path(source).stem
            elif isinstance(source, ParameterSource) and source.source_type == "csv":
                base_name = Path(source.data).stem
            else:
                base_name = "unknown_source"

            if trial_numbers is not None:
                trial_list = [trial_numbers] if isinstance(trial_numbers, int) else list(trial_numbers)
                trials_str = "_".join(map(str, trial_list)) if len(trial_list) <= 5 else f"{len(trial_list)}trials"
                output_filename = f"ParetoFront_TaskResults_{base_name}_trials_{trials_str}"
            else:
                output_filename = f"ParetoFront_TaskResults_{base_name}"

        self.save_results(results, output_filename)
        return results

    def save_results(self, results: List[TaskResults], output_filename: str) -> None:
        """Persist evaluation outputs to pickle and CSV files."""
        output_dir = Path("ParetoFront_CQandMC")
        output_dir.mkdir(exist_ok=True)

        pickle_file = output_dir / f"{output_filename}_max.pkl"
        with open(pickle_file, "wb") as handle:
            pickle.dump(results, handle)
        print(f"Saved full results to: {pickle_file}")

        csv_file = output_dir / f"{output_filename}_max.csv"
        summary_rows = [
            {
                "trial_number": result.trial_number,
                "gamma": result.gamma,
                "theta": result.theta,
                "m0": result.m0,
                "beta_prime": result.beta_prime,
                "kr_threshold": result.kr_threshold,
                "gr_threshold": result.gr_threshold,
            }
            for result in results
        ]

        df = pd.DataFrame(summary_rows)
        df.to_csv(csv_file, index=False)
        print(f"Saved summary results to: {csv_file}")


def main(filename: str, kr_threshold: float = 0.001) -> None:
    """Entry point for standalone execution."""
    evaluator = ParetoPointEvaluator()
    try:
        results = evaluator.evaluate_all_points(
            source=filename,
            kr_gr_config={"threshold": kr_threshold},
            output_filename=filename[:-4] if filename.lower().endswith(".csv") else filename,
        )
        print(f"KR/GR evaluation complete: {len(results)} Pareto points processed")
    except FileNotFoundError:
        print("CSV file not found; skipping evaluation")


if __name__ == "__main__":
    main("CQ_MC_Pareto_beta50_20250825_121711_pareto.csv")
