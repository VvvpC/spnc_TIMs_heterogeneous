"""
Optuna-based hyperparameter search for the TI46 spoken digit recognition task.
--------------------------------------------------------------------------------
This script implements a systematic hyperparameter search using Optuna to find
the optimal configuration for a superparamagnetic nanodot reservoir on the
TI46 task. It is based on the logic provided for building a task-specific
Optuna search script.

Key Components:
1.  **Search Space (HYPERSPACE_TI46)**: Defines the search range for each
    hyperparameter, including 'gamma', 'theta', 'm0', 'h', 'beta_prime',
    and 'Nvirt'.
2.  **Objective Function (objective_TI46)**: For each trial, this function:
    - Samples a set of hyperparameters.
    - Instantiates a `ReservoirTaskParams` object.
    - Calls the `evaluate_size_TI46` function to compute the recognition
      accuracy.
    - Returns the accuracy, which Optuna aims to maximize.
    - Implements early trial pruning to save computational resources on
      unpromising parameter sets.
3.  **Study Execution**: The main part of the script sets up and runs the
    Optuna study, saving progress to a SQLite database. It allows resuming
    from a previous state.
4.  **Results & Monitoring**: After the search, it prints the best trial found.
    It also provides the command to launch the Optuna dashboard for visual
    analysis of the results.

How to Run:
-----------
Execute the script from your terminal. You can customize the search process
using command-line arguments.

# Run with default settings (100 trials)
python Optuna_TaskIndependent_Metrics/Optuna_TI46_Hyperparameter_Search.py

# Run for 500 trials with a 1-hour timeout
python Optuna_TaskIndependent_Metrics/Optuna_TI46_Hyperparameter_Search.py --trials 500 --timeout 3600

# Run and start the dashboard afterwards
python Optuna_TaskIndependent_Metrics/Optuna_TI46_Hyperparameter_Search.py --trials 200 --dashboard
"""
import os
import sys
import argparse
import time

# Add project root to path to allow importing custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import optuna
import optunahub


from Reservoirs_diameter_Tasks import ReservoirTaskParams, evaluate_size_TI46

# 1. 核心架构设计: 超参数空间定义
HYPERSPACE_TI46 = {
    "gamma": (0, 0.5),
    "theta": (0.01, 10),
    "m0": (0.001, 0.5),
    "h": (0.3, 0.5),
    "beta_prime": (20,50),
    
}

# 2. 目标函数设计
def objective_TI46(trial: optuna.Trial) -> float:
    """
    Optuna objective function for TI46 task.
    - Samples hyperparameters from HYPERSPACE_TI46.
    - Instantiates ReservoirTaskParams.
    - Calls evaluate_size_TI46 to get accuracy.
    - Implements early pruning for low-accuracy trials.
    - Returns the accuracy for single-objective maximization.
    """
    # 2.1 采样超参数
    gamma = trial.suggest_float("gamma", *HYPERSPACE_TI46["gamma"])
    theta = trial.suggest_float("theta", *HYPERSPACE_TI46["theta"])
    m0 = trial.suggest_float("m0", *HYPERSPACE_TI46["m0"])
    h = trial.suggest_float("h", *HYPERSPACE_TI46["h"])
    beta_prime = trial.suggest_float("beta_prime", *HYPERSPACE_TI46["beta_prime"])


    # 3.1 数据流处理: 构建 ReservoirTaskParams 实例
    rparams = ReservoirTaskParams(
        h=h,
        m0=m0,
        Nvirt=50,
        beta_prime=beta_prime,
        ref_beta_prime=beta_prime,  # 单储层调查：使用当前beta_prime作为参考值
        speakers=None,  # None means use all speakers for the TI46 task
        params={
            "gamma": gamma,
            "theta": theta,
            "Nvirt": 50,
        },
    )

    # 2.3 评估函数集成: 调用 evaluate_size_TI46
    # The evaluation function can raise exceptions (e.g., from linear algebra).
    # We catch them to avoid crashing the whole study.
    try:
        ti46_result = evaluate_size_TI46(rparams, verbose=False)
        accuracy = float(ti46_result.get("Accuracy", 0.0))
    except Exception as e:
        print(f"Trial {trial.number} failed with exception: {e}")
        # Prune trial if evaluation fails
        raise optuna.exceptions.TrialPruned()

    # 3.2 性能优化: 早期剪枝
    # This implements both "early pruning" and "constrained optimization"
    MIN_ACCURACY_THRESHOLD = 0.1  # Prune if accuracy is below 10%
    if accuracy < MIN_ACCURACY_THRESHOLD:
        raise optuna.exceptions.TrialPruned()

    return accuracy

# 3. 结果管理与优化策略
def create_study(study_name: str):
    """
    Creates the Optuna study with settings matching Optuna_CQ_MC_Pareto.
    - Handles study naming with automatic suffix for existing studies.
    - Uses optunahub AutoSampler.
    - Maximizes a single objective: TI46 accuracy.
    - Persists results to a SQLite database.
    """
    # Create a study, and add a suffix to the study if the study name already exists
    suffix = 0
    
    # set up the storage and study name
    storage = "sqlite:///db.sqlite3"
    new_study_name = study_name
    
    while True:
        try:
            # try to load the study
            optuna.load_study(study_name=new_study_name, storage=storage)
            # if the study exists, add a suffix to the study name
            suffix += 1
            new_study_name = f"{study_name}_{suffix}"
        except KeyError:
            # if the study does not exist, break the loop
            print(f"Study '{new_study_name}' doesn't exist, Create it.")
            break
    
    module = optunahub.load_module(package="samplers/auto_sampler")
    
    # set up the object of the study
    study = optuna.create_study(
        # set the samplers
        sampler=module.AutoSampler(),
        # set the direction of the objectives
        direction="maximize",  # Maximize TI46 Accuracy
        storage=storage,
        study_name=new_study_name,
    )
    
    # Add system attributes to the study for better tracking
    study.set_system_attr("hyperspace", HYPERSPACE_TI46)
    
    return study

def run_study(n_trials: int, timeout: int, study_name: str):
    """
    Creates and runs the Optuna study.
    """
    study = create_study(study_name)
    
    print(f"Starting study '{study.study_name}'. Running for {n_trials} trials with a timeout of {timeout} seconds.")
    
    study.optimize(
        objective_TI46,
        n_trials=n_trials,
        timeout=timeout,
        # n_jobs=-1, # Uncomment to enable parallel execution using all CPU cores
        catch=(ValueError, FloatingPointError), # Catch numerical errors during evaluation
    )

    print("\n" + "="*50)
    print("STUDY COMPLETED")
    print(f"Study name: {study.study_name}")
    print(f"Number of finished trials: {len(study.trials)}")

    print("\nBest trial:")
    best_trial = study.best_trial
    print(f"  Value (Accuracy): {best_trial.value:.4f}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    print("="*50 + "\n")
    
    return study

def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for TI46 task.")
    parser.add_argument("--trials", type=int, default=400, help="Number of trials to run.")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout for the study in seconds.")
    parser.add_argument("--study-name", type=str, default="TI46_hyper_search", help="Name for the Optuna study.")
    parser.add_argument("--dashboard", action="store_true", help="Show command to run Optuna dashboard after the study.")
    
    args = parser.parse_args()
    
    start_time = time.time()
    run_study(
        n_trials=args.trials,
        timeout=args.timeout,
        study_name=args.study_name,
    )
    end_time = time.time()
    
    print(f"Total optimization time: {end_time - start_time:.2f} seconds.")

    if args.dashboard:
        print("\nTo view the Optuna dashboard, run the following command in your terminal:")
        print("optuna-dashboard sqlite:///db.sqlite3")


if __name__ == "__main__":
    main()