
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from contextlib import suppress
import optuna
from optuna.samplers import GPSampler
from Optuna_Dashboard import run_dashboard

# Import the metric/evaluation toolkit
from formal_Parameter_Dynamics_Preformance import (
    ReservoirParams,
    evaluate_MC,
    evaluate_KRandGR,
)
from ParetoFront_CQandMC.CQ_MC_ParetofrontPoints import eva_narma10, eva_ti46

# ──────────────────────────────────────────────────────────────────────────────
# 1. Search‑space definition
# ──────────────────────────────────────────────────────────────────────────────
HYPERSPACE = {
    "gamma": (0, 0.5),       
    "theta": (0.01, 0.6),       
    "m0": (0.001, 0.008),                   
    "beta_prime": (20, 50),              
}

# ──────────────────────────────────────────────────────────────────────────────
# 2. Objective function – returns (CQ, MC)
# ──────────────────────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial):
    """Single Optuna trial evaluating CQ and MC."""
    # Sample hyper‑parameters
    gamma = trial.suggest_float("gamma", *HYPERSPACE["gamma"])
    theta = trial.suggest_float("theta", *HYPERSPACE["theta"])
    m0 = trial.suggest_float("m0", *HYPERSPACE["m0"])
    beta_prime = trial.suggest_float("beta_prime", *HYPERSPACE["beta_prime"])

    # Build a ReservoirParams instance with the sampled values
    rparams = ReservoirParams(
        h=0.4,
        m0=m0,
        Nvirt=200,
        beta_prime=beta_prime,
        params={
            "gamma": gamma,
            "theta": theta,
            "Nvirt": 200,
        },
    )

    # Evaluate task
    narma10_dict = eva_narma10(rparams)
    ti46_dict = eva_ti46(rparams)
    error_rate = (1-ti46_dict['accuracy'])*100
      # Nwash = 7, Nequal = 7
    narma10_nrmse = float(narma10_dict.get("NRMSE", 0.0))
    ti46_error_rate = float(error_rate)



    # Early pruning if hopeless
    if narma10_nrmse < 0 or ti46_error_rate < 0:
        raise optuna.exceptions.TrialPruned()

    # Optuna will *maximise* both outputs
    return narma10_nrmse, ti46_error_rate

# ──────────────────────────────────────────────────────────────────────────────
# 3. Study setup
# ──────────────────────────────────────────────────────────────────────────────



def create_study():
    # Create a study, and add a suffix to the study if the study name already exists
    suffix = 0

    # set up the storage and study name
    storage = "sqlite:///db.sqlite3" 
    study_name = "NARMA10_TI46_Pareto"
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
            print(f"Study '{new_study_name}' doesn't exist, Create it。")
            break
    
    sampler = GPSampler()


    # set up the object of the study
    study = optuna.create_study(
        # set the samplers
        sampler=sampler,
        # set the direction of the objectives
        directions=["minimize", "minimize"],  
        storage=storage,
        study_name=new_study_name,
    )
    
    return study


# ──────────────────────────────────────────────────────────────────────────────
# 4. Run the study
# ──────────────────────────────────────────────────────────────────────────────

def run_study():
    
    study = create_study()

    study.optimize(
        objective,
        n_trials=400,  # Number of trials from command line argument
        catch=(ValueError, FloatingPointError),
    )

    print("\nPareto front (all non-dominated trials):")
    for t in study.best_trials:
        print('  Values: ', t.values)
        print('  Params:')
        for key, value in t.params.items():
            print(f'    {key}: {value}')
        print('-------------------')


if __name__ == "__main__":
    run_study()