"""

24/06/25 Chen
Optuna-based multi_objective optimisation for a superparamagnetic nanodot reservoir
-------------------------------------------------------------------------------
This script searches for the Pareto front of

    • CQ  = KR - GR   (capacity for kernel–rank minus generalisation‑rank)
    • MC               (linear memory capacity)

using the Optuna NSGA_II sampler.

It combines the original Optuna scaffolding (Optuna_* files) with the task-
independent metric utilities in *Parameter_Dynamics_Preformance.py*.

How it works
============
1.  **Search space** - The dictionary `HYPERSPACE` defines the bounds for each
    tunable reservoir hyperparameter.
2.  **Objective** - For every trial we
        • instantiate a fresh `ReservoirParams` object with the sampled
          hyperparameters;
        • call `evaluate_MC` **once** and `evaluate_KRandGR` **once** (both are
          imported from *Parameter_Dynamics_Preformance.py*);
        • compute `CQ = KR - GR` and return the pair `(CQ,MC)`.
    Trials that clearly under perform (e.g.`CQ<0` or `MC<0.5`) are pruned
    early to save compute.
3.  **Study** - `optuna.create_study(directions=["maximize", "maximize"])` is
    used so that Optuna treats both outputs as objectives to **maximise**.  The
    `NSGAIISampler` is ideal for Pareto optimisation.
4.  **Results** - After optimisation the script prints the nondominated set
    (`study.best_trials`) and optionally persists the study to
    `sqlite:///cq_mc.db`.

Run it via:

bash
python optuna_cq_mc_pareto.py --trials 200


Adjust the `--trials` flag or the `timeout` parameter as needed.
"""
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

    # Evaluate task‑independent metrics
    mc_dict = evaluate_MC(rparams)            
    kgr_dict = evaluate_KRandGR(rparams)      # Nwash = 7, Nequal = 7

    MC = float(mc_dict.get("MC", 0.0))
    KR = float(kgr_dict.get("KR", 0.0))
    GR = float(kgr_dict.get("GR", 0.0))
    CQ = KR - GR

    # Early pruning if hopeless
    if MC < 0 or CQ < 0:
        raise optuna.exceptions.TrialPruned()

    # Optuna will *maximise* both outputs
    return CQ, MC

# ──────────────────────────────────────────────────────────────────────────────
# 3. Study setup
# ──────────────────────────────────────────────────────────────────────────────



def create_study():
    # Create a study, and add a suffix to the study if the study name already exists
    suffix = 0

    # set up the storage and study name
    storage = "sqlite:///db.sqlite3" 
    study_name = "CQ_MC_Pareto"
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
        directions=["maximize", "maximize"],  
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