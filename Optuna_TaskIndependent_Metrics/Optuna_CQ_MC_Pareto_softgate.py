"""

24/06/25 Chen - Modified with NARMA10 soft gating
Optuna-based multi_objective optimisation for a superparamagnetic nanodot reservoir
-------------------------------------------------------------------------------
This script searches for the Pareto front of

    • CQ  = KR - GR   (capacity for kernel–rank minus generalisation‑rank)
    • MC               (linear memory capacity)

using the Optuna GPSampler with NARMA10 soft gating constraint.

It combines the original Optuna scaffolding (Optuna_* files) with the task-
independent metric utilities in *Parameter_Dynamics_Preformance.py*.

How it works
============
1.  **Search space** - The dictionary `HYPERSPACE` defines the bounds for each
    tunable reservoir hyperparameter.
2.  **Objective** - For every trial we
        • instantiate a fresh `ReservoirParams` object with the sampled
          hyperparameters;
        • first evaluate NARMA10 performance as a gating constraint;
        • if NARMA10 NRMSE exceeds threshold, apply penalty values;
        • otherwise call `evaluate_MC` and `evaluate_KRandGR` normally;
        • compute `CQ = KR - GR` and return the pair `(CQ,MC)`.
    This soft gating ensures only viable parameter combinations contribute
    to the Pareto front while maintaining GP model continuity.
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
    evaluate_NARMA10,
)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Search‑space definition and NARMA10 threshold
# ──────────────────────────────────────────────────────────────────────────────
HYPERSPACE = {
    "gamma": (0, 0.5),       
    "theta": (0.01, 0.6),       
    "m0": (0.001, 0.008),                   
    "beta_prime": (20, 50),              
}

# NARMA10 soft gating threshold
NARMA10_THRESHOLD = 0.8  # NRMSE threshold - parameters with NRMSE > threshold get penalty
PENALTY_CQ = -5.0      # Penalty value for CQ when NARMA10 threshold exceeded
PENALTY_MC = -5.0      # Penalty value for MC when NARMA10 threshold exceeded

# ──────────────────────────────────────────────────────────────────────────────
# 2. Objective function – returns (CQ, MC)
# ──────────────────────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial):
    """Single Optuna trial evaluating CQ and MC with NARMA10 soft gating."""
    # Sample hyper‑parameters
    gamma = trial.suggest_float("gamma", *HYPERSPACE["gamma"])
    theta = trial.suggest_float("theta", *HYPERSPACE["theta"])
    m0 = trial.suggest_float("m0", *HYPERSPACE["m0"])
    beta_prime = trial.suggest_float("beta_prime", *HYPERSPACE["beta_prime"])

    # Build a ReservoirParams instance with normal Nvirt for final metrics
    rparams = ReservoirParams(
        h=0.4,
        m0=m0,
        Nvirt=200,  # Full size for accurate MC/CQ computation
        beta_prime=beta_prime,
        params={
            "gamma": gamma,
            "theta": theta,
            "Nvirt": 200,
        },
    )
    
    # Build a lightweight copy with smaller Nvirt for fast NARMA10 screening
    rparams_softgate = ReservoirParams(
        h=0.4,
        m0=m0,
        Nvirt=50,   # Much smaller for fast NARMA10 evaluation
        beta_prime=beta_prime,
        params={
            "gamma": gamma,
            "theta": theta,
            "Nvirt": 50,
        },
    )

    # Step 1: Fast NARMA10 screening with lightweight parameters
    try:
        narma10_dict = evaluate_NARMA10(rparams_softgate, Ntrain=500, Ntest=200,)
        narma10_nrmse = float(narma10_dict.get("NRMSE", float('inf')))
        
        # Store NARMA10 result in trial for analysis
        trial.set_user_attr("narma10_nrmse", narma10_nrmse)
        
    except Exception as e:
        # If NARMA10 evaluation fails, apply maximum penalty
        trial.set_user_attr("narma10_nrmse", float('inf'))
        trial.set_user_attr("narma10_failed", True)
        return PENALTY_CQ, PENALTY_MC

    # Step 2: Soft gating - if NARMA10 performance is poor, apply penalties
    if narma10_nrmse > NARMA10_THRESHOLD:
        trial.set_user_attr("gating_status", "penalized")
        return PENALTY_CQ, PENALTY_MC
    
    # Step 3: If NARMA10 passes threshold, evaluate normal metrics
    trial.set_user_attr("gating_status", "passed")
    
    try:
        # Evaluate task‑independent metrics
        mc_dict = evaluate_MC(rparams)            
        kgr_dict = evaluate_KRandGR(rparams)      # Nwash = 7, Nequal = 7

        MC = float(mc_dict.get("MC", 0.0))
        KR = float(kgr_dict.get("KR", 0.0))
        GR = float(kgr_dict.get("GR", 0.0))
        CQ = KR - GR
        
        # Store detailed results in trial
        trial.set_user_attr("MC", MC)
        trial.set_user_attr("KR", KR)
        trial.set_user_attr("GR", GR)
        trial.set_user_attr("CQ", CQ)

        # Additional soft constraint for clearly poor performers
        if MC < 0 or CQ < 0:
            trial.set_user_attr("additional_penalty", True)
            raise optuna.exceptions.TrialPruned()

        # Optuna will *maximise* both outputs
        return CQ, MC
        
    except Exception as e:
        # If MC/KR/GR evaluation fails, apply penalty but less severe than NARMA10 failure
        trial.set_user_attr("mc_krgr_failed", True)
        return PENALTY_CQ * 0.5, PENALTY_MC * 0.5

# ──────────────────────────────────────────────────────────────────────────────
# 3. Study setup
# ──────────────────────────────────────────────────────────────────────────────

def create_study():
    # Create a study, and add a suffix to the study if the study name already exists
    suffix = 0

    # set up the storage and study name
    storage = "sqlite:///db.sqlite3" 
    study_name = "CQ_MC_Pareto_SoftGate"
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