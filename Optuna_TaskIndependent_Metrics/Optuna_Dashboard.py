import subprocess

def run_dashboard():
    """run the optuna dashboard"""
    subprocess.run(["optuna-dashboard", "sqlite:///db.sqlite3"])

'''
optuna-dashboard sqlite:///db.sqlite3
'''