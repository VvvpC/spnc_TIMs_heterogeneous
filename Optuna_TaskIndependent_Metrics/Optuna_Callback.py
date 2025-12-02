import optuna

'''
this function is used to control the termination of the study under certain conditions.

'''

def set_callback(target):

    def callback(study, trial):
        # Access the best trial's values for each objective
        best_performance = study.best_value

        # Stop the study if both objectives meet their respective targets
        if best_performance <= target:
            study.stop()
            
    return callback