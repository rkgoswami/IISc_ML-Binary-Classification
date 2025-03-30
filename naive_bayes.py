import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.naive_bayes import GaussianNB
from skopt.space import Real

from data_loader import load_and_preprocess_data
from model_evaluation import perform_search_using_kfold, evaluate_model
from print_console_report import console_report

# Suppress skopt duplicate evaluation warning
warnings.filterwarnings("ignore", category=UserWarning, module="skopt.optimizer.optimizer")
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def naive_bayes(X_train=None, X_val=None, X_test=None,
                y_train=None, y_val=None, y_test=None):

    # Load the preprocessed data, if not passed
    if X_train is None:
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data()

    # Combine train + validation for hyperparameter tuning
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    # split_idx = [-1]*len(X_train) + [0]*len(X_val) # used kfold based on suggestion by professor


    # Initialize model
    model = GaussianNB()

    # Define the hyperparameter for grid and bayesian search
    param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7]
    }
    param_space = {
        'var_smoothing': Real(1e-9, 1e-7, 'log-uniform')
    }

    # Perform hyperparameter tuning on train data (70%)
    grid_search, bayes_search = perform_search_using_kfold(
        model, param_grid, param_space, X_train_val, y_train_val
    )

    # Select the best model (Bayesian or Grid Search)
    if bayes_search.best_score_ > grid_search.best_score_:
        best_model = bayes_search.best_estimator_
        best_search_type = "Bayesian"
    else:
        best_model = grid_search.best_estimator_
        best_search_type = "Grid"

    # Validate the best model on the 10% validation set
    val_accuracy = evaluate_model(best_model, X_val, y_val)

    # Retrain the best model on Train+Val (80%) and test on 20%
    best_model.fit(X_train_val, y_train_val)
    test_accuracy = evaluate_model(best_model, X_test, y_test)

    grid_vs_bayes = {
        "model_name": "Naive Bayes",
        "grid_best_score": grid_search.best_score_,
        "grid_best_params": grid_search.best_params_,
        "bayesian_best_score": bayes_search.best_score_,
        "bayesian_best_params": bayes_search.best_params_,
    }

    best_model_result =  {
        "model_name": "Naive Bayes",
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "best_params": best_model.get_params(),
        "best_search_type": best_search_type
    }
    return grid_vs_bayes, best_model_result


if __name__ == '__main__':
    vs, result = naive_bayes()
    print("\nComparison Result:")
    console_report([vs])
    print("\nBest Model Result:")
    console_report([result])
