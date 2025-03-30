import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC
from skopt.space import Real

from data_loader import load_and_preprocess_data
from model_evaluation import evaluate_model, perform_search_using_kfold
from print_console_report import console_report

# Suppress skopt duplicate evaluation warning
warnings.filterwarnings("ignore", category=UserWarning, module="skopt.optimizer.optimizer")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def svm_linear(X_train=None, X_val=None, X_test=None,
               y_train=None, y_val=None, y_test=None):

    # Load the preprocessed data, if not passed
    if X_train is None:
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data()

    # Combine train + validation for hyperparameter tuning
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    #split_idx = [-1]*len(X_train) + [0]*len(X_val)

    """Since dataset is small we prefer using SVC else we can use LinearSVC or SGDClassifier"""
    # Initialize the model with linear kernel
    model = SVC(kernel='linear', random_state=42)

    # Define the hyperparameter for grid and bayesian search
    """
        Available hyperparameter for SVC:
            C: default=1.0              # Regularization parameter. The strength of the regularization is inversely proportional to C.
            kernel: default='rbf'       # Options: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, 
            degree: default=3           # Degree of the polynomial kernel function (‘poly’). Must be non-negative
            gamma: default='scale'      # Options: ‘scale’, ‘auto’
            tol: default=1e-3           # Tolerance for stopping criterion.
            class_weight: default=None  # dict or ‘balanced’
        
        We will only be using 'C' for hyperparameter tuning as its most impactful for controlling model complexity
    """
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100]
    }
    param_space = {
        'C': Real(0.01, 100, prior='log-uniform')
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

    # Retrain the best model on Train + Val (80%) and test on 20%
    best_model.fit(X_train_val, y_train_val)
    test_accuracy = evaluate_model(best_model, X_test, y_test)

    grid_vs_bayes = {
        "model_name": "Linear Kernel SVM",
        "grid_best_score": grid_search.best_score_,
        "grid_best_params": grid_search.best_params_,
        "bayesian_best_score": bayes_search.best_score_,
        "bayesian_best_params": bayes_search.best_params_,
    }

    best_model_result = {
        "model_name": "Linear Kernel SVM",
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "best_params": best_model.get_params(),
        "best_search_type": best_search_type
    }

    return grid_vs_bayes, best_model_result


if __name__ == '__main__':
    vs, result = svm_linear()
    print("\nComparison Result:")
    console_report([vs])
    print("\nBest Model Result:")
    console_report([result])