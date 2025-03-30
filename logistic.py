import pandas as pd
from sklearn.linear_model import LogisticRegression
from skopt.space import Real, Categorical

from data_loader import load_and_preprocess_data
from model_evaluation import perform_search_using_kfold, evaluate_model
from print_console_report import console_report


def logistic_regression(X_train=None, X_val=None, X_test=None,
                        y_train=None, y_val=None, y_test=None):

    # Load the preprocessed data, if not passed
    if X_train is None:
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data()

    # Combine train + validation for hyperparameter tuning
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    # Initialize model
    model = LogisticRegression(max_iter=1000)

    # Define the hyperparameter for grid and bayesian search
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    param_space = {
        'C': Real(0.01, 100, prior='log-uniform'),
        'penalty': Categorical(['l1', 'l2']),
        'solver': Categorical(['liblinear'])
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
        "model_name": "Logistic Regression",
        "grid_best_score": grid_search.best_score_,
        "grid_best_params": grid_search.best_params_,
        "bayesian_best_score": bayes_search.best_score_,
        "bayesian_best_params": bayes_search.best_params_,
    }

    best_model_result = {
        "model_name": "Logistic Regression",
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "best_search_type": best_search_type,
        "best_params": best_model.get_params()
    }

    return grid_vs_bayes, best_model_result


if __name__ == '__main__':
    vs, result = logistic_regression()
    print("\nComparison Result:")
    console_report([vs])
    print("\nBest Model Result:")
    console_report([result])


