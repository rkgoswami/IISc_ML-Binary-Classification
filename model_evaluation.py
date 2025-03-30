from sklearn.model_selection import PredefinedSplit, GridSearchCV, KFold
from skopt import BayesSearchCV

from bayes_search import bayes_search
from grid_search import grid_search

""" 
    This is a utility function, 
    which perform both Grid Search and Bayesian Search 
    for Hyperparameter tuning
"""
def perform_search_using_kfold(model, param_grid, param_space, X_train, y_train):

    # Define K-Fold cross-validation on the training data (70%)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Grid Search
    grid_search_cv = grid_search(model, param_grid, X_train, y_train, kfold)

    # Bayesian Search
    bayes_search_cv = bayes_search(model, param_space, X_train, y_train, kfold)


    return grid_search_cv, bayes_search_cv

""" 
    This is a utility function, 
    which perform both Grid Search and Bayesian Search 
    for Hyperparameter tuning via Predefined Split, which
    ensures the validation data is reserved for this purpose, 
    while the training data (70%) is used for fitting
"""
def perform_search_using_predefined_split(model, param_grid, param_space, X_train_val, y_train_val, split_idx):

    # Define predefined split for cross-validation
    ps = PredefinedSplit(test_fold=split_idx)  # Keep this

    # Grid Search
    grid_search_cv = grid_search(model, param_grid, X_train_val, y_train_val, ps)

    # Bayesian Search
    bayes_search_cv = bayes_search(model, param_space, X_train_val, y_train_val, ps)

    return grid_search_cv, bayes_search_cv


""" 
    This is a utility function which give score of the model
"""
def evaluate_model(model, X_test, y_test):
    return model.score(X_test, y_test)
