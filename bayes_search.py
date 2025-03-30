from skopt import BayesSearchCV

def bayes_search(model, param_space, X_train_val, y_train_val, cv=1):
    bayes_search_cv = BayesSearchCV(
        model,
        param_space,
        cv=cv,
        n_iter=50,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )
    bayes_search_cv.fit(X_train_val, y_train_val)

    return bayes_search_cv