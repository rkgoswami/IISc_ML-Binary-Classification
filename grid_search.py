from sklearn.model_selection import GridSearchCV


def grid_search(model, param_grid, X_train_val, y_train_val, cv=1):
    # Grid Search
    grid_search_cv = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring='accuracy',     # scoring strategy
        n_jobs=-1               # use all processors in parallel
    )
    grid_search_cv.fit(X_train_val, y_train_val)

    return grid_search_cv