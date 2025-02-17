# ------------------------------ #
#                                #
#  version 0.0.2                 #
#                                #
#  Aleksiej Ostrowski, 2022      #
#                                #
#  https://aleksiej.com          #
#                                #
# ------------------------------ #

"""

Based on script by Sylwia Mielnicka
https://github.com/SylwiaOliwia2/xgboost-AutoTune

"""

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import random
import math


def split_dict2(d, n_chunks, seed=42):
    random.seed(seed)
    keys = list(d.keys())
    random.shuffle(keys)
    for i in range(0, len(keys), n_chunks):
        yield {k: d[k] for k in keys[i : i + n_chunks]}


def fit_parameters(model, X_train, y_train, scoring, n_folds=5, seed=42):

    params = set(model.get_params().keys())

    regressor = model.__class__.__name__
    greater_is_better_ = (
        scoring.__dict__["_sign"] == 1
    )  # for score where greater is better

    if regressor not in ["XGBRegressor", "GradientBoostingRegressor", "LGBMRegressor"]:
        print(f"Maybe your {regressor} doesn't support")

    """

    XGBRegressor:

    'objective', 'base_score', 'booster', 'colsample_bylevel',
    'colsample_bynode', 'colsample_bytree', 'gamma', 'gpu_id', 'importance_type',
    'interaction_constraints', 'learning_rate', 'max_delta_step', 'max_depth',
    'min_child_weight', 'missing', 'monotone_constraints', 'n_estimators',
    'n_jobs', 'num_parallel_tree', 'random_state', 'reg_alpha', 'reg_lambda',
    'scale_pos_weight', 'subsample', 'tree_method', 'validate_parameters',
    'verbosity'

    GradientBoostingRegressor:

    'max_features', 'alpha', 'loss', 'n_estimators', 'min_samples_split', 'init', 
    'max_leaf_nodes', 'criterion', 'n_iter_no_change', 'validation_fraction',
    'min_impurity_decrease', 'warm_start', 'tol', 'subsample', 'max_depth',
    'learning_rate', 'min_weight_fraction_leaf', 'random_state', 'min_samples_leaf',
    'ccp_alpha', 'min_impurity_split', 'verbose'

    LGBMRegressor:

    'reg_alpha', 'subsample', 'num_leaves', 'silent', 'min_child_samples',
    'objective', 'n_estimators', 'min_split_gain', 'reg_lambda', 'importance_type',
    'min_child_weight', 'n_jobs', 'subsample_freq', 'class_weight', 'learning_rate',
    'max_depth', 'colsample_bytree', 'subsample_for_bin', 'boosting_type',
    'random_state'

    """

    domain_params_dicts = {
        "n_estimators": [100, 150, 200, 300],
        "learning_rate": [0.0001, 0.001, 0.005, 0.01, 0.04, 0.07, 0.1, 0.2, 0.3, 0.7],
        "max_depth": [3, 5, 7, 9],
        "min_child_weight": [0.001, 0.1, 1, 5, 10, 20],
        "gamma": [0.0, 0.5, 1.0, 3.0, 5.0, 10.0, 20.0, 40.0],
        "colsample_bynode": [0.0, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0],
        "min_weight_fraction_leaf": [0.0, 0.01, 0.1, 0.3, 0.5],
        "min_impurity_decrease": [0, 10, 50, 100, 300, 1000],
        "min_child_samples": [5, 10, 20, 50, 100, 150],
        "min_split_gain": [0.0, 0.01, 0.05],
        "max_delta_step": [0, 5, 30, 100, 300, 500],
        "min_samples_split": [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1.0],
        "min_samples_leaf": [0.001, 0.01, 0.1, 0.3, 0.5],
        "num_leaves": [15, 30, 70, 100, 150, 200, 300],
        "subsample": [1.0 / len(X_train), 0.01, 0.3, 0.5, 0.7, 1.0],
        "colsample_bytree": [1.0 / X_train.shape[1], 0.01, 0.1, 0.3, 0.5, 0.7, 1.0],
        "max_leaf_nodes": [None, 2, 10, 50, 100, 300, 1000],
        "reg_alpha": [1e-5, 1e-2, 0.1, 1, 25, 100],
        "reg_lambda": [1e-5, 1e-2, 0.1, 1, 25, 100],
        "alpha": [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9],
        "ccp_alpha": [0, 10, 50, 100, 300, 1000],
        "tol": [1e-4, 1e-5, 1e-6],
    }

    allowed = {k: v for k, v in domain_params_dicts.items() if k in params}

    if len(allowed) == 0:
        assert "There are no allowed parameters."

    new_params = split_dict2(allowed, 2, seed)

    probe = []
    union = {}

    for params_dict in new_params:

        print("GridSearch for ", params_dict)

        clf = GridSearchCV(
            model, params_dict, scoring=scoring, verbose=0, cv=n_folds, refit=True
        )

        res = clf.fit(X_train, y_train)

        print(f"Regressor {regressor}, score: {res.best_score_}")

        # print(clf.best_score_)
        # print(clf.best_estimator_)
        # print(sorted(clf.cv_results_))
        # print(clf.best_params_)

        probe.append((res.best_score_, res.best_params_))
        union.update(res.best_params_)

        if len(union) > 2:
            model.set_params(**union)
            model.fit(X_train, y_train)

            new_score = scoring._score_func(model.predict(X_train), y_train)
            probe.append((new_score, model.get_params()))

    # print(union)

    print(probe)

    the_best_score, the_best_params = (
        max(probe, key=lambda x: abs(x[0]))
        if greater_is_better_
        else min(probe, key=lambda x: abs(x[0]))
    )

    # if not math.isclose(the_best_score, abs(new_score), rel_tol=1e-6):

    model.set_params(**the_best_params)
    model.fit(X_train, y_train)

    print(f"Regressor {regressor}, the best: {model.get_params()}")

    return model


if __name__ == "__main__":

    import xgboost
    import lightgbm

    # from catboost import CatBoostRegressor

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import make_scorer, mean_squared_error

    import pandas as pd
    import numpy as np

    train_ = pd.read_csv(
        "./dataset/train_parody.csv", sep=",", header=None, na_values=["NULL"]
    )
    pred_ = pd.read_csv(
        "./dataset/test_parody.csv", sep=",", header=None, na_values=["NULL"]
    )

    X, Y = np.ascontiguousarray(train_.iloc[:, :-1]), np.ascontiguousarray(
        train_.iloc[:, -1]
    )
    X2, Y2 = np.ascontiguousarray(pred_.iloc[:, :-1]), np.ascontiguousarray(
        pred_.iloc[:, -1]
    )

    table = """| Regressor | RMSE, bare | RMSE, more trees instead | RMSE, autotuned |\n"""
    table += """| --- | --- | --- | --- |\n"""

    table += "| **XGBRegressor** |"

    model = xgboost.XGBRegressor()
    model.fit(X, Y, eval_metric="rmse")

    Y2_pred = model.predict(X2)
    RMSE = mean_squared_error(Y2_pred, Y2)

    table += " %.6f |" % RMSE

    model = xgboost.XGBRegressor(n_estimators=500)
    model.fit(X, Y, eval_metric="rmse")

    Y2_pred = model.predict(X2)
    RMSE = mean_squared_error(Y2_pred, Y2)

    table += " %.6f |" % RMSE

    accuracy = make_scorer(mean_squared_error, greater_is_better=False)
    fitted_model = fit_parameters(
        model=xgboost.XGBRegressor(),
        X_train=X,
        y_train=Y,
        scoring=accuracy,
        n_folds=3,
        seed=100,
    )

    Y2_pred = fitted_model.predict(X2)
    RMSE = mean_squared_error(Y2_pred, Y2)

    table += " %.6f |\n" % RMSE

    table += "| **GradientBoostingRegressor** |"

    model = GradientBoostingRegressor()
    model.fit(X, Y)  # , eval_metric='rmse')

    # print(model.get_params())

    Y2_pred = model.predict(X2)
    RMSE = mean_squared_error(Y2_pred, Y2)

    table += " %.6f |" % RMSE

    model = GradientBoostingRegressor(n_estimators=500)
    model.fit(X, Y)

    Y2_pred = model.predict(X2)
    RMSE = mean_squared_error(Y2_pred, Y2)

    table += " %.6f |" % RMSE

    accuracy = make_scorer(mean_squared_error, greater_is_better=False)
    fitted_model = fit_parameters(
        model=GradientBoostingRegressor(),
        X_train=X,
        y_train=Y,
        scoring=accuracy,
        n_folds=3,
        seed=100,
    )

    Y2_pred = fitted_model.predict(X2)
    RMSE = mean_squared_error(Y2_pred, Y2)

    table += " %.6f |\n" % RMSE

    table += "| **LGBMRegressor** |"

    model = lightgbm.LGBMRegressor(verbose=-1)
    model.fit(X, Y)  # , eval_metric='rmse')

    Y2_pred = model.predict(X2)
    RMSE = mean_squared_error(Y2_pred, Y2)

    table += " %.6f |" % RMSE

    model = lightgbm.LGBMRegressor(verbose=-1, n_estimators=500)
    model.fit(X, Y)

    Y2_pred = model.predict(X2)
    RMSE = mean_squared_error(Y2_pred, Y2)

    table += " %.6f |" % RMSE

    accuracy = make_scorer(mean_squared_error, greater_is_better=False)
    fitted_model = fit_parameters(
        model=lightgbm.LGBMRegressor(),
        X_train=X,
        y_train=Y,
        scoring=accuracy,
        n_folds=3,
        seed=100,
    )

    Y2_pred = fitted_model.predict(X2)
    RMSE = mean_squared_error(Y2_pred, Y2)

    table += " %.6f |\n" % RMSE

    """
    table += "| bare **CatBoostRegressor** |"

    model = CatBoostRegressor(verbose=0)
    model.fit(X, Y)  # , eval_metric='rmse')

    Y2_pred = model.predict(X2)
    RMSE = mean_squared_error(Y2_pred, Y2)

    table += " %.5f |\n" % RMSE
    table += "| autotuned **CatBoostRegressor** |"

    accuracy = make_scorer(mean_squared_error, greater_is_better=False)
    fitted_model = fit_parameters(
        model=CatBoostRegressor(), X_train=X, y_train=Y, scoring=accuracy, n_folds=2
    )

    Y2_pred = fitted_model.predict(X2)
    RMSE = mean_squared_error(Y2_pred, Y2)

    table += " %.5f |\n" % RMSE
    """

    with open("result_table.txt", "w") as out:
        out.write(table)
