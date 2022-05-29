# https://stackoverflow.com/questions/22878743/how-to-split-dictionary-into-multiple-dictionaries-fast
def split_dict(d, n_chunks):
    keys = list(d.keys())
    for i in range(0, len(keys), n_chunks):
        yield {k: d[k] for k in keys[i : i + n_chunks]}

import random

def split_dict2(d, n_chunks, seed = 42):
    random.seed(seed)
    keys = list(d.keys())
    random.shuffle(keys)
    for i in range(0, len(keys), n_chunks):
        yield {k: d[k] for k in keys[i : i + n_chunks]}


if __name__ == "__main__":

    d = {
        "n_estimators": [100, 150, 200, 300],
        "learning_rate": [0.0001, 0.001, 0.005, 0.01, 0.04, 0.07, 0.1, 0.2, 0.3],
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
        "subsample": [0.01, 0.1, 0.3, 0.5, 0.7, 1.0],
        "colsample_bytree": [0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 1.0],
        "max_leaf_nodes": [2, 10, 50, 100, 300, 1000],
        "reg_alpha": [1e-5, 1e-2, 0.1, 1, 25, 100],
        "reg_lambda": [1e-5, 1e-2, 0.1, 1, 25, 100],
        "alpha": [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9],
        "ccp_alpha": [0, 10, 50, 100, 300, 1000],
    }

    print(d)
    print("splited: ", list(split_dict2(d, 2, seed = 18)))
