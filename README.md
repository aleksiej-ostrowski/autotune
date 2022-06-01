This experimental version for auto-tuning is based 
on script by Sylwia Mielnicka https://github.com/SylwiaOliwia2/xgboost-AutoTune 

# AutoTune for regressors
Package allows for auto-tuning `XGBRegressor`, `GradientBoostingRegressor` and `LGBMRegressor` parameters. Model usues `GridSearchCV`. Tested for Python 3.8.10

## Create scorer
You need to define scoring. The easier way is using `sklearn.metrics.make_scorer`. 

## Fast run

```python
import xgboost
from sklearn.metrics import make_scorer, mean_squared_error
from autotune import fit_parameters

accuracy = make_scorer(mean_squared_error, greater_is_better=False)
fitted_model = fit_parameters(model=xgboost.XGBRegressor(), X_train=X, y_train=Y, scoring=accuracy, seed=100)    

Y2_pred = fitted_model.predict(X2)
```

### Parameters

* **X_train** - pandas DataFrame, input variables.
* **y_train** - pandas series, value for prediction.
* **scoring** - used to evaluate the best model.
* **n_folds** - number of folds used in GridSearchCV.

## Fast run test

```bash
bash test.sh
```

## Test results

| Regressor | RMSE, bare | RMSE, more trees instead | RMSE, autotuned |
| --- | --- | --- | --- |
| **XGBRegressor** | 0.000511 | 0.000493 | 0.000123 |
| **GradientBoostingRegressor** | 0.000662 | 0.000100 | 0.000065 |
| **LGBMRegressor** | 0.000333 | 0.000181 | 0.000111 |

**Warning!** To get the best autotuning, you have to play around with the parameter `seed` in the function `fit_parameters(seed=200)`.
