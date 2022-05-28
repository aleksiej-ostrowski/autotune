This experimental version for auto-tuning is based on script by Sylwia Mielnicka https://github.com/SylwiaOliwia2/xgboost-AutoTune 

# AutoTune for regressors
Package allows for auto-tuning `XGBRegressor`, `GradientBoostingRegressor` and `LGBMRegressor` parameters. Model usues `GridSearchCV`. Tested for Python 3.8.10

## Create scorer
You need to define scoring. The easier way is using `sklearn.metrics.make_scorer`. 

## Fast run

```
import xgboost
from sklearn.metrics import make_scorer, mean_squared_error
from autotune import fit_parameters

accuracy = make_scorer(mean_squared_error, greater_is_better=False)
fitted_model = fit_parameters(model = xgboost.XGBRegressor(), X_train = X, y_train = Y, scoring=accuracy, n_folds=2)    

Y2_pred = fitted_model.predict(X2)
```

### Parameters:

* **X_train** - pandas DataFrame, input variables.
* **y_train** - pandas series, value for prediction.
* **scoring** - used to evaluate the best model.
* **n_folds** - number of folds used in GridSearchCV.


## Test results

| Regressor | RMSE |
| --- | --- |
| bare **XGBRegressor** | 0.00051 |
| autotuned **XGBRegressor** | 0.00022 |
| bare **GradientBoostingRegressor** | 0.00066 |
| autotuned **GradientBoostingRegressor** | 0.00017 |
| bare **LGBMRegressor** | 0.00033 |
| autotuned **LGBMRegressor** | 0.00015 |
