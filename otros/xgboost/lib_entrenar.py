#
# Librería de funciones para entrenar y evaluar
# modelos XGBoost.
#

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

def ajustar(dtrain, scale_pos_weight, ns, nrounds, ngrid):
    """
    Encuentra los mejores parámetros mediante una búsqueda
    en grilla y validación cruzada usando la métrica aucpr.

    Parameters
    ----------
    dtrain :
        Los datos de entrenamiento.
    scale_pos_weight : float
        Proporción positivos / negativos que sirve como a priori para el modelo.
    ns : integer
        Número de elementos en cada parámetro de la grilla.
    nrounds : integer
        Números de iteraciones del boost.
    ngrid : integer
        Número de muestras de la grilla.

    Returns
    -------
    tuple : (best_early_stopping, bestparams)
        El mejor número de iteraciones para no sobreajustar y los mejores parámetros.
    """

    nfold = 5
    early_stopping_rounds = 10
    seed = 121234

    maxresult = 0
    rng = np.random.default_rng()

    eta = rng.choice(np.linspace(0.005, 0.6, ns), ngrid)
    gamma = rng.choice(np.linspace(0, 20, ns), ngrid) # [0, inf) Larger is more conservative
    max_depth = rng.choice(np.arange(2,20,1), ngrid) # Larger tends to overfits.
    Lambda = rng.choice(np.linspace(0, 20, ns), ngrid)
    metrics = ["aucpr"]
    eval_metric = ["aucpr"]
    num_boost_round = nrounds

    for i in np.arange(ngrid) :
        params = {
        "objective"        : "binary:logistic",
        "eval_metric"      : eval_metric,
        "eta"              : eta[i],
        "gamma"            : gamma[i],
        "max_depth"        : max_depth[i],
        "lambda"           : Lambda[i],
        "alpha"            : 0,
        "scale_pos_weight" : scale_pos_weight,
        }

        results = xgb.cv(params, dtrain,
            num_boost_round=num_boost_round,
            nfold=nfold,
            metrics=metrics,
            early_stopping_rounds=early_stopping_rounds,
            seed=seed,
            )

        rounds = len(results)
        print(i," rounds=",rounds," aucpr=",round(results['test-aucpr-mean'][rounds-1],4), " std=", round(results['train-aucpr-std'][rounds-1],4))

        if maxresult < results['test-aucpr-mean'].max() :
            maxresult = results['test-aucpr-mean'].max()
            best_early_stopping = len(results)
            bestparams = params

    return (best_early_stopping, bestparams)