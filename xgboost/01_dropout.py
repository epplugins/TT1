# Objetivo: abandona entre 1er y 2do parcial.
# Todas la variables que se pueden usar para este análisis con SEDE y no sala.
# Busca los mejores hiperparámetros: métrica aucpr.
# Entrena el modelo y lo guarda.

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score

# Funciones para entrenar y evaluar los modelos.
import lib_entrenar as le

df = pd.read_csv("../datos/dataset_02-feateng.csv")
df['SEDE'] = df['SEDE'].astype('str')
df['MATERIA'] = df['MATERIA'].astype('str')
df['edad'] = df['edad'].astype('category')
df['objetivo'] = (df['condición'] == 'Abandona2').astype(int)

elegidas = np.loadtxt("indices_estudiante_train.txt.gz")
df_train = df.loc[elegidas].copy()
print("Cantidad de observaciones para entrenamiento: ",len(df_train))
print("Positivos/Total: ", round(df_train.objetivo.sum()/len(df_train),3))
conj_test = np.loadtxt("indices_estudiante_test.txt.gz")
df_test = df.loc[conj_test].copy()

columnas = ['cuat', 'SEDE', 'MATERIA', 'pa1', 'pa1_prom', 'facultad',
        'extranjero', 'edad', 'prom_edad', 'turno', 'n_alum', 'p_ext',
        'recurso', 'p_recursa', 'abandona1_p']
X_train = df_train[columnas].copy()
y_train = df_train[['objetivo']].copy()
X_test = df_test[columnas].copy()
y_test = df_test[['objetivo']].copy()

print("Variables: ", X_train.columns.values)

# Columnas con variables categóricas:
cats = ['cuat', 'SEDE', 'MATERIA', 'facultad', 'extranjero', 'edad',
       'turno']
for col in cats:
   X_train[col] = X_train[col].astype('category')
   X_test[col] = X_test[col].astype('category')

aps = np.array([])
for c in columnas:
    X_tr = X_train.drop(c, axis=1).copy()
    dtrain = xgb.DMatrix(X_tr, y_train, enable_categorical=True)

    # aucpr
    print("\n===================")
    print("Validación cruzada con aucpr\n")
    scale_pos_weight = 0.3
    ns = 40
    nrounds = 1000
    ngrid = 30
    metric = 'aucpr'
    best_early_stopping, bestparams = le.ajustar(dtrain, scale_pos_weight, ns, nrounds, ngrid)

    print("Mejor N: ", best_early_stopping)
    print(bestparams)
    bst = xgb.train(bestparams, dtrain, num_boost_round=best_early_stopping)

    X_te = X_test.drop(c, axis=1).copy()
    dtest = xgb.DMatrix(X_te, y_test, enable_categorical=True)
    y_pred = bst.predict(dtest)
    ap = average_precision_score(y_test, y_pred)
    aps = np.concatenate([aps, [ap]])

np.savetxt("salida_01_droput_aps.txt", aps)
print("\nAps guardados en salida_01_droput_aps.txt")
