# Test de permutaciones: permutando la variable objetivo.
# Objetivo: abandona entre 1er y 2do parcial.
# Todas la variables que se pueden usar para este análisis.
# Busca los mejores hiperparámetros: métrica aucpr.
# Entrena el modelo.
# Calcula average precision score.

import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
import xgboost as xgb

# Funciones para entrenar y evaluar los modelos.
import lib_entrenar as le

df = pd.read_csv("../datos/dataset_02-feateng.csv")
df['SEDE'] = df['SEDE'].astype('str')
df['MATERIA'] = df['MATERIA'].astype('str')
df['edad'] = df['edad'].astype('category')
df['objetivo'] = (df['condición'] == 'Abandona2').astype(int)

conj_train = np.loadtxt("indices_estudiante_train.txt.gz")
df2 = df.loc[conj_train].copy()
print("Cantidad de observaciones para entrenamiento: ",len(df2))
print("Positivos/Total: ", round(df2.objetivo.sum()/len(df2),3))

columnas = ['cuat', 'SEDE', 'MATERIA', 'pa1', 'pa1_prom', 'facultad',
        'extranjero', 'edad', 'prom_edad', 'turno', 'n_alum', 'p_ext',
        'recurso', 'p_recursa', 'abandona1_p']
X_train = df2[columnas].copy()
y_train = df2[['objetivo']].copy()
print("Variables: ", X_train.columns.values)

# Columnas con variables categóricas:
cats = ['cuat', 'SEDE', 'MATERIA', 'facultad', 'extranjero', 'edad',
       'turno']

for col in cats:
   X_train[col] = X_train[col].astype('category')

# Variable objetivo permutada, manteniendo la proporción.
# rng = np.random.default_rng()
# pos = np.sum(y_train.values)/len(y_train)
# neg = 1-pos
# y_train_rand = rng.choice([0, 1], len(y_train), p=[neg, pos])

dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=True)

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

conj_test = np.loadtxt("indices_estudiante_test.txt.gz")
df2_test = df.loc[conj_test].copy()
print("Cantidad de observaciones para test: ",len(df2_test))
print("Positivos/Total: ", round(df2_test.objetivo.sum()/len(df2_test),3))

X_test = df2_test[columnas].copy()
y_test = df2_test[['objetivo']].copy()
y_test = np.random.permutation(y_test)

for col in cats:
   X_test[col] = X_test[col].astype('category')

dtest = xgb.DMatrix(X_test, y_test, enable_categorical=True)

y_pred = bst.predict(dtest)
print("\nClasificador aleatorio.\nAverage precision score =", round(average_precision_score(y_test, y_pred),3))
