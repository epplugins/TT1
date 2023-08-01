# Objetivo: abandona entre 1er y 2do parcial.
# Todas la variables que se pueden usar para este análisis con SEDE y no sala, con pa1.
# Busca los mejores hiperparámetros: métrica aucpr.
# Entrena el modelo y lo guarda.

import pandas as pd
import numpy as np
import xgboost as xgb

# Funciones para entrenar y evaluar los modelos.
import lib_entrenar as le

df = pd.read_csv("../datos/dataset_02-feateng.csv")
df['SEDE'] = df['SEDE'].astype('str')
df['MATERIA'] = df['MATERIA'].astype('str')
df['edad'] = df['edad'].astype('category')
df['objetivo'] = (df['condición'] == 'Abandona2').astype(int)

elegidas = np.loadtxt("indices_estudiante_train.txt.gz")
df2 = df.loc[elegidas].copy()
print("Cantidad de observaciones para entrenamiento: ",len(df2))
print("Positivos/Total: ", round(df2.objetivo.sum()/len(df2),3))

columnas = ['cuat', 'SEDE', 'MATERIA', 'pa1', 'pa1_prom', 'codCarrera',
        'extranjero', 'edad', 'prom_edad', 'turno', 'n_alum', 'p_ext',
        'recurso', 'p_recursa', 'abandona1_p']
X_train = df2[columnas].copy()
y_train = df2[['objetivo']].copy()
print("Variables: ", X_train.columns.values)

# Columnas con variables categóricas:
cats = ['cuat', 'SEDE', 'MATERIA', 'codCarrera', 'extranjero', 'edad',
       'turno']
for col in cats:
   X_train[col] = X_train[col].astype('category')

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
bst.save_model("modelo_01d_estudiante_aucpr.json")
print("\nModelo guardado en modelo_01d_estudiante_aucpr.json")
