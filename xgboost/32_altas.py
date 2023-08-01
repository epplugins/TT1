# Objetivo: abandona entre 1er y 2do parcial.
# Notas Altas.
# Todas la variables que se pueden usar para este análisis con SEDE y no sala.
# Busca los mejores hiperparámetros: métrica aucpr.
# Entrena el modelo y lo guarda.

import pandas as pd
import numpy as np
import xgboost as xgb

# Funciones para entrenar y evaluar los modelos.
import lib_entrenar as le

df = pd.read_csv("../datos/dataset_30-notas_altas.csv")
df['SEDE'] = df['SEDE'].astype('str')
df['MATERIA'] = df['MATERIA'].astype('str')
df['edad'] = df['edad'].astype('category')
df['objetivo'] = df['objetivo'].astype('int')
print("\nCantidad de observaciones con notas > 3: ", len(df))
print("Positivos/Total: ", round(df.objetivo.sum()/len(df),3))

elegidas = np.loadtxt("indices_notas_altas_train.txt.gz")
df2 = df.loc[elegidas].copy()
print("Cantidad de observaciones para entrenamiento: ",len(df2))
print("Positivos/Total: ", round(df2.objetivo.sum()/len(df2),3))

columnas = ['cuat', 'SEDE', 'MATERIA', 'pa1', 'facultad',
       'extranjero', 'turno', 'n_alum', 'p_ext', 'recurso', 'p_recursa',
       'pa1_prom', 'edad', 'prom_edad', 'abandona1_p', 'objetivo']
df2 = df2[columnas]
df2 = pd.get_dummies(df2, columns=['SEDE', 'turno', 'edad',
                                   'facultad'])

X_train = df2.drop(['objetivo'], axis=1).copy()
y_train = df2[['objetivo']].copy()
print("Variables: ", X_train.columns.values)

# Columnas con variables categóricas:
cats = ['cuat', 'MATERIA', 'extranjero']
for col in cats:
   X_train[col] = X_train[col].astype('category')

dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=True)

# aucpr
print("\n===================")
print("Validación cruzada con aucpr\n")
scale_pos_weight = 0.05
ns = 60
nrounds = 1000
ngrid = 200
metric = 'aucpr'
best_early_stopping, bestparams = le.ajustar(dtrain, scale_pos_weight, ns, nrounds, ngrid)

print("Mejor N: ", best_early_stopping)
print(bestparams)
bst = xgb.train(bestparams, dtrain, num_boost_round=best_early_stopping)
bst.save_model("modelo_32_altas_aucpr.json")
print("\nModelo guardado en modelo_32_altas_aucpr.json")
