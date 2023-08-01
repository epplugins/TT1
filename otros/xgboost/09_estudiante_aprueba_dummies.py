# Objetivo: aprueba los parciales.
# Repeticiones cambiando las observaciones de los conj. de entrenamiento y de prueba.
# SOLO 5 repeticiones.
# Todas la variables que se pueden usar para este análisis más dummies.
# Busca los mejores hiperparámetros: métrica aucpr.
# Entrena el modelo y guarda AP score y la importancia de las variables.

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

# Funciones para entrenar y evaluar los modelos.
import lib_entrenar as le

df = pd.read_csv("../datos/dataset_02-feateng.csv")
df['SEDE'] = df['SEDE'].astype('str')
df['MATERIA'] = df['MATERIA'].astype('str')
df['edad'] = df['edad'].astype('category')
df['objetivo'] = ((df['condición'] == 'Promociona') | (df['condición'] == 'Examen')).astype(int)
df2 = df.loc[(df['valido2']==1) & (df['condición'] != 'Abandona1')].copy()
columnas = ['cuat', 'SEDE', 'MATERIA', 'codCarrera', 'facultad',
        'extranjero', 'edad', 'prom_edad', 'turno', 'n_alum', 'p_ext',
        'recurso', 'p_recursa', 'sala', 'abandona1_p', 'objetivo']
df2 = df2[columnas]
df2 = pd.get_dummies(df2, columns=['SEDE', 'turno', 'edad',
                                   'codCarrera', 'facultad', 'sala'])

X = df2.drop(['objetivo'], axis=1).copy()
y = df2[['objetivo']].copy()

# Columnas con variables categóricas:
cats = ['cuat', 'MATERIA', 'extranjero']
for col in cats:
   X[col] = X[col].astype('category')

# Una lista de estados para el barajamiento de la librería train_test_split.
# random_states = [1, 82, 93023, 3232, 22, 1394, 77384, 888, 982, 2]
random_states = [1, 82, 93023, 3232, 22]
j = 0
importancia = pd.DataFrame()
aps = np.array([])
for state in random_states :
   # Separar los datos.
    X_train, X_test, y_train, y_test = train_test_split(X, y,
            test_size=0.3,
            random_state=state,
            stratify=y)

    dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, y_test, enable_categorical=True)

    # aucpr
    print("\n===================")
    print("Experimento ",j,"\n")
    scale_pos_weight = 0.5
    ns = 40
    nrounds = 1000
    ngrid = 5
    metric = 'aucpr'
    best_early_stopping, bestparams = le.ajustar(dtrain, scale_pos_weight, ns, nrounds, ngrid)

    print("Mejor N: ", best_early_stopping)
    print(bestparams)
    bst = xgb.train(bestparams, dtrain, num_boost_round=best_early_stopping)

    y_pred = bst.predict(dtest)
    aps = np.concatenate([aps,[average_precision_score(y_test, y_pred)]])
    importancia[j] = pd.DataFrame.from_dict(bst.get_fscore(), orient='index')
    j = j+1

np.savetxt("salida_09_average_precision_scores.txt", aps)
importancia.to_csv("salida_09_importancias.csv")

print("Resultados guardados en")
print("salida_09_average_precision_scores.txt y")
print("salida_09_importancias.csv")
