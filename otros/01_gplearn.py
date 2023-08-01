import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicRegressor
from sympy import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


df_cursos = pd.read_csv("../datos/dataset_04-feateng-cursos.csv")
df_cursos['SEDE'] = df_cursos['SEDE'].astype('str')
df_cursos['cuat'] = df_cursos['cuat'].astype('str')
df_cursos['MATERIA'] = df_cursos['MATERIA'].astype('str')
df_cursos['sala'] = df_cursos['sala'].astype('str')
df2 = df_cursos.loc[(df_cursos['valido2'] == 1) &
                    (df_cursos['abandona1_p'] < 1)].copy()
print("Cantidad de cursos válidos:", len(df2))

df2['n_alum_scaled'] = MinMaxScaler().fit_transform(np.array(df2['n_alum']).reshape(-1,1))
df2['prom_edad_scaled'] = MinMaxScaler().fit_transform(np.array(df2['prom_edad']).reshape(-1,1))
df2 = df2.drop(['anio', 'curso', 'n_alum', 'prom_edad', 'pa2_prom',
                'aprueba_p', 'final_prom',
                'valido1', 'valido2'], axis=1)
# Sedes grandes y otras (O).
df2.loc[~df2['SEDE'].isin(['2','10','4','1','5','6']), 'SEDE'] = 'O'

X = df2.drop(['pa1_prom', 'abandona1_p', 'abandona2_p', 'aprueba_rel_p'], axis=1)
X = pd.get_dummies(X, columns=['cuat', 'SEDE', 'MATERIA', 'turno'])
y = df2['aprueba_rel_p']

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

function_set = ['add', 'sub', 'mul', 'div','cos','sin','neg','inv']

est_gp = SymbolicRegressor(generations=100,
                           stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0,
                          feature_names=X_train.columns)

est_gp.fit(X_train, y_train)
print('R2:',est_gp.score(X_test,y_test))