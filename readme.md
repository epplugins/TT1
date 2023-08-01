# Taller de Tesis 1

## Entorno

El archivo tt1-environment.yml contiene toda la información de las librerías y paquetes del entorno de trabajo.

Crear entorno desde archivo yml en miniconda:
```bash
conda env create -f tt1-environment.yml
```

## Preparación de los datos

El código de esta sección no puede ser ejecutado porque se eliminaron los datos originales para no mantener datos de estudiantes en un repositorio público, pero se pueden consultar sus estados ejecutados.

Dentro del directorio ``preparacion_dataset``:
1. ``01_dataprep.ipynb``: limpieza y estrandarización.
    * Resultados guardados en: ``datos/dataset_01.csv``
    * ``datos/codigos_carreras.csv``: referencia para poder eliminar la columna con los nombres de las carreras.
1. ``02_feateng.ipynb``: ingeniería de variables.
1. ``03_grupos_dni.ipynb`` : una simple exploración sobre la distribución de los números de DNI por cuatrimestre.


## Análisis Exploratorio

El código y sus resultados se encuentran dentro del directorio ``analisis_exploratorio`` .  


## Métodos

Directorio ``/xgboost`` .  

Comenzar en la notebook ``xgboost/00_inicio.ipynb``

