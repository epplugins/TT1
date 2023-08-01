## TODO

1. 01_dataprep - Imputación de carrera y facultad en los códigos que no tienen esa información:
    1. Creo que es NMAR, se puede hacer un estudio de agrupamiento para decidir mejor que carrera y/o facultad usar en cada caso.
    1. buscar referencias para esto.
1. Feat eng: carrera que no requiere esta materia en cbc
1. Hot-encode del conjunto de notas altas.
1. Repetir análisis con el conjunto de notas bajas.
1. Figuras de curvas PR.
1. Ingeniería de variables: incluir variable si vuelve a cursar la materia luego de reprobar. Es otra medida de deserción.
1. Terminar otros análisis:
    1. Por curso
    1. Condición: Aprueba


## Git

```bash
git clone git@eppluginsgithub:epplugins/tt1 TT1
```

https://git-scm.com/book/en/v2/Git-Branching-Rebasing

Si se desea una rama de desarrollo:

```bash
git checkout dev
```

```bash
git rebase main
git checkout main
git merge dev
```

## Environment

Create an environment using the latest nodejs:
```bash
conda create -n tt1 -c conda-forge nodejs
```

```bash
conda install -n tt1 -c conda-forge python matplotlib numpy pandas jupyter_server jupyterlab scikit-image tikzplotlib statsmodels gplearn sympy
```

For interactive plots:
```bash
conda install -n tt1 -c conda-forge ipympl
```

```bash
conda install -n tt1 -c plotly plotly
```

### Automatic exploratory data analisys:

https://github.com/ydataai/ydata-profiling

Cuidado, cambia la versión de matplotlib.
```bash
conda activate tt1
pip install -U ydata-profiling
```

https://github.com/microsoft/vscode-jupyter/wiki/IPyWidget-Support-in-VS-Code-Python

Add this to settings.json:
```bash
"jupyter.widgetScriptSources": ["jsdelivr.com", "unpkg.com"],
```

Export environment:
```bash
conda activate tt1
conda env export > tt1-environment.yml
```

Creating environment from yml file:
```bash
conda env create -f tt1-environment.yml
```

## Exporting matplotlib figures to pgfplots
```tex
\usepackage{layouts}
```
Use this to get the width:
```tex
\printinunitsof{in}\prntlen{\textwidth}
```
and adjust the matplotlib figure based on that:
```python
fig.set_size_inches(w=4.7747, h=3.5)
```

