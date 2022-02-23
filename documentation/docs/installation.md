# Installation
The bare installation of PHOTONAI Graph will get you up and running with machine learning graph analysis.

!!! info
    Some analysis may require additional packages. PHOTONAI Graph will inform you about these requirements if needed.
    If you want to install them in advance please read [additional packages](#additional-packages).


## Prerequisites
We recommend using [anaconda](https://www.anaconda.com/) environments. Alternatively you can simply use virtual environments or install PHOTONAI Graph system wide.

PHOTONAI Graph is currently tested with python 3.9
```shell
conda create -n photonai_graph
conda activate photonai_graph
conda install python=3.9
```
## Installation via PIP

The PHOTONAI Graph library can simply be installed via pip:

```shell
pip install photonai-graph
```

If you prefer to use the dev version of PHOTONAI Graph:

!!! warning 
    This Version could be unstable due to active development

```shell
pip install git+https://github.com/wwu-mmll/photonai_graph@dev
```

## Installation verification
You can verify your installation by starting python and runnning:

```python
import photonai_graph
```

If this command does not raise an error, you are ready to go.

## Additional packages
To be able to use all modules of the toolbox you have to install these additional packages:
```shell
pip install tensorflow
pip install torch
pip install dgl
pip install grakel
```