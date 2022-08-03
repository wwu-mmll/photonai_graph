[![Python application](https://github.com/wwu-mmll/photonai_graph/actions/workflows/lintandtest.yml/badge.svg)](https://github.com/wwu-mmll/photonai_graph/actions/workflows/lintandtest.yml)
[![Coverage Status](https://coveralls.io/repos/github/wwu-mmll/photonai_graph/badge.svg?branch=master)](https://coveralls.io/github/wwu-mmll/photonai_graph?branch=master)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=wwu-mmll_photonai_graph&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=wwu-mmll_photonai_graph)
![GitHub](https://img.shields.io/github/license/wwu-mmll/photonai_graph)
[![Twitter URL](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fwwu_mmll)](https://twitter.com/wwu_mmll)

![PHOTONAI Graph](https://raw.githubusercontent.com/wwu-mmll/photonai_graph/master/documentation/docs/assets/img/photonai-02.png)

# photonai-graph
Photon Graph is an extension for the PHOTON framework that allows for the use of machine learning based on graphs. Furthermore, the Graph Utilities contain a wide variety of functions that allow for the visualization and converting of graphs.

# Documentation
You can find a detailed documentation here: https://wwu-mmll.github.io/photonai_graph/

# Installation

To install photonai-graph create a dedicated conda/python environment and activate it. Then install photonai-graph via

```bash
pip install photonai-graph
```

To be able to use all modules of the toolbox you will still need to install tensorflow, dgl, pytorch and grakel according to your system configuration, for example with

```bash
pip install tensorflow
pip install torch
pip install dgl
pip install grakel
```

For graph embeddings the gem python package is needed, along with tensorflow. Please install tensorflow according to your system.

```bash
pip install nxt-gem
pip install tensorflow
```

For graph kernels the grakel package needs to be installed. You can install grakel via pip.

```bash
pip install git+https://github.com/ysig/GraKeL.git@cfd14e0543075308d201327ac778a48643f81095'
```

For graph neural networks pytorch and deep graph library are required. You can install them via pip

```bash
pip install torch
pip install dgl
```


