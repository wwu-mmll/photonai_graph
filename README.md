# photonai-graph
Photon Graph is an extension for the PHOTON framework that allows for the use of machine learning based on graphs. Furthermore, the Graph Utilities contain a wide variety of functions that allow for the visualization and converting of graphs.

# Documentation
You can find a detailed documentation here: https://wwu-mmll.github.io/photonai_graph/

# Installation

To install photonai-graph create a dedicated conda/python environment and activate it. Then install photonai-graph via

```bash
pip install git+https://github.com/wwu-mmll/photonai_graph.git@dev
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
pip install git+https://github.com/jernsting/nxt_gem.git
pip install tensorflow
```

For graph kernels the grakel package needs to be installed. You can install grakel via pip.

```bash
pip install git+https://github.com/ysig/GraKeL.git@cfd14e0543075308d201327ac778a48643f81095'
```

For graph neural networks pytorch and deep grap library are required. You can install them via pip

```bash
pip install torch
pip install dgl
```


