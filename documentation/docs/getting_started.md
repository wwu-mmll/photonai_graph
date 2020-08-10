# This is how you get started

In order to get started, you will need connectivity matrices or data, that is already graph data. For this introduction we will assume that you are using connectivity matrices.

## Installation

First install photonai-graph into a new python/conda environment.

```
pip install photonai
pip install photonai-graph
```

## Loading your matrices

If you are using dense matrices, photonai-graph assumes that they have a certain shape: They should come as numpy matrices or array with the dimensions **Subjects x nodes x nodes x modalities (optional)**. If you are using matlab files, from CONN or other popular neuroimaging connectivity toolboxes, you will have to import them. An example function on how to import matlab data matrices can be seen [here](https://github.com/BenisFarmen/connectivity_loading/blob/master/load_functions.py). 

With the help of these support scripts it will be easier/more convenient to load the connectivity matrices especially if you come from a different programming language or background. They allow for loading the matrices into the right format with one line of code.

```python
from  script import function

matrices = load_conn_matrix('/path/to/your/data.mat')
```

## Load your labels

Now you can load your labels. Make sure that your labels match the number of subjects.

```python
import pandas as pd

df = pd.read_csv('/path/to/labels.csv')
labels = df['your_label']
```

## Set up pipeline

Now you can set up a photon pipeline. You will need to instantiate a Hyperpipe class which manages your validation settings for hyperparameter optimization.


## Evaluate pipeline

After fitting your pipeline you can now evaluate the pipeline by uploading your json results file to the photon explorer.

## Why do we need graphs?

```python
from photonai-graph import GraphConstruction

graph = GraphConstruction()
graph.do_something_cool()
```



```python
import pandas
```



!!! note "Important Information"

  Please consider that this toolkit is still under active development. Use wisely!



![Image Title](img/test.jpg)

