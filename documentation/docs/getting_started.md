# Getting started

In order to get started, you will need connectivity matrices or data, that is already graph data. For this introduction we will assume that you are using connectivity matrices.

## Installation

First install photonai-graph into a new python/conda environment.

```
pip install photonai
pip install photonai-graph
```

## Loading your matrices

If you are using dense matrices, photonai-graph assumes that they have a certain shape: They should come as numpy matrices or array with the dimensions **Subjects x nodes x nodes x modalities (optional)**. If you are using matlab files, from CONN or other popular neuroimaging connectivity toolboxes, you will have to import them. An example function on how to import matlab data matrices can be seen [here](https://github.com/BenisFarmen/connectivity_loading/blob/master/load_functions.py). 

With such a function you can then load your data.

```python
from  connectivity_loading import load_conn

matrices = load_conn('/path/to/your/data.mat')
```

## Load your labels

Now you can load your labels. Make sure that your labels match the number of subjects.

```python
import pandas as pd

df = pd.read_csv('/path/to/labels.csv')
labels = df['your_label']
```

## Set up pipeline

Now you can set up a photon pipeline. You will need to instantiate a Hyperpipe class which manages your validation settings for hyperparameter optimization. Then you can add elements to the pipeline.

```python
from photonai.base import Hyperpipe, PipelineElement
from sklearn.model_selection import KFold

my_pipe = Hyperpipe('basic_gmeasure_pipe',
                    inner_cv=KFold(n_splits=5),
                    outer_cv=KFold(n_splits=5),
                    optimizer='grid_search',
                    metrics=['mean_absolute_error'],
                    best_config_metric='mean_absolute_error')
```

Having chosen the pipeline settings, you can now add elements to your pipeline. Here we first add a constructor to threshold our connectivity matrices, then a measure transformer to extract graph measures which will then be used by the SVM to estimate the label.

```python
my_pipe.add(PipelineElement('GraphConstructorThreshold',
                            hyperparameters={'threshold': 0.8}))

my_pipe.add(PipelineElement('GraphMeasureTransform',
                            hyperparameters={'graph_functions': {"large_clique_size": {},
                                                                 "global_efficiency": {},
                                                                 "overall_reciprocity": {},
                                                                 "local_efficiency": {}}}))

my_pipe.add(PipelineElement('SVR'))
```

After setting up a desired pipeline, you will only have to fit it on your data.

```python
my_pipe.fit(matrices, labels)
```

## Evaluate pipeline

After fitting your pipeline you can now evaluate the pipeline by uploading your json results file to the photon explorer. You can also use your trained model to predict on different data.

```python
ext_data = load_conn('/path/to/your/other_data.mat')

my_pipe.predict(ext_data)
```

