# Getting started

In order to get started, you will need connectivity matrices or data, that is already graph data. 
For this introduction we will assume that you are using connectivity matrices.

!!! info "Connectivity matrix"
    Connectivity matrices are simply adjacency matrics.
    Most commonly those matrices have to be thresholded with a static threshold or a percentage.
    If your input  adjacency matrices also require a thresholding, you can also simply use the
    Thresholding technique described below.

## 1. Loading data
As this toolbox is build with neuro scientists in mind, the first example will cover the case of connectivity matrices.
However, PHOTONAI Graph is not limited to connectivity matrices.

If you are using matlab files, from NBS, CONN or other popular neuroimaging connectivity toolboxes, you will have to 
import them. An example function on how to import matlab data matrices is contained in PHOTONAI Graph.
We cannot guarante that this function is correctly loading your data. If in doubt please write your own importer.

!!! info 
    If you are using dense matrices, photonai-graph assumes that they have a certain shape: 
    They should come as numpy matrices or array with the dimensions *Subjects x nodes x nodes x modalities (optional)*. 

With such a function you can then load your data:

```python
from photonai_graph.GraphUtilities import load_conn

matrices = load_conn('/path/to/your/data.mat')
```

!!! danger
    Please make sure the imported data has the exact structure you are expecting.
    We cannot guarantee to load your data correctly under any circumstances. If in doubt
    write an importer for your own data.

## 2. Load your labels

Now you can load your labels. Make sure that your labels match the number of subjects.

```python
import pandas as pd

df = pd.read_csv('/path/to/labels.csv')
labels = df['your_label']
```

## 3. Set up pipeline

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

!!! info
    Since PHOTONAI Graph is based on PHOTONAI, a nested cross validation is evaluated by default. You are also able to 
    add more transformers and estimators already contained in PHOTONAI. For more information pleas read the documentation
    of [PHOTONAI](https://wwu-mmll.github.io/photonai/).

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

## 4. Evaluate pipeline

Since PHOTONAI Graph is based on PHOTONAI, you can use the inspection and visualization functions of PHOTONAI.
These are documented <a target='_blank' href='https://wwu-mmll.github.io/photonai/getting_started/output/'>here</a>.

Especially uploading the generated `photon_result_file.json` to the <a target='_blank' href='https://explorer.photon-ai.com/'>PHOTONAI Explorer</a>
can become very handy when evaluating your models.
 You can also use your trained model to predict on different data.

```python
ext_data = load_conn('/path/to/your/other_data.mat')

my_pipe.predict(ext_data)
```
