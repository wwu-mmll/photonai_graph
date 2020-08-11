# Graph Construction

The Graph Construction module is a collection of transformer classes that construct adjacency matrices for graphs based on connectivity matrices. This includes well established methods like thresholding the connectivity matrix, but also allows for more complex adjacency matrix construction methods like kNN-based matrix construction. The adjacency matrices that the transform method of each class returns can then be converted into different formats like networkx, scipy sparse or dgl graphs.

Depending on the type of data you are working with, you might have very noisy data. Or you might be interested only in weaker connections. Or you want the same adjacency matrix for every graph. Transforming your connectivity matrix into an adjacency matrix of your choice can be achieved by the different graph constructor classes, which implement different transformations of the connectivity matrix. In the case of noisy data, thresholding the connections might reduce noise and increase computational speed, by having to evaluate less edges later on. Picking a threshold or percentage window allows you to discard other connections, focusing on the information that is contained in the connections that fall in your window range. And creating the same adjacency matrix for every graph, with only different node features might allow you to use graph algorithms you might otherwise not be able to use.

## Class - GraphConstructor

Base class inherited by all Graph Constructors. Implements the fit method, which initializes the mean matrix, for mean matrix based graph construction.

| Parameter | type | Description |
| -----     | ----- | ----- |
| transform_style | str, default="mean" | generate an adjacency matrix based on the mean matrix like in Ktena et al.: "mean"; or generate a different matrix for every individual: "individual" |
| adjacency_axis | int, default=0 | position of the adjacency matrix, default being zero |


## Class - GraphConstructorThreshold

Transformer class for generating adjacency matrices from connectivity matrices. Thresholds matrix based on a chosen threshold value.

| Parameter | type | Description |
| -----     | ----- | ----- |
| threshold | float | threshold value below which to set matrix entries to zero |
| adjacency_axis | int | position of the adjacency matrix, default being zero |
| concatentation_axis | int | axis along which to concatenate the adjacency and feature matrix |
| one_hot_nodes | int | Whether to generate a one hot encoding of the nodes in the matrix (1) or not (0) |
| return_adjacency_only | int | whether to return the adjacency matrix only (1) or also a feature matrix (0) |
| fisher_transform | int | whether to perform a fisher transform of each matrix (1) or not (0) | 
| use_abs | int, default=0 | changes the values to absolute values. Is applied after fisher transform and before z-score transformation |
| zscore | int, default=0 | performs a zscore transformation of the data. Applied after fisher transform and np_abs |
| use_abs_zscore | int, default=0 | whether to use the absolute values of the z-score transformation or allow for negative values |
| verbosity | int, default=0 | The level of verbosity, 0 is least talkative and gives only warn and error, 1 gives adds info and 2 adds debug |

#### Example

Use outside of a PHOTON pipeline

```python
constructor = GraphConstructorThreshold(threshold=0.5,
                            		fisher_transform=1,
			    		use_abs=1)
```

Or as part of a pipeline

```python
my_pipe.add(PipelineElement('GraphConstructorThreshold',
                            hyperparameters={'threshold': 0.5}))
```
   
## Class - GraphConstructorPercentage

Transformer class for generating adjacency matrices from connectivity matrices. Selects the top x percent of connections and sets all other connections to zero.

| Parameter | type | Description |
| -----     | ----- | ----- |
| percentage | float | value of percent of connections to discard. A value of 0.9 keeps only the top 10% |
| adjacency_axis | int | position of the adjacency matrix, default being zero |
| concatentation_axis | int | axis along which to concatenate the adjacency and feature matrix |
| one_hot_nodes | int | Whether to generate a one hot encoding of the nodes in the matrix (1) or not (0) |
| return_adjacency_only | int | whether to return the adjacency matrix only (1) or also a feature matrix (0) |
| fisher_transform | int | whether to perform a fisher transform of each matrix (1) or not (0) | 
| use_abs | int, default=0 | changes the values to absolute values. Is applied after fisher transform and before z-score transformation |
| zscore | int, default=0 | performs a zscore transformation of the data. Applied after fisher transform and np_abs |
| use_abs_zscore | int, default=0 | whether to use the absolute values of the z-score transformation or allow for negative values |
| verbosity | int, default=0 | The level of verbosity, 0 is least talkative and gives only warn and error, 1 gives adds info and 2 adds debug |

#### Example

Use outside of a PHOTON pipeline

```python
constructor = GraphConstructorPercentage(percentage=0.9,
                            		 fisher_transform=1,
			    		 use_abs=1)
```

Or as part of a pipeline

```python
my_pipe.add(PipelineElement('GraphConstructorPercentage',
                            hyperparameters={'percentage': 0.9}))
```


## Class - GraphConstructorThresholdWindow

Transformer class for generating adjacency matrices from connectivity matrices. Thresholds matrix based on a chosen threshold window. Values outside this threshold window will be set to zero.

| Parameter | type | Description |
| -----     | ----- | ----- |
| threshold_upper | float | upper limit of the threshold window |
| threshold_lower | float | lower limit of the threshold window |
| adjacency_axis | int | position of the adjacency matrix, default being zero |
| concatentation_axis | int | axis along which to concatenate the adjacency and feature matrix |
| one_hot_nodes | int | Whether to generate a one hot encoding of the nodes in the matrix (1) or not (0) |
| return_adjacency_only | int | whether to return the adjacency matrix only (1) or also a feature matrix (0) |
| fisher_transform | int | whether to perform a fisher transform of each matrix (1) or not (0) | 
| use_abs | int, default=0 | changes the values to absolute values. Is applied after fisher transform and before z-score transformation |
| zscore | int, default=0 | performs a zscore transformation of the data. Applied after fisher transform and np_abs |
| use_abs_zscore | int, default=0 | whether to use the absolute values of the z-score transformation or allow for negative values |
| verbosity | int, default=0 | The level of verbosity, 0 is least talkative and gives only warn and error, 1 gives adds info and 2 adds debug |

#### Example

Use outside of a PHOTON pipeline

```python
constructor = GraphConstructorThresholdWindow(threshold_upper=0.7,
                            		      threshold_lower=0.3,
			    		      use_abs=1)
```

Or as part of a pipeline

```python
my_pipe.add(PipelineElement('GraphConstructorThresholdWindow',
                            hyperparameters={'threshold_upper': 0.7, 'threshold_lower': 0.3}))
```


## Class - GraphConstructorPercentageWindow


Transformer class for generating adjacency matrices from connectivity matrices. Selects a window of the x1-th percentile to the x2-th percentile of connections and sets all other connections to zero.

| Parameter | type | Description |
| -----     | ----- | ----- |
| percentage_upper | float | upper limit of the percentage window |
| percentage_lower | float | lower limit of the percentage window |
| adjacency_axis | int | position of the adjacency matrix, default being zero |
| concatentation_axis | int | axis along which to concatenate the adjacency and feature matrix |
| one_hot_nodes | int | Whether to generate a one hot encoding of the nodes in the matrix (1) or not (0) |
| return_adjacency_only | int | whether to return the adjacency matrix only (1) or also a feature matrix (0) |
| fisher_transform | int | whether to perform a fisher transform of each matrix (1) or not (0) | 
| use_abs | int, default=0 | changes the values to absolute values. Is applied after fisher transform and before z-score transformation |
| zscore | int, default=0 | performs a zscore transformation of the data. Applied after fisher transform and np_abs |
| use_abs_zscore | int, default=0 | whether to use the absolute values of the z-score transformation or allow for negative values |
| verbosity | int, default=0 | The level of verbosity, 0 is least talkative and gives only warn and error, 1 gives adds info and 2 adds debug |

#### Example

Use outside of a PHOTON pipeline

```python
constructor = GraphConstructorPercentageWindow(percentage_upper=0.9,
                                               percentage_lower=0.7
                            		       fisher_transform=1,
			    		       use_abs=1)
```

Or as part of a pipeline

```python
my_pipe.add(PipelineElement('GraphConstructorPercentageWindow',
                            hyperparameters={'percentage_upper': 0.9, 'percentage_lower': 0.7}))
```


## Class - GraphConstructorKNN

Transformer class for generating adjacency matrices from connectivity matrices. Selects the k nearest neighbours for each node based on pairwise distance. Recommended for functional connectivity. Adapted from Ktena et al, 2017.

| Parameter | type | Description |
| -----     | ----- | ----- |
| k_distance | int | the k nearest neighbours value, for the kNN algorithm. |
| adjacency_axis | int | position of the adjacency matrix, default being zero |
| one_hot_nodes | int | Whether to generate a one hot encoding of the nodes in the matrix (1) or not (0) |
| return_adjacency_only | int | whether to return the adjacency matrix only (1) or also a feature matrix (0) |
| fisher_transform | int | whether to perform a fisher transform of each matrix (1) or not (0) | 
| use_abs | int, default=0 | changes the values to absolute values. Is applied after fisher transform and before z-score transformation |
| zscore | int, default=0 | performs a zscore transformation of the data. Applied after fisher transform and np_abs |
| use_abs_zscore | int, default=0 | whether to use the absolute values of the z-score transformation or allow for negative values |
| verbosity | int, default=0 | The level of verbosity, 0 is least talkative and gives only warn and error, 1 gives adds info and 2 adds debug |

#### Example

Use outside of a PHOTON pipeline

```python
constructor = GraphConstructorKNN(k_distance=6,
                            	  fisher_transform=1,
			    	  use_abs=1)
```

Or as part of a pipeline

```python
my_pipe.add(PipelineElement('GraphConstructorKNN',
                            hyperparameters={'k_distance': 6}))
```


## Class - GraphConstructorSpatial

Transformer class for generating adjacency matrices from connectivity matrices. Selects the k nearest neighbours for each node based on spatial distance of the coordinates in the chosen atlas. Adapted from Ktena et al, 2017.

| Parameter | type | Description |
| -----     | ----- | ----- |
| k_distance | int | the k nearest neighbours value, for the kNN algorithm. |
| atlas_name | str | name of the atlas coordinate file |
| atlas_path | str | path to the atlas coordinate file |
| adjacency_axis | int | position of the adjacency matrix, default being zero |
| one_hot_nodes | int | Whether to generate a one hot encoding of the nodes in the matrix (1) or not (0) |
| return_adjacency_only | int | whether to return the adjacency matrix only (1) or also a feature matrix (0) |
| fisher_transform | int | whether to perform a fisher transform of each matrix (1) or not (0) | 
| use_abs | int, default=0 | changes the values to absolute values. Is applied after fisher transform and before z-score transformation |
| zscore | int, default=0 | performs a zscore transformation of the data. Applied after fisher transform and np_abs |
| use_abs_zscore | int, default=0 | whether to use the absolute values of the z-score transformation or allow for negative values |
| verbosity | int, default=0 | The level of verbosity, 0 is least talkative and gives only warn and error, 1 gives adds info and 2 adds debug |

#### Example

Use outside of a PHOTON pipeline

```python
constructor = GraphConstructorSpatial(k_distance=7,
				      transform_style="individual",
				      atlas_name="ho_coords.csv",
				      atlas_path="path/to/your/data/",
                            	      fisher_transform=1,
			    	      use_abs=1)
```

Or as part of a pipeline

```python
my_pipe.add(PipelineElement('GraphConstructorSpatial',
                            hyperparameters={'k_distance': 7, 'transform_style': "individual", 
                            'atlas_name': "ho_coords.csv", 'atlas_path': "path/to/your/data/"}))
```

## Class - GraphConstructorRandomWalks

Transformer class for generating adjacency matrices from connectivity matrices. Generates a kNN matrix and performs random walks on these. The coocurrence of two nodes in those walks is then used to generate a higher-order adjacency matrix, by applying the kNN algorithm on the matrix again. Adapted from Ma et al., 2019.

| Parameter | type | Description |
| -----     | ----- | ----- |
| k_distance | int | the k nearest neighbours value, for the kNN algorithm. |
| transform_style | str | generate an adjacency matrix based on the mean matrix like in Ktena et al.: "mean" or per person "individual" |
| number_of_walks | int, default=10 | number of walks to take to sample the random walk matrix |
| walk_length | int, default=5 | length of the random walk, as the number of steps |
| window_size | int, default=5 | size of the sliding window from which to sample to coocurrence of two nodes |
| no_edge_weight | int, default=1 | whether to return an edge weight (0) or not (1) |
| adjacency_axis | int | position of the adjacency matrix, default being zero |
| one_hot_nodes | int | Whether to generate a one hot encoding of the nodes in the matrix (1) or not (0) |
| return_adjacency_only | int | whether to return the adjacency matrix only (1) or also a feature matrix (0) |
| fisher_transform | int | whether to perform a fisher transform of each matrix (1) or not (0) | 
| use_abs | int, default=0 | changes the values to absolute values. Is applied after fisher transform and before z-score transformation |
| zscore | int, default=0 | performs a zscore transformation of the data. Applied after fisher transform and np_abs |
| use_abs_zscore | int, default=0 | whether to use the absolute values of the z-score transformation or allow for negative values |
| verbosity | int, default=0 | The level of verbosity, 0 is least talkative and gives only warn and error, 1 gives adds info and 2 adds debug |

#### Example

Use outside of a PHOTON pipeline

```python
constructor = GraphConstructorRandomWalks(k_distance=5,
					  transform_style="individual",
					  number_of_walks=25,
                                          fisher_transform=1,
			    		  use_abs=1)
```

Or as part of a pipeline

```python
my_pipe.add(PipelineElement('GraphConstructorRandomWalks',
                            hyperparameters={'k_distance': 5, 'transform_style': "individual", 
                            'number_of_walks': 25}))
```


