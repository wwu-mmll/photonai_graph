# Graph Utility functions

The graph utility functions module is a set of utility functions when working with graphs. They contain functions for plotting graphs, converting them and checking for certain properties (for example: checking if the graph asteroidal). The module is split into two main parts: conversion functions, which handle conversions between different graph formats and non-conversion functions (saving/writing graphs, plotting them, ...). These functions can also be used outside of the PHOTONAI Graph framework to handle conversions or check graph layouts.

## Graph Utility Functions

#### draw_connectivity_matrix

> ```python
> draw_connectivity_matrix(matrix, colorbar=False, adjacency_axis=None)
> ```

>**Parameters**

 matrix : *(numpy.ndarray, numpy.matrix or a list of those)* the input matrix or matrices from which to draw the connectivity matrix
            
 colorbar : boolean, default=False
            Whether to use a colorbar in the drawn plot
	
	adjacency_axis : int, default=None
	    position of the the adjacency axis, if specified the array is assumed to
	    have an additional axis where the matrix is stored.
