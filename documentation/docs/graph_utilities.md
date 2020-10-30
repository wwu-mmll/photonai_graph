# Graph Utility functions

The graph utility functions module is a set of utility functions when working with graphs. They contain functions for plotting graphs, converting them and checking for certain properties (for example: checking if the graph asteroidal). The module is split into two main parts: conversion functions, which handle conversions between different graph formats and non-conversion functions (saving/writing graphs, plotting them, ...). These functions can also be used outside of the PHOTONAI Graph framework to handle conversions or check graph layouts.

## Graph Utility Functions

#### draw_connectogram

Creates a connectogram plot from a networkx graph.

> ```python
> draw_connectogram(graph, edge_rad=None, colorscheme=None, nodesize=None, node_shape='o', weight=None, path=None, show=True)
> ```

> Parameters

* graph : *(nx.class.graph.Graph)* input graph, a single networkx graph
            
* edge_rad : *(str, default=None)* edge radius, controlling the curvature of the drawn edges

* colorscheme : colormap for drawing the connectogram

* nodesize : *(int, default=None)* controls size of the drawn nodes

* node_shape : *(str, default='o')* shape of the drawn nodes

* weight : *(float, default=None)* threshold below which edges are coloured differently than above

* path : *(str, default=None)* path where to save the plots as string, if no path is declared, plots are not saved. Path needs to be the full path including file name and ending, unlike in draw_connectograms.

* show : *(bool, default=True)* whether to plot the graph or not. Set it to false in headless environments
        
	
	
#### draw_connectivity_matrix

Creates a matplotlib plot from a connectivity matrix.

> ```python
> draw_connectivity_matrix(matrix, colorbar=False, adjacency_axis=None)
> ```

> Parameters

* matrix : *(numpy.ndarray, numpy.matrix or a list of those)* the input matrix or matrices from which to draw the connectivity matrix
            
* colorbar : *(boolean, default=False)* Whether to use a colorbar in the drawn plot
	
* adjacency_axis : *(int, default=None)* position of the the adjacency axis, if specified the array is assumed to have an additional axis where the matrix is stored.
