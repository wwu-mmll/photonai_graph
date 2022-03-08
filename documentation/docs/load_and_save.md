PHOTONAI Graph offers methods for saving and loading networkx graphs.
Additionally, there are transformation functions for converting the graphs from one format into another.

!!! info
    If a specific format is required for your analysis, PHOTONAI Graph will try to automatically
    convert your input graphs.


## Loading graphs
Graphs can be loaded from different formats into networkx format.
From this networkx graph you can use different conversion functions to convert them into the 
desired format.

### Load networkx graph from file
::: photonai_graph.GraphConversions.load_file_to_networkx

## Saving graphs
Graphs can be saved as networkx 

### Graph to networkx file
::: photonai_graph.GraphConversions.save_networkx_to_file