# Graph Kernels

Graph Kernels are learnable functions that map the graph structure into a lower dimensional representation and are commonly used to solve different problems in graph classification, or regression. In photonai-graph we utilize the grakel graph kernel package. We provide an adapter function that transforms connectivity matrices or networkx graphs into grakel conform graph objects.

## GrakelAdapter

The GrakelAdapter is a transformer that can be inserted into a pipeline, to transform connectivity matrices or networkx graphs into grakel graph objects.

| Parameter | type | Description |
| -----     | ----- | ----- |
| input_type | str, default="dense" | the type of input to be converted, dense or networkx |
| node_labels | list, default=None | list of node labels if graphs are constructed from networkx graphs |
| edge_labels | list, default=None | list of edge labels if graphs are constructed from networkx graphs |
| node_feature_construction | str, default="mean" | mode of feature construction for graphs constructed from adjacency matrices |
| adjacency_axis | int, default=0 | position of the adjacency axis, default being 0 |
| feature_axis | int, default=1 | position of the feature axis, default being 1 |
