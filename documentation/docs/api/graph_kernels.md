# Graph Kernels

Graph Kernels are learnable functions that map the graph structure into a lower dimensional representation and are commonly used to solve different problems in graph classification, or regression. In photonai-graph we utilize the grakel graph kernel package. We provide an adapter function that transforms connectivity matrices or networkx graphs into grakel conform graph objects.

## GrakelAdapter
::: photonai_graph.GraphKernels.GrakelAdapter.__init__
    rendering:
        show_root_toc_entry: False

## Grakel Kernels

The available graph kernels include all currently available grakel graph kernels (see https://ysig.github.io/GraKeL/0.1a7/graph_kernel.html). Please make sure that you choose the right graph structure meaning that you supply node and edge labels or attributes are required by the specific kernel you are using. A guide on the recommended labels and features can be found in the link above.

For node labels set node_feature_construction to "mean", "sum" or "degree", for node attributes set it to "features".
