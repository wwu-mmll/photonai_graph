# Workflow

The starting point for PHOTON Graph can be either connectivity matrices or data that is already in a graph format (networkx, dgl, sparse/dense adjacency matrices). Depending on your starting point, there are different ways in which you will you will have set up your pipeline. In the case that you have connectivity matrices, you will need to use graph constructors to turn those connectivity matrices into adjacency matrices. After can choose from different options of doing machine learning with your graph data.

### Connectivity matrices

If you have connectivity matrices as your starting point, these might be noisy and densely connected, as for example in the case of resting state functional connectivity in the area of neuroscience. In order to reduce the amount connections and possibly reduce noise, one could threshold the graph so weak connections will be discarded. This is not the only possible way to construct adjacency matrices, and many more methods have been implemented, which can be found in the graph constructor section.

After transforming your matrix, using a graph constructor you can then use this matrix to do machine learning with it.

### Machine Learning on Graphs

Once you have a graph structure, you can then use this graph structure to do machine learning on it in a variety of ways. One option would be to extract graph measures and use these graph measures to do classical machine learning on them. The measures preserve graph information, that would be lost if only looking at node values for example. Depending on the measure it might contain global or local graph information. A similar idea applies to graph embeddings and kernels. They provide lower-dimensional representations of the graph structure, while still preserving graph information. The resulting embedding/kernel transformation can then be used to do classical machine learning.
In contrast Graph Neural Nets are neural networks modified, so that they will make use of the graph information directly. With a variety of networks available...
