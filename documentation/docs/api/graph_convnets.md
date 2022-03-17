# Graph Neural Networks

Graph Neural Networks are a class of neural networks, optimized for deep learning on graphs. They provide an end-to-end solution for machine learning on graphs, unlike graph kernels/embeddings where a transformation step is applied before a "classical" machine learning algorithm. As they have attracted more attention in the recent years, a range of different architectures for this class has sprung up ranging from the Graph Convolutional Network (GCN, Kipf et al. 2017) to Graph Attention Networks (citation needed). The different architectures learn from the graph and it's overall structure, making use of the graph information unlike classical neural networks.

The graph neural network module of PHOTONAI Graph provides a variety of customizable out-of-the-box graph neural networks. They can be instantiated in one line of code and easily integrate into PHOTONAI pipelines.

## Graph Neural Network Module

The Graph Neural Network module consists of three parts. The Layer Module, where different layers are implemented and the message-passing steps of these are defined. The Model module where the module is constructed as a class (see pytorch neural networks). And the GraphConvNet module which calls the models and implements fit and transform steps, making them sklearn conform. This module also handles data conversions, converting graphs to the right format for the networks, which are written in pytorch.

You can also write your own custom graph neural network architecture, and register them via the PHOTON register function (link here). When writing your own custom neural nets you are free to choose your own package, as long as they implement fit, transform and predict functions like the GraphConvNet module classes. These can also be used as a blueprint if you want to integrate your own graph neural network architectures into PHOTONAI.

## DglModel
::: photonai_graph.NeuralNets.dgl_base.DGLModel.__init__
    rendering:
        show_root_toc_entry: False

## DGLRegressorBaseModel
::: photonai_graph.NeuralNets.dgl_base.DGLRegressorBaseModel.__init__
    rendering:
        show_root_toc_entry: False

## DGLClassifierBaseModel
::: photonai_graph.NeuralNets.dgl_base.DGLClassifierBaseModel.__init__

## GCNClassifierModel
::: photonai_graph.NeuralNets.GCNModel.GCNClassifierModel.__init__
    rendering:
        show_root_toc_entry: False


## SGConvClassifierModel
::: photonai_graph.NeuralNets.SGCModel.SGConvClassifierModel.__init__
    rendering:
        show_root_toc_entry: False


## GATClassifierModel

::: photonai_graph.NeuralNets.GATModel.GATClassifierModel.__init__
    rendering:
        show_root_toc_entry: False


## GCNRegressorModel
::: photonai_graph.NeuralNets.GCNModel.GCNRegressorModel.__init__
    rendering:
        show_root_toc_entry: False

## SGConvRegressorModel
::: photonai_graph.NeuralNets.SGCModel.SGConvRegressorModel.__init__
    rendering:
        show_root_toc_entry: False

## GATRegressorModel
::: photonai_graph.NeuralNets.GATModel.GATRegressorModel.__init__
    rendering:
        show_root_toc_entry: False