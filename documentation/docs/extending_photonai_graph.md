Extending PHOTONAI Graph is equally easy as extending PHOTONAI itself.

For new Transformers see <a target="_blank" href="https://wwu-mmll.github.io/photonai/api/custom_transformer/">adding Transformers</a>.

For new Estimators see <a target="_blank", href="https://wwu-mmll.github.io/photonai/api/custom_estimator/">adding Estimators</a>

## PHOTONAI Graph specific base classes

PHOTONAI Graph offers a few base classes for simple integration.

To extend PHOTONAI Graph simply inherit from the desired base class
and register your custom model to PHOTONAI.

### GraphConstructor Base Class
::: photonai_graph.GraphConstruction.GraphConstructor.__init__
    rendering:
        show_root_toc_entry: False

### DGL Base Class - Classification
::: photonai_graph.NeuralNets.dgl_base.DGLClassifierBaseModel.__init__
    rendering:
        show_root_toc_entry: False

### DGL Base Class - Regression
::: photonai_graph.NeuralNets.dgl_base.DGLRegressorBaseModel.__init__