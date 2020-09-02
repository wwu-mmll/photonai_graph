from photonai_graph.NeuralNets.dgl_base import DGLmodel
from photonai_graph.NeuralNets.NNModels import SGConvClassifier


class SGConvRegressorModel(DGLmodel):
    """
    Graph convolutional network for graph regression. Simple Graph
    convolutional layers from Wu, Felix, et al., 2018. Implementation
    based on dgl & pytorch.


    Parameters
    ----------
    * `in_dim` [int, default=1]:
        input dimension
    * `hidden_layers` [int, default=2]:
        number of hidden layers used by the model
    * `hidden_dim` [int, default=256]:
        dimensions in the hidden layers

    """

    def __init__(self,
                 in_dim: int = 1,
                 hidden_layers: int = 2,
                 hidden_dim: int = 256):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim

    def fit(self, X, y):

        # handle inputs
        X_trans = self.handle_inputs(X, self.adjacency_axis, self.feature_axis)
        # get data loader
        data_loader = self.get_data_loader(X_trans, y)
        # specify model with optimizer etc
        self.model = SGConvClassifier(self.in_dim, self.hidden_dim, 1, self.hidden_layers)
        # get optimizers
        loss_func, optimizer = self.get_regressor()
        # train model
        self.model.train()
        self.train_model(self.nn_epochs, self.model, optimizer, loss_func, data_loader)

        return self

    def predict(self, x):
        """returns the argmax of the predictions"""
        return self.predict_regressor(x)
