from stellargraph.layer import GCNSupervisedGraphClassification
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import mse


from photonai_graph.GraphConvNet.abc_graph_net import GraphNet


class GraphConvNet_Regressor(GraphNet):

    def __init__(self, multi_class: bool = True,
                 hidden_layer_sizes: list = None,
                 gcn_layer_sizes=None,
                 learning_rate: float = 0.005,
                 loss: str = "mse",
                 epochs: int = 200,
                 folds: int = 10,
                 n_repeats: int = 5,
                 nn_batch_size: int = 64,
                 metrics: list = None,
                 callbacks: list = None,
                 verbosity=1,
                 dropout_rate=0.5,  # list or float
                 activations='relu',  # list or str
                 optimizer="adam"):  # list or keras.optimizer
        super().__init__()
        # Default variables should not be mutable.
        if gcn_layer_sizes is None:
            gcn_layer_sizes = [64, 64]
        self.GCN_layer_sizes = gcn_layer_sizes
        self.learning_rate = learning_rate
        self._loss = ""
        self._multi_class = None
        self.loss = loss
        self.multi_class = multi_class
        self.epochs = epochs
        self.folds = folds
        self.n_repeats = n_repeats
        self.nn_batch_size = nn_batch_size
        self.dropout_rate = dropout_rate

        if callbacks:
            self.callbacks = callbacks
        else:
            self.callbacks = []

        if not metrics:
            metrics = ['accuracy']

        self.model = None

    def create_graph_classification_model(self, generator):
        gc_model = GCNSupervisedGraphClassification(
            layer_sizes=self.GCN_layer_sizes,
            activations=["relu"] * len(self.GCN_layer_sizes),
            generator=generator,
            dropout=self.dropout_rate,
        )
        x_inp, x_out = gc_model.in_out_tensors()
        predictions = Dense(units=32, activation="relu")(x_out)
        predictions = Dense(units=16, activation="relu")(predictions)
        predictions = Dense(units=1)(predictions)

        # Let's create the Keras model and prepare it for training
        model = Model(inputs=x_inp, outputs=predictions)
        model.compile(optimizer=Adam(self.learning_rate), loss=mse, metrics=["mae", "mse"])

        return model

    def train_fold(self, model, train_gen, test_gen, es, epochs):
        history = model.fit(train_gen, epochs=epochs, validation_data=test_gen, verbose=0, callbacks=[es])
        # calculate performance on the test data and return along with history
        test_metrics = model.evaluate(test_gen, verbose=0)
        test_acc = test_metrics[model.metrics_names.index("mae")]

        return history, test_acc

    @staticmethod
    def get_generators(train_data, test_data, train_labels, test_labels, batch_size, generator):
        train_gen = generator.flow(train_data, targets=train_labels, batch_size=batch_size)
        test_gen = generator.flow(test_data, targets=test_labels, batch_size=batch_size)

        return train_gen, test_gen
