from stellargraph.layer import DeepGraphCNN
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPool1D, Flatten
from tensorflow.keras.losses import binary_crossentropy


from photonai_graph.GraphConvNet.abc_graph_net import GraphNet


class DeepGraphCNN_Classifier(GraphNet):

    def __init__(self, multi_class: bool = True,
                 hidden_layer_sizes: list = None,
                 GCN_layer_sizes=None,
                 learning_rate: float = 0.005,
                 loss: str = "binary_crossentropy",
                 epochs: int = 200,
                 folds: int = 10,
                 n_repeats: int = 5,
                 nn_batch_size: int =64,
                 metrics: list = None,
                 callbacks: list = None,
                 verbosity=1,
                 dropout_rate=0.5,  # list or float
                 activations='tanh',  # list or str
                 optimizer="adam"):  # list or keras.optimizer
        super().__init__()
        if GCN_layer_sizes is None:
            GCN_layer_sizes = [32, 32, 32, 1]
        self.GCN_layer_sizes = GCN_layer_sizes
        self.learning_rate = learning_rate
        self._loss = ""
        self._multi_class = None
        self.loss = loss
        self.multi_class = multi_class
        self.epochs =epochs
        self.folds = folds
        self.n_repeats = n_repeats
        self.nn_batch_size = nn_batch_size
        self.dropout_rate = dropout_rate
        self.activations = activations

        if callbacks:
            self.callbacks = callbacks
        else:
            self.callbacks = []

        if not metrics:
            metrics = ['accuracy']

        self.model = None

    def create_graph_classification_model(self, generator):
        model = DeepGraphCNN(
            layer_sizes=self.GCN_layer_sizes,
            activations=[self.activations]*len(self.GCN_layer_sizes),
            generator=generator,
            k=30
        )
        x_inp, x_out = model.in_out_tensors()

        x_out = Conv1D(filters=16, kernel_size=97, strides=97)(x_out)
        x_out = MaxPool1D(pool_size=2)(x_out)
        x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)
        x_out = Flatten()(x_out)
        x_out = Dense(units=128, activation="relu")(x_out)
        x_out = Dropout(rate=self.dropout_rate)(x_out)
        predictions = Dense(units=1, activation="sigmoid")(x_out)

        model = Model(inputs=x_inp, outputs=predictions)
        # Let's create the Keras model and prepare it for training
        model = Model(inputs=x_inp, outputs=predictions)
        model.compile(optimizer=Adam(self.learning_rate), loss=binary_crossentropy, metrics=["acc"])

        return model

    def train_fold(self, model, train_gen, test_gen, es, epochs):
        history = model.fit(train_gen, epochs=epochs, validation_data=test_gen, verbose=0, callbacks=[es])
        # calculate performance on the test data and return along with history
        test_metrics = model.evaluate(test_gen, verbose=0)
        test_acc = test_metrics[model.metrics_names.index("acc")]

        return history, test_acc

    @staticmethod
    def get_generators(train_data, test_data, train_labels, test_labels, batch_size, generator):
        train_gen = generator.flow(train_data, targets=train_labels, batch_size=batch_size)
        test_gen = generator.flow(test_data, targets=test_labels, batch_size=batch_size)

        return train_gen, test_gen

    def fit(self, X, y):
        # encode targets the right way
        graph_labels = self.encode_targets(y)
        super().fit(X, graph_labels)

    @staticmethod
    def encode_targets(y, type="sigmoid"):
        y_labels = y.copy()
        if type == "sigmoid":
            y_labels[y_labels == 0] = -1
            y_labels[y_labels == 1] = 1
        elif type == "multiclass":
            y_labels = y_labels

        return y_labels
