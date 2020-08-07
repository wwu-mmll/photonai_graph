import os
import scipy
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .abc_graph_constructor_adjacency import GraphConstructorAdjacency


class GraphConstructorSpatial(BaseEstimator, TransformerMixin, GraphConstructorAdjacency):
    _estimator_type = "transformer"

    """
    Transformer class for generating adjacency matrices 
    from connectivity matrices. Selects the k nearest
    neighbours for each node based on spatial distance
    of the coordinates in the chosen atlas.
    Adapted from Ktena et al, 2017.


    Parameters
    ----------
    * `k_distance` [int]:
        the k nearest neighbours value, for the kNN algorithm.
    * `transform_style` [str, default="mean"]:
        generate an adjacency matrix based on the mean matrix like in Ktena et al.: "mean" 
        Or generate a different matrix for every individual: "individual"
    * `atlas_name` [str, default="ho"]:
        name of the atlas coordinate file
    * `atlas_path` [str, default="ho"]:
        path to the atlas coordinate file
    * `adjacency_axis` [int]:
        position of the adjacency matrix, default being zero
    * `one_hot_nodes` [int]:
        Whether to generate a one hot encoding of the nodes in the matrix.
    * `return_adjacency_only` [int]:
        whether to return the adjacency matrix only (1) or also a feature matrix (0)
    * `verbosity` [int, default=0]:
        The level of verbosity, 0 is least talkative and gives only warn and error, 1 gives adds info and 2 adds debug
    * `logs` [str, default='']:
        Path to the log data

    Example
    -------
        constructor = GraphConstructorSpatial(k_distance=7,
                                              transform_style="individual",
                                              atlas_name="ho_coords.csv",
                                              atlas_path="path/to/your/data/",
                                              fisher_transform=1,
                                              use_abs=1)
   """

    def __init__(self, k_distance=10,
                 atlas_name='ho', atlas_folder="", logs=''):
        self.k_distance = k_distance
        self.atlas_name = atlas_name
        self.atlas_folder = atlas_folder
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        # todo: is this function really necessary?
        pass

    def distance_scipy_spatial(self, z, k, metric='euclidean'):
        # todo: check if this function could be static
        """Compute exact pairwise distances."""
        d = scipy.spatial.distance.pdist(z, metric)
        d = scipy.spatial.distance.squareform(d)
        # k-NN photonai_graph.
        idx = np.argsort(d)[:, 1:k + 1]
        d.sort()
        d = d[:, 1:k + 1]

        return d, idx

    def get_atlas_coords(self, atlas_name, root_folder):
        # todo: check if this function could be static
        """
            atlas_name   : name of the atlas used
        returns:
            matrix       : matrix of roi 3D coordinates in MNI space (num_rois x 3)
        """
        root_folder = root_folder
        coords_file = os.path.join(root_folder, atlas_name + '_coords.csv')
        coords = np.loadtxt(coords_file, delimiter=',')

        if atlas_name == 'ho':
            coords = np.delete(coords, 82, axis=0)

        return coords

    def transform(self, X):

        # todo: X_mean is never used. Do we really need to compute this?
        # use the mean 2d image of all samples for creating the different photonai_graph structures
        X_mean = np.squeeze(np.mean(X, axis=0))

        # get atlas coords
        coords = self.get_atlas_coords(atlas_name=self.atlas_name, root_folder=self.atlas_folder)

        # generate adjacency matrix
        dist, idx = self.distance_scipy_spatial(coords, k=10, metric='euclidean')
        adjacency = self.adjacency(dist, idx).astype(np.float32)

        # turn adjacency into numpy matrix for concatenation
        adjacency = adjacency.toarray()

        X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], -1))
        # X = X[..., None] + adjacency[None, None, :] #use broadcasting to speed up computation
        adjacency = np.repeat(adjacency[np.newaxis, :, :, np.newaxis], X.shape[0], axis=0)
        X = np.concatenate(adjacency, X, axis=3)

        return X
