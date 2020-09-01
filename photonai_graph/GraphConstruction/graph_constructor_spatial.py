import os
import scipy
import numpy as np
import scipy.spatial
from photonai_graph.GraphConstruction.graph_constructor import GraphConstructor


class GraphConstructorSpatial(GraphConstructor):
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

    Example
    -------
        constructor = GraphConstructorSpatial(k_distance=7,
                                              transform_style="individual",
                                              atlas_name="ho_coords.csv",
                                              atlas_path="path/to/your/data/",
                                              fisher_transform=1,
                                              use_abs=1)
   """

    def __init__(self,
                 k_distance: int = 10,
                 atlas_name: str = 'ho',
                 atlas_folder: str = "",
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.k_distance = k_distance
        self.atlas_name = atlas_name
        self.atlas_folder = atlas_folder

    def get_spatial(self, X):
        """Returns the adjacency based on the spatial matrix"""
        # get atlas coords
        coords = self.get_atlas_coords(atlas_name=self.atlas_name, root_folder=self.atlas_folder)
        # generate adjacency matrix
        dist, idx = self.distance_scipy_spatial(coords, k=self.k_distance, metric='euclidean')
        adjacency = self.adjacency(dist, idx).astype(np.float32)
        # turn adjacency into numpy matrix for concatenation
        adjacency = adjacency.toarray()
        # repeat into desired length
        adjacency = np.repeat(adjacency[np.newaxis, :, :, np.newaxis], X.shape[0], axis=0)

        return adjacency

    def transform_test(self, X):
        """Transform input matrices accordingly"""
        adj, feat = self.get_mtrx(X)
        # do preparatory matrix transformations
        adj = self.prep_mtrx(adj)
        # threshold matrix
        adj = self.get_spatial(adj)
        # get feature matrix
        X_transformed = self.get_features(adj, feat)

        return X_transformed

    @staticmethod
    def distance_scipy_spatial(z, k, metric='euclidean'):
        """Compute exact pairwise distances."""
        d = scipy.spatial.distance.pdist(z, metric)
        d = scipy.spatial.distance.squareform(d)
        # k-NN photonai_graph.
        idx = np.argsort(d)[:, 1:k + 1]
        d.sort()
        d = d[:, 1:k + 1]

        return d, idx

    @staticmethod
    def get_atlas_coords(atlas_name, root_folder):
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
