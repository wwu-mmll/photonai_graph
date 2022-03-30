import os
import scipy
import numpy as np
import scipy.spatial
from photonai_graph.GraphConstruction.graph_constructor import GraphConstructor


class GraphConstructorSpatial(GraphConstructor):
    _estimator_type = "transformer"

    def __init__(self,
                 k_distance: int = 10,
                 atlas_name: str = 'ho',
                 atlas_folder: str = "",
                 one_hot_nodes: int = 0,
                 use_abs: int = 0,
                 fisher_transform: int = 0,
                 use_abs_fisher: int = 0,
                 zscore: int = 0,
                 use_abs_zscore: int = 0,
                 adjacency_axis: int = 0,
                 logs: str = None):
        """
        Transformer class for generating adjacency matrices
        from connectivity matrices. Selects the k nearest
        neighbours for each node based on spatial distance
        of the coordinates in the chosen atlas. This method
        can be applied to both DTI and rs-fMRI data.
        Adapted from Ktena et al, 2017.

        !!! danger
            Currently considered untested!
            See <a href='https://github.com/wwu-mmll/photonai_graph/issues/65' target='_blank'>Ticket</a>

        Parameters
        ----------
        k_distance: int
            the k nearest neighbours value, for the kNN algorithm.
        atlas_name: str,default="ho"
            name of the atlas coordinate file
        atlas_folder: str,default="ho"
            path to the atlas coordinate file
        one_hot_nodes: int,default=0
            Whether to generate a one hot encoding of the nodes in the matrix (1) or not (0)
        use_abs: bool, default = False
            whether to convert all matrix values to absolute values before applying
            other transformations
        fisher_transform: int,default=0
            whether to perform a fisher transform of each matrix (1) or not (0)
        use_abs_fisher: int,default=0
            changes the values to absolute values. Is applied after fisher transform and before z-score transformation
        zscore: int,default=0
            performs a zscore transformation of the data. Applied after fisher transform and np_abs
        use_abs_zscore: int,default=0
            whether to use the absolute values of the z-score transformation or allow for negative values
        adjacency_axis: int,default=0
            position of the adjacency matrix, default being zero
        logs: str, default=None
            Path to the log data

        Example
        -------
        Use outside of a PHOTON pipeline

        ```python
        constructor = GraphConstructorSpatial(k_distance=7,
                                              atlas_name="ho_coords.csv",
                                              atlas_path="path/to/your/data/",
                                              fisher_transform=1,
                                              use_abs=1)
        ```

        Or as part of a pipeline

        ```python
        my_pipe.add(PipelineElement('GraphConstructorSpatial',
                                    hyperparameters={'k_distance': 7,
                                    'atlas_name': "ho_coords.csv", 'atlas_path': "path/to/your/data/"}))
        ```
       """
        super(GraphConstructorSpatial, self).__init__(one_hot_nodes=one_hot_nodes,
                                                      use_abs=use_abs,
                                                      fisher_transform=fisher_transform,
                                                      use_abs_fisher=use_abs_fisher,
                                                      zscore=zscore,
                                                      use_abs_zscore=use_abs_zscore,
                                                      adjacency_axis=adjacency_axis,
                                                      logs=logs)
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

    def transform(self, X) -> np.ndarray:
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
        coords_file = os.path.join(root_folder, atlas_name + '_coords.csv')
        coords = np.loadtxt(coords_file, delimiter=',')

        if atlas_name == 'ho':
            coords = np.delete(coords, 82, axis=0)

        return coords
