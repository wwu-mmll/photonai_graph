"""
===========================================================
Project: PHOTON Graph
===========================================================
Description
-----------
A wrapper containing functions for turning connectivity matrices into photonai_graph structures

Version
-------
Created:        12-08-2019
Last updated:   04-05-2020


Author
------
Vincent Holstein
Translationale Psychiatrie
Universitaetsklinikum Muenster
"""

#TODO: "deposit" atlas coordinate files
#TODO: add advanced documentation for every method
#TODO: debug the copy steps, add a fisher transform for the connectivity matrix values

from photonai_graph.GraphUtilities import individual_ztransform, individual_fishertransform
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import sklearn
import scipy
import os
from itertools import islice, combinations


class GraphConstructorKNN(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    #constructs the adjacency matrix for the connectivity matrices by a kNN approach
    #adapted from Ktena et al., 2017

    def __init__(self, k_distance = 10, transform_style = "mean", adjacency_axis = 0, logs=''):
        self.k_distance = k_distance
        self.transform_style = transform_style
        self.adjacency_axis = adjacency_axis
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        pass

    def distance_sklearn_metrics(self, z, k, metric='euclidean'):
        """Compute exact pairwise distances."""
        d = sklearn.metrics.pairwise.pairwise_distances(
            z, metric=metric, n_jobs=-2)
        # k-NN photonai_graph.
        idx = np.argsort(d)[:, 1:k + 1]
        d.sort()
        d = d[:, 1:k + 1]
        return d, idx

    def adjacency(self, dist, idx):
        """Return the adjacency matrix of a kNN photonai_graph."""
        M, k = dist.shape
        assert M, k == idx.shape
        assert dist.min() >= 0

        # Weights.
        sigma2 = np.mean(dist[:, -1]) ** 2
        dist = np.exp(- dist ** 2 / sigma2)

        # Weight matrix.
        I = np.arange(0, M).repeat(k)
        J = idx.reshape(M * k)
        V = dist.reshape(M * k)
        W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))

        # No self-connections.
        W.setdiag(0)

        # Non-directed photonai_graph.
        bigger = W.T > W
        W = W - W.multiply(bigger) + W.T.multiply(bigger)

        assert W.nnz % 2 == 0
        assert np.abs(W - W.T).mean() < 1e-10
        assert type(W) is scipy.sparse.csr.csr_matrix

        return W

    def transform(self, X):
        # transform each photonai_graph through its own adjacency or all graphs
        if self.transform_style == "mean" or self.transform_style == "Mean":
            # use the mean 2d image of all samples for creating the different photonai_graph structures
            X_mean = np.squeeze(np.mean(X, axis=0))

            # select the proper matrix in case you have multiple
            if np.ndim(X_mean) == 3:
                X_mean = X_mean[:, :, self.adjacency_axis]
            elif np.ndim(X_mean) == 2:
                X_mean = X_mean
            else:
                raise ValueError('The input matrices need to have 3 or 4 dimensions. Please check your input matrix.')

            # generate adjacency matrix
            d, idx = self.distance_sklearn_metrics(X_mean, k=self.k_distance, metric='euclidean')
            adjacency = self.adjacency(d, idx).astype(np.float32)

            #turn adjacency into numpy matrix for concatenation
            adjacency = adjacency.toarray()

            X_transformed = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], -1))
            # X = X[..., None] + adjacency[None, None, :] #use broadcasting to speed up computation
            adjacency = np.repeat(adjacency[np.newaxis, :, :, np.newaxis], X.shape[0], axis=0)
            X_transformed = np.concatenate((adjacency, X_transformed), axis=3)

        elif self.transform_style == "individual" or self.transform_style == "Individual":
            # make a copy of x
            X_knn = X.copy()
            # select the proper matrix in case you have multiple
            if np.ndim(X_knn) == 4:
                X_knn = X_knn[:, :, :, self.adjacency_axis]
                X_features = X_knn.copy()
            elif np.ndim(X_knn) == 3:
                X_knn = X_knn
                X_features = X_knn.copy()
                X_features = X_features.reshape((X_features.shape[0], X_features.shape[1], X_features.shape[2], -1))
            else:
                raise ValueError('The input matrices need to have 3 or 4 dimensions. Please check your input matrix.')

            # make a list for all adjacency matrices
            adjacency_list = []
            # generate adjacency matrix for each individual
            for i in range(X_knn.shape[0]):
                d, idx = self.distance_sklearn_metrics(X_knn[i, :, :], k=self.k_distance, metric='euclidean')
                adjacency = self.adjacency(d, idx).astype(np.float32)

                # turn adjacency into numpy matrix for concatenation
                adjacency = adjacency.toarray()
                adjacency_list.append(adjacency)

            # X = X[..., None] + adjacency[None, None, :] #use broadcasting to speed up computation
            adjacency_list = np.asarray(adjacency_list)
            adjacency_list = adjacency_list.reshape((adjacency_list.shape[0], adjacency_list.shape[1], adjacency_list.shape[2], -1))
            X_transformed = np.concatenate((adjacency_list, X_features), axis=3)

        else:
            raise KeyError('Only mean and individual transform are supported. '
                           'Please check your spelling for the parameter transform_style.')

        return X_transformed



class GraphConstructorSpatial(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, k_distance = 10,
                 atlas_name = 'ho', atlas_folder = "", logs=''):
        self.k_distance = k_distance
        self.atlas_name = atlas_name
        self.atlas_folder = atlas_folder
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()


    def fit(self, X, y):
        pass


    def distance_scipy_spatial(self, z, k, metric='euclidean'):
        """Compute exact pairwise distances."""
        d = scipy.spatial.distance.pdist(z, metric)
        d = scipy.spatial.distance.squareform(d)
        # k-NN photonai_graph.
        idx = np.argsort(d)[:, 1:k + 1]
        d.sort()
        d = d[:, 1:k + 1]

        return d, idx


    def adjacency(self, dist, idx):
        """Return the adjacency matrix of a kNN photonai_graph."""
        M, k = dist.shape
        assert M, k == idx.shape
        assert dist.min() >= 0

        # Weights.
        sigma2 = np.mean(dist[:, -1]) ** 2
        dist = np.exp(- dist ** 2 / sigma2)

        # Weight matrix.
        I = np.arange(0, M).repeat(k)
        J = idx.reshape(M * k)
        V = dist.reshape(M * k)
        W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))

        # No self-connections.
        W.setdiag(0)

        # Non-directed photonai_graph.
        bigger = W.T > W
        W = W - W.multiply(bigger) + W.T.multiply(bigger)

        assert W.nnz % 2 == 0
        assert np.abs(W - W.T).mean() < 1e-10
        assert type(W) is scipy.sparse.csr.csr_matrix

        return W


    def get_atlas_coords(self, atlas_name, root_folder):
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
        # use the mean 2d image of all samples for creating the different photonai_graph structures
        X_mean = np.squeeze(np.mean(X, axis=0))

        #get atlas coords
        coords = self.get_atlas_coords(atlas_name=self.atlas_name, root_folder=self.atlas_folder)

        # generate adjacency matrix
        dist, idx = self.distance_scipy_spatial(coords, k=10, metric='euclidean')
        adjacency = self.adjacency(dist, idx).astype(np.float32)

        #turn adjacency into numpy matrix for concatenation
        adjacency = adjacency.toarray()

        X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], -1))
        # X = X[..., None] + adjacency[None, None, :] #use broadcasting to speed up computation
        adjacency = np.repeat(adjacency[np.newaxis, :, :, np.newaxis], X.shape[0], axis=0)
        X = np.concatenate(adjacency, X, axis=3)

        return X





class GraphConstructorThreshold(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    # threshold a matrix to generate the adjacency matrix
    # you can use both a different and the own matrix

    def __init__(self, k_distance=10, threshold=0.1, adjacency_axis=0,
                 concatenation_axis=3,
                 one_hot_nodes=0,
                 return_adjacency_only=0,
                 fisher_transform=0,
                 use_abs=0,
                 zscore=1,
                 use_abs_zscore=0,
                 logs=''):
        self.k_distance = k_distance
        self.threshold = threshold
        self.adjacency_axis = adjacency_axis
        self.concatenation_axis = concatenation_axis
        self.one_hot_nodes = one_hot_nodes
        self.return_adjacency_only = return_adjacency_only
        self.fisher_transform = fisher_transform
        self.use_abs = use_abs
        self.zscore = zscore
        self.use_abs_zscore = use_abs_zscore
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        pass

    def transform(self, X):

        # ensure that the array has the "right" number of dimensions
        if np.ndim(X) == 4:
            Threshold_matrix = X[:, :, :, self.adjacency_axis].copy()
            X_transformed = X.copy()
            if self.fisher_transform == 1:
                Threshold_matrix = individual_fishertransform(Threshold_matrix)
            if self.use_abs == 1:
                Threshold_matrix = np.abs(Threshold_matrix)
            if self.zscore == 1:
                Threshold_matrix = individual_ztransform(Threshold_matrix)
            if self.use_abs_zscore == 1:
                Threshold_matrix = np.abs(Threshold_matrix)
        elif np.ndim(X) == 3:
            Threshold_matrix = X.copy()
            X_transformed = X.copy().reshape(X.shape[0], X.shape[1], X.shape[2], -1)
            if self.fisher_transform == 1:
                Threshold_matrix = individual_fishertransform(Threshold_matrix)
            if self.use_abs == 1:
                Threshold_matrix = np.abs(Threshold_matrix)
            if self.zscore == 1:
                Threshold_matrix = individual_ztransform(Threshold_matrix)
            if self.use_abs_zscore == 1:
                Threshold_matrix = np.abs(Threshold_matrix)

        else:
            raise Exception('encountered unusual dimensions, please check your dimensions')
        #This creates and indvidual adjacency matrix for each person

        Threshold_matrix[Threshold_matrix > self.threshold] = 1
        Threshold_matrix[Threshold_matrix < self.threshold] = 0
        #add extra dimension to make sure that concatenation works later on
        Threshold_matrix = Threshold_matrix.reshape(Threshold_matrix.shape[0], Threshold_matrix.shape[1], Threshold_matrix.shape[2], -1)

        #Add the matrix back again
        if self.one_hot_nodes == 1:
            #construct an identity matrix
            identity_matrix = np.identity((X.shape[1]))
            #expand its dimension for later re-addition
            identity_matrix = np.reshape(identity_matrix, (-1, identity_matrix.shape[0], identity_matrix.shape[1]))
            identity_matrix = np.reshape(identity_matrix, (identity_matrix.shape[0], identity_matrix.shape[1], identity_matrix.shape[2], -1))
            one_hot_node_features = np.repeat(identity_matrix, X.shape[0], 0)
            #concatenate matrices
            X_transformed = np.concatenate((Threshold_matrix, one_hot_node_features), axis=self.concatenation_axis)
        else:
            if self.return_adjacency_only == 0:
                X_transformed = np.concatenate((Threshold_matrix, X_transformed), axis=self.concatenation_axis)
            elif self.return_adjacency_only == 1:
                X_transformed = Threshold_matrix.copy()
            else:
                return ValueError("The argument return_adjacency_only takes only values 0 or 1 no other values. Please check your input values")
            #X_transformed = np.delete(X_transformed, self.adjacency_axis, self.concatenation_axis)


        return X_transformed




class GraphConstructorPercentage(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    # use the mean 2d FA or NOS DTI-Matrix of all samples for thresholding the graphs

    def __init__(self, percentage = 0.8, adjacency_axis = 0,
                 fisher_transform = 0, use_abs = 0, logs=''):
        self.percentage = percentage
        self.adjacency_axis = adjacency_axis
        self.fisher_transform = fisher_transform
        self.use_abs = use_abs
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        pass

    def transform(self, X):

        #generate binary matrix
        BinaryMatrix = np.zeros((1, X.shape[1], X.shape[2], 1))

        for i in range(X.shape[0]):
            #select top percent connections
            # calculate threshold from given percentage cutoff
            if np.ndim(X) == 3:
                lst = X[i, :, :].tolist()
                BinarizedMatrix = X[i, :, :].copy()
                if self.fisher_transform == 1:
                    np.arctanh(BinarizedMatrix)
                if self.use_abs == 1:
                    BinarizedMatrix = np.abs(BinarizedMatrix)
                X_transformed = X.copy()
                X_transformed = X_transformed.reshape((X_transformed.shape[0], X_transformed.shape[1], X_transformed.shape[2], -1))
            elif np.ndim(X) == 4:
                lst = X[i, :, :, self.adjacency_axis].tolist()
                BinarizedMatrix = X[i, :, :, self.adjacency_axis].copy()
                if self.fisher_transform == 0:
                    np.arctanh(BinarizedMatrix)
                if self.use_abs == 1:
                    BinarizedMatrix = np.abs(BinarizedMatrix)
                X_transformed = X.copy()
            else:
                raise ValueError('Input matrix needs to have either 3 or 4 dimensions not more or less.')
            lst = [item for sublist in lst for item in sublist]
            lst.sort()
            #new_lst = lst[int(len(lst) * self.percentage): int(len(lst) * 1)]
            #threshold = new_lst[0]
            threshold = lst[int(len(lst) * self.percentage)]


            #Threshold matrix X to create adjacency matrix
            BinarizedMatrix[BinarizedMatrix > threshold] = 1
            BinarizedMatrix[BinarizedMatrix < threshold] = 0
            BinarizedMatrix = BinarizedMatrix.reshape((-1, BinaryMatrix.shape[0], BinaryMatrix.shape[1]))
            BinarizedMatrix = BinarizedMatrix.reshape((BinaryMatrix.shape[0], BinaryMatrix.shape[1], BinaryMatrix.shape[2], -1))

            #concatenate matrix back
            BinaryMatrix = np.concatenate((BinaryMatrix, BinarizedMatrix), axis = 3)

        #drop first matrix as it is empty
        BinaryMatrix = np.delete(BinaryMatrix, 0, 3)
        BinaryMatrix = np.swapaxes(BinaryMatrix, 3, 0)
        X_transformed = np.concatenate((BinaryMatrix, X_transformed), axis = 3)

        return X_transformed




#uses random walks to generate the connectivity matrix for photonai_graph structures
class GraphConstructorRandomWalks(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, k_distance=10, number_of_walks=10, walk_length=10, window_size=5,
                 no_edge_weight = 1,
                 transform_style = "mean", adjacency_axis=0, logs=''):
        self.k_distance = k_distance
        self.number_of_walks = number_of_walks
        self.walk_length = walk_length
        self.window_size = window_size
        self.transform_style = transform_style
        self.adjacency_axis = adjacency_axis
        self.no_edge_weight = no_edge_weight
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        pass

    def distance_sklearn_metrics(self, z, k, metric='euclidean'):
        """Compute exact pairwise distances."""
        d = sklearn.metrics.pairwise.pairwise_distances(
            z, metric=metric, n_jobs=-2)
        # k-NN photonai_graph.
        idx = np.argsort(d)[:, 1:k + 1]
        d.sort()
        d = d[:, 1:k + 1]
        return d, idx

    def adjacency(self, dist, idx):
        """Return the adjacency matrix of a kNN photonai_graph."""
        M, k = dist.shape
        assert M, k == idx.shape
        assert dist.min() >= 0

        # Weights.
        sigma2 = np.mean(dist[:, -1]) ** 2
        dist = np.exp(- dist ** 2 / sigma2)

        # Weight matrix.
        I = np.arange(0, M).repeat(k)
        J = idx.reshape(M * k)
        V = dist.reshape(M * k)
        W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))

        # No self-connections.
        W.setdiag(0)

        # Non-directed photonai_graph.
        bigger = W.T > W
        W = W - W.multiply(bigger) + W.T.multiply(bigger)

        assert W.nnz % 2 == 0
        assert np.abs(W - W.T).mean() < 1e-10
        assert type(W) is scipy.sparse.csr.csr_matrix

        return W

    def random_walk(self, adjacency, walk_length, num_walks):
        """Performs a random walk on a given photonai_graph"""
        # a -> adj
        # i -> starting row
        walks = []  # holds transitions
        elements = np.arange(adjacency.shape[0])  # for our photonai_graph [0,1,2,3]
        for k in range(num_walks):
            node_walks = []
            for i in range(elements.shape[0]):
                index = np.random.choice(elements, replace=False)  # current index for this iteration
                count = 0 # count of transitions
                walk = []
                while count < walk_length:
                    count += 1
                    walk.append(index)
                    probs = adjacency[index]  # probability of transitions
                    # sample from probs
                    sample = np.random.choice(elements, p=probs)  # sample a target using probs
                    index = sample # go to target
                node_walks.append(walk)
            walks.append(node_walks)

        return walks

        '''

        dims = 2
        step_n = walk_length
        step_set = [-1, 0, 1]
        #start on a given random vertex
        origin = vertices[np.random.randint(low=0, high=vertices.shape[0]), np.random.randint(low=0, high=vertices.shape[1])]

        #step onto other random vertex
        step_shape = (step_n, dims)
        steps = np.random.choice(a=step_set, size=step_shape)
        #append that vertex value to the series
        path = np.concatenate([origin, steps])
        #slide window and check if they co-occur
        start = path[:1]
        stop = path[-1:]


        return #list of lists with random walks
        
        frequency = np.zeros((adjacency.shape))
        for i in range(0, self.number_of_walks):
            shuffled_vertices = random.shuffle(d) #do stuff
            for i in range(len(shuffled_vertices)):
                for j in range(len(shuffled_vertices[i])):
                    vertex = d[i, j]
                    self.random_walk(vertices = shuffled_vertices, walk_length=self.walk_length, startpoint=vertex)
        
        '''

    def sliding_window(self, seq, n):
        "Returns a sliding window (of width n) over data from the iterable"
        "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result




        return #sliding window co-ocurrence

    def sliding_window_frequency(self, X_mean, walk_list):

        coocurrence = np.zeros((X_mean.shape[0], X_mean.shape[1]))

        for i in walk_list:
            node_walks = i
            for x in node_walks:
                for w in self.sliding_window(x, self.window_size):
                    for subset in combinations(w, 2):
                        coord1 = subset[0]
                        coord2 = subset[1]
                        coocurrence[coord1, coord2] += 1
                    # if they occur at the same time add + 1 to the frequency table

        return coocurrence

    def transform(self, X):
        # transform each individual or make a mean matrix
        if self.transform_style == "mean":
            # use the mean 2d image of all samples for creating the different photonai_graph structures
            X_mean = np.squeeze(np.mean(X, axis=0))

            # select the proper matrix in case you have multiple
            if np.ndim(X_mean) == 3:
                X_mean = X_mean[:, :, self.adjacency_axis]
            elif np.ndim(X_mean) == 2:
                X_mean = X_mean
            else:
                raise ValueError('The input matrices need to have 3 or 4 dimensions. Please check your input matrix.')

            d, idx = self.distance_sklearn_metrics(X_mean, k=self.k_distance, metric='euclidean')
            adjacency = self.adjacency(d, idx).astype(np.float32)

            adjacency = adjacency.toarray()
            if self.no_edge_weight == 1:
                adjacency[adjacency > 0] = 1

            adjacency_rowsum = np.sum(adjacency, axis=1)
            adjacency_norm = adjacency/adjacency_rowsum[:, np.newaxis]

            walks =self.random_walk(adjacency=adjacency_norm, walk_length=self.walk_length, num_walks=self.number_of_walks)

            higherorder_adjacency = self.sliding_window_frequency(X_mean=adjacency, walk_list=walks)

            # obtain the kNN photonai_graph from the new adjacency matrix
            d, idx = self.distance_sklearn_metrics(higherorder_adjacency, k=10, metric='euclidean')
            higherorder_adjacency = self.adjacency(d, idx).astype(np.float32)

            # convert this adjacency matrix to dense format
            higherorder_adjacency = higherorder_adjacency.toarray()

            # reshape X to add the new adjacency
            X_transformed = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], -1))
            # X = X[..., None] + adjacency[None, None, :] #use broadcasting to speed up computation
            adjacency = np.repeat(higherorder_adjacency[np.newaxis, :, :, np.newaxis], X.shape[0], axis=0)
            X_transformed = np.concatenate((adjacency, X_transformed), axis=3)

        elif self.transform_style == "individual":
            X_rw = X.copy()
            # select the proper matrix in case you have multiple
            if np.ndim(X_rw) == 4:
                X_rw = X_rw[:, :, :, self.adjacency_axis]
                X_features = X_rw.copy()
            elif np.ndim(X_rw) == 3:
                X_rw = X_rw
                X_features = X_rw.copy()
                X_features = X_features.reshape((X_features.shape[0], X_features.shape[1], X_features.shape[2], -1))
            else:
                raise ValueError('The input matrices need to have 3 or 4 dimensions. Please check your input matrix.')

            adjacency_list = []
            for i in range(X.shape[0]):
                d, idx = self.distance_sklearn_metrics(X_rw[i, :, :], k=self.k_distance, metric='euclidean')
                adjacency = self.adjacency(d, idx).astype(np.float32)

                # normalize adjacency
                if self.no_edge_weight == 1:
                    adjacency[adjacency > 0] = 1

                adjacency_rowsum = np.sum(adjacency, axis=1)
                adjacency_norm = adjacency / adjacency_rowsum[:, np.newaxis]

                walks = self.random_walk(adjacency=adjacency_norm, walk_length=self.walk_length,
                                         num_walks=self.number_of_walks)

                higherorder_adjacency = self.sliding_window_frequency(X_mean=adjacency, walk_list=walks)

                # obtain the kNN photonai_graph from the new adjacency matrix
                d, idx = self.distance_sklearn_metrics(higherorder_adjacency, k=10, metric='euclidean')
                higherorder_adjacency = self.adjacency(d, idx).astype(np.float32)

                # convert this adjacency matrix to dense format
                higherorder_adjacency = higherorder_adjacency.toarray()
                # turn adjacency into numpy matrix for concatenation
                adjacency = adjacency.toarray()
                adjacency_list.append(adjacency)

            adjacency_list = np.asarray(adjacency_list)
            adjacency_list = adjacency_list.reshape((adjacency_list.shape[0], adjacency_list.shape[1],
                                                    adjacency_list.shape[2], -1))
            X_transformed = np.concatenate((adjacency_list, X_features), axis=3)


        else:
            raise KeyError('Only mean and individual transform are supported. '
                           'Please check your spelling for the parameter transform_style.')

        return X_transformed
