from sklearn.base import BaseEstimator, TransformerMixin

from grakel import VertexHistogram, EdgeHistogram, ShortestPath, GraphletSampling, RandomWalk, NeighborhoodHash,\
    WeisfeilerLehman, NeighborhoodSubgraphPairwiseDistance, LovaszTheta, SvmTheta, OddSth, GraphHopper, Propagation,\
    PyramidMatch, SubgraphMatching, MultiscaleLaplacian, CoreFramework


class GrakelTransformer(BaseEstimator, TransformerMixin):

    transformations = {
        "VertexHistogram": VertexHistogram,
        "EdgeHistogram": EdgeHistogram,
        "ShortestPath": ShortestPath,
        "GraphletSampling": GraphletSampling,
        "RandomWalk": RandomWalk,
        "NeighborhoodHash": NeighborhoodHash,
        "WeisfeilerLehman": WeisfeilerLehman,
        "NeighborhoodSubgraphPairwiseDistance": NeighborhoodSubgraphPairwiseDistance,
        "LovaszTheta": LovaszTheta,
        "SvmTheta": SvmTheta,
        "OddSth": OddSth,
        "GraphHopper": GraphHopper,
        "Propagation": Propagation,
        "PyramidMatch": PyramidMatch,
        "SubgraphMatching": SubgraphMatching,
        "MultiscaleLaplacian": MultiscaleLaplacian,
        "CoreFramework": CoreFramework
    }

    def __init__(self, transformation: str = None, **kwargs):
        """A transformer class for transforming graphs in grakel format.

        Parameters
        ----------
        args
            Transformer arguments
        transformation
            The transformation requested by the user
        kwargs
            Transformer arguments
        """
        if transformation not in self.transformations.keys():
            raise ValueError(f"The requested transformation {transformation} was not found.")
        self.transformation_class = self.transformations[transformation]
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        self.kwargs_keys = kwargs.keys()
        self.transformation = None

    def fit(self, X, y=None):
        kwargs = {k: getattr(self, k) for k in self.kwargs_keys}
        self.transformation = self.transformation_class(**kwargs)
        self.transformation.fit(X, y)
        return self

    def transform(self, X):
        if self.transformation is None:
            raise ValueError("Transformation was not fit yet.")
        return self.transformation.transform(X)
