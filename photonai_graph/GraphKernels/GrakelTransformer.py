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

    def __init__(self, *args, transformation: str = None, **kwargs):
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
        transformation_class = self.transformations[transformation]
        self.transformation = transformation_class(*args, **kwargs)

    def fit(self, X, y=None):
        self.transformation.fit(X, y)

    def transform(self, X):
        self.transformation.transform(X)
