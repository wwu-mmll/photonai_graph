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

    def __new__(cls, transformation: str = None, **kwargs):
        if transformation not in cls.transformations:
            raise ValueError(f"The requested transformation {transformation} was not found.")
        return cls.transformations[transformation](**kwargs)
