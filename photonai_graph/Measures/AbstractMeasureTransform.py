from abc import ABC

import igraph
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.base import BaseEstimator, TransformerMixin


class AbstractMeasureTransform(ABC, BaseEstimator, TransformerMixin):
    """
    Base class for measurement transforms
    """

    def __init__(self, graph_functions: dict = None):
        self.graph_functions = graph_functions

    def _shared_inner_transform(self, x_transformed):
        for c_measure in range(len(self.graph_functions)):
            expected_values = max([len(graph[c_measure]) for graph in x_transformed])
            for graph in x_transformed:
                if len(graph[c_measure]) < expected_values:
                    graph[c_measure] = [np.NAN] * expected_values

        return x_transformed

    @staticmethod
    def _shared_transform(x_transformed):
        for graph_idx in range(len(x_transformed)):
            g_m = list()
            for measure in x_transformed[graph_idx]:
                g_m.extend(measure)
            x_transformed[graph_idx] = g_m

        x_transformed = np.asarray(x_transformed)

        return x_transformed

    def handle_outputs(self, results, measure_list):
        # handle results
        if isinstance(results, dict):
            for rskey, rsval in results.items():
                self.handle_outputs(rsval, measure_list)
            return measure_list

        if isinstance(results, list):
            measure_list.extend(results)
            return measure_list

        # currently only networkx functions return tuples
        # The second return value can be discarded in these functions
        if isinstance(results, tuple):
            for result in results:
                self.handle_outputs(result, measure_list)
            return measure_list

        if isinstance(results, igraph.Graph):
            return measure_list

        if isinstance(results, nx.Graph):
            return measure_list

        measure_list.append(results)
        return measure_list

    def _shared_extraction(self, x_graphs, ids, path):
        x_transformed = self._inner_transform(x_graphs)

        measurements = []
        for graph, gid in zip(x_transformed, ids):
            for measurement_id, result in enumerate(graph):
                for res in result:
                    current_measurement = [gid, list(self.graph_functions.keys())[measurement_id], res]
                    measurements.append(current_measurement)

        df = pd.DataFrame(measurements)

        col_names = ["graph_id", "measure", "value"]

        df.to_csv(path_or_buf=path, header=col_names, index=None)

    def _inner_transform(self, X):
        """
        Inner transformation

        Parameters
        ----------
        X

        Returns
        -------

        """