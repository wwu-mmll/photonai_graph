import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from photonai_graph.Controllability.controllability_functions import modal_control, \
    average_control


class ControllabilityMeasureTransform(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self,
                 mod_control: int = 1,
                 ave_control: int = 1,
                 adjacency_axis: int = 0,
                 logs: str = None):
        """
            Class for extraction of controllability measures. Allows
            for extraction of controllability measures.

            Parameters
            ----------
            mod_control: int,default = 1
                Whether to calculate nodewise modal controllability (1) or not (0).
            ave_control: int,default=1
                Whether to calculate nodewise average controllability (1) or not (0).
            adjacency_axis: int,default=0
                position of the adjacency matrix, default being zero
            logs: str,default=None
            """
        self.mod_control = mod_control
        self.ave_control = ave_control
        self.adjacency_axis = adjacency_axis
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()
        if not self.mod_control and not self.ave_control:
            raise ValueError("You need to select either average, modal or both controllabilities."
                             "Please check your hyperparameter choices.")

    def fit(self, X, y):
        return self

    def transform(self, X, as_array: bool = True):

        controllabilities = []
        for subj in range(X.shape[0]):
            vec = None
            if self.mod_control:
                vec = modal_control(X[subj, :, :, self.adjacency_axis])
            if self.ave_control:
                ac = average_control(X[subj, :, :, self.adjacency_axis])
                if vec is None:
                    vec = ac
                else:
                    vec = np.concatenate((vec, ac))

            controllabilities.append(vec)
        if not as_array:
            return controllabilities

        return np.asarray(controllabilities)

    def extract_measures(self,
                         X: np.ndarray = None,
                         path: str = None,
                         ids: List[int] = None,
                         node_list: List[str] = None):
        """Extract controllability measures and write them to a csv file

        Parameters
        ----------
        X: np.ndarray,default=None
            Input numpy array to transform
        path: str,default=None
            Output path for generated CSV
        ids: List[int],default=None
            List of ids for the graphs. If None the graphs are enumerated
        node_list: List[str],default=None
            List of names for the nodes. If None the nodes of the graphs are enumerated and entitled by the calculated
            controllability measure

        """

        # check if id list is correct length
        if ids is not None:
            if not len(ids) == len(list(range(X.shape[0]))):
                raise ValueError("ID list does not match number of samples.")
        else:
            ids = list(range(X.shape[0]))

        # check if node list is correct length
        if node_list is not None:
            if not len(node_list) == len(list(range(X.shape[1]))):
                raise ValueError("Node list does not match number of nodes.")
        else:
            node_list = list(range(X.shape[1]))
            for i in node_list:
                node_list[i] = "node_" + str(node_list[i])

        # base columns for every csv
        cols = ["ID"]

        controllabilities = self.transform(X, as_array=False)

        df = pd.DataFrame(controllabilities)
        df.insert(0, "ID", ids)

        if self.mod_control:
            cols += self.__add_prefix(node_list, "modal control ")
        if self.ave_control:
            cols += self.__add_prefix(node_list, "average control ")
        df.to_csv(path_or_buf=path, header=cols, index=False)

    @staticmethod
    def __add_prefix(x_list, prefix):
        """Adds a prefix to a list of strings"""
        y_list = x_list.copy()
        for i in range(len(y_list)):
            y_list[i] = prefix + y_list[i]

        return y_list
