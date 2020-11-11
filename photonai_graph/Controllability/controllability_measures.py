from photonai_graph.Controllability.controllability_functions import modal_control, \
    average_control
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import os


class ControllabilityMeasureTransform(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    """
    Class for extraction of controllability measures. Allows
    for extraction of controllability measures.

    Parameters
    ----------
    * 'mod_control' [int, default = 1]:
        Whether to calculate nodewise modal controllability (1) or not (0).
    * 'ave_control' [int, default=1]:
        Whether to calculate nodewise average controllability (1) or not (0).
    * `adjacency_axis` [int, default=0]:
        position of the adjacency matrix, default being zero
    """

    def __init__(self,
                 mod_control: int = 1,
                 ave_control: int = 1,
                 adjacency_axis: int = 0,
                 logs = ""):
        self.mod_control = mod_control
        self.ave_control = ave_control
        self.adjacency_axis = adjacency_axis
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        return self

    def transform(self, X):

        controllabilities = []

        for subj in range(X.shape[0]):

            if self.mod_control == 1 and self.ave_control == 0:
                vec = modal_control(X[subj, :, :, self.adjacency_axis])
            elif self.ave_control == 1 and self.mod_control == 0:
                vec = average_control(X[subj, :, :, self.adjacency_axis])
            elif self.mod_control == 1 and self.ave_control == 1:
                modcontr = modal_control(X[subj, :, :, self.adjacency_axis])
                avecontr = average_control(X[subj, :, :, self.adjacency_axis])
                vec = np.concatenate((modcontr, avecontr))
            else:
                raise Exception("You need to select either average, modal or both controllabilities."
                                "Please check your hyperparameter choices.")

            controllabilities.append(vec)

        return np.asarray(controllabilities)

    def extract_measures(self, X, path, ids=None, node_list=None):
        """Extract controllability measures and write them to a csv file"""

        # check if id list is correct length
        if ids is not None:
            if not len(ids) == len(list(range(X.shape[0]))):
                raise Exception("ID list does not match number of samples.")
        else:
            ids = self.get_id_list(X)

        # check if node list is correct length
        if node_list is not None:
            if not len(node_list) == len(list(range(X.shape[1]))):
                raise Exception("Node list does not match number of nodes.")
        else:
            node_list = self.get_node_list(X)

        # base columns for every csv
        base_list = ["ID"]

        controllabilities = []

        for subj in range(X.shape[0]):

            if self.mod_control == 1 and self.ave_control == 0:
                vec = modal_control(X[subj, :, :, self.adjacency_axis])
            elif self.ave_control == 1 and self.mod_control == 0:
                vec = average_control(X[subj, :, :, self.adjacency_axis])
            elif self.mod_control == 1 and self.ave_control == 1:
                modcontr = modal_control(X[subj, :, :, self.adjacency_axis])
                avecontr = average_control(X[subj, :, :, self.adjacency_axis])
                vec = np.concatenate((modcontr, avecontr))
            else:
                raise Exception("You need to select either average, modal or both controllabilities."
                                "Please check your hyperparameter choices.")

            controllabilities.append(vec)

        df = pd.DataFrame(controllabilities)
        df["ID"] = ids

        if self.mod_control == 1 and self.ave_control == 0:
            mod_node_list = self.add_prefix(node_list, "modal control ")
            cols = base_list + mod_node_list
            df.to_csv(path_or_buf=path, header=cols, index=False)
        elif self.mod_control == 0 and self.ave_control == 1:
            ave_node_list = self.add_prefix(node_list, "average control ")
            cols = base_list + ave_node_list
            df.to_csv(path_or_buf=path, header=cols, index=False)
        elif self. mod_control == 1 and self.ave_control == 1:
            mod_node_list = self.add_prefix(node_list, "modal control ")
            ave_node_list = self.add_prefix(node_list, "average control ")
            cols = base_list + mod_node_list + ave_node_list
            df.to_csv(path_or_buf=path, header=cols, index=False)
        else:
            raise Exception("You need to select either average, modal or both controllabilities."
                            "Please check your hyperparameter choices.")

    @staticmethod
    def get_node_list(X):
        """Returns a list with node names labeled by strings"""
        lst = list(range(X.shape[1]))
        for i in lst:
            lst[i] = "node_" + str(lst[i])

        return lst

    @staticmethod
    def get_id_list(X):
        """Returns a list with ids as a continous series of integers"""
        lst = list(range(X.shape[0]))

        return lst

    @staticmethod
    def add_prefix(x_list, prefix):
        """Adds a prefix to a list of strings"""
        for i in x_list:
            i = prefix + i

        return x_list
