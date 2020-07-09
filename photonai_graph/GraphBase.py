import os
from photonai_graph.GraphUtilities import RegisterGraph_force


class GraphBase():
    _estimator_type = "transformer"
    # mother class for Graph modules
    def __init__(self, logs=''):
        self.registered = RegisterGraph_force()
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()