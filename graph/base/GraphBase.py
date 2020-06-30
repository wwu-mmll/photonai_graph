import os
from photonai.graph.base.GraphUtilities import RegisterGraph_force

class GraphBase():
    _estimator_type = "transformer"
    # mother calss for Graph modules
    def __init__(self, logs=''):
        self.registered = RegisterGraph_force()
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()