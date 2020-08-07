"""
ONLY HERE FOR LEGACY REASONS
===========================================================
Project: PHOTON Graph
===========================================================
Description
-----------
A wrapper containing functions for turning connectivity matrices into photonai_graph structures

Version
-------
Created:        12-08-2019
Last updated:   03-08-2020


Author
------
Vincent Holstein
Translationale Psychiatrie
Universitaetsklinikum Muenster
"""

# TODO: "deposit" atlas coordinate files ?
# TODO: add advanced documentation for every method
# TODO: add a fisher transform for the connectivity matrix values
# TODO: make mean matrix an attribute
# TODO: make a constructor mother class
# TODO: add weighted method for different transformers

from photonai_graph.GraphUtilities import individual_ztransform, individual_fishertransform
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import sklearn
import scipy
import os
from itertools import islice, combinations














# A method to construct a graph based on the percentage of strongest connections



# a method to construct graphs based on a threshold window



# a method to construct graphs based on a percentage window



# uses random walks to generate the connectivity matrix for photonai_graph structures

