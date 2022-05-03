import sys

from photonai.base import PhotonRegistry
import os
try:
    import dgl
except ImportError:  # pragma: no cover
    pass
try:
    import grakel
except ImportError:  # pragma: no cover
    pass
try:
    import torch
except ImportError:  # pragma: no cover
    pass
try:
    from gem.embedding.hope import HOPE
except ImportError:  # pragma: no cover
    pass
try:
    import euler
except ImportError:  # pragma: no cover
    pass

"""This file is by intention excluded from coverage calculation"""


def delete_photonai_graph():  # pragma: no cover
    reg = PhotonRegistry()
    reg.delete_module('photonai_graph')


def register_photonai_graph():  # pragma: no cover
    current_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photonai_graph/photonai_graph.json")
    reg = PhotonRegistry()
    reg.add_module(current_path)


def assert_imported(packages: list = None):  # pragma: no cover
    for package in packages:
        if package in sys.modules:
            return True
        else:
            raise ImportError(f"Could not load {package}. Please install it according to the documentation")
