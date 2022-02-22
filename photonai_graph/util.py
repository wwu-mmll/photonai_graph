from photonai.base import PhotonRegistry
import os
try:
    import dgl
except ImportError:
    pass
try:
    import grakel
except ImportError:
    pass
try:
    import torch
except ImportError:
    pass
try:
    from gem.embedding.hope import HOPE
except ImportError:
    pass
try:
    import euler
except ImportError:
    pass


def delete_photonai_graph():
    reg = PhotonRegistry()
    reg.delete_module('photonai_graph')


def register_photonai_graph():
    current_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photonai_graph/photonai_graph.json")
    reg = PhotonRegistry()
    reg.add_module(current_path)

    
packages = ["dgl", "grakel", "pytorch", "gem"]


def assert_imported(package: list = None):
    if "dgl" in package:
        try:
            dgl.graph()
        except NameError as e:
            print("dgl has to be installed. Please install running version with pip install.")
            raise e
        except Exception:
            pass
    if "grakel" in package:
        try:
            grakel.Graph()
        except NameError as e:
            print("Grakel has to be installed. Please install running version with pip install.")
            raise e
        except Exception:
            pass
    if "pytorch" in package:
        try:
            torch.tensor()
        except NameError as e:
            print("pytorch has to be installed. Please install running version with pip install.")
            raise e
        except Exception:
            pass
    if "gem" in package:
        try:
            HOPE()
        except NameError as e:
            print("gem has to be installed. Please install running version with pip install.")
            raise e
        except Exception:
            pass
    if "euler" in package:
        try:
            euler()
        except NameError as e:
            print("pytorch has to be installed. Please install running version with pip install.")
            raise e
        except Exception:
            pass
    
