from photonai.base import PhotonRegistry
import os


def delete_photonai_graph():
    reg = PhotonRegistry()
    reg.delete_module('photonai_graph')


def register_photonai_graph():
    current_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photonai_graph/photonai_graph.json")
    reg = PhotonRegistry()
    reg.add_module(current_path)
