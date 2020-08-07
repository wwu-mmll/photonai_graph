from photonai_graph.GraphConstruction import GraphConstructorThreshold
import pickle
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

with open("/spm-data/Scratch/spielwiese_janernsting/Reinforcement Learning Graph Analyse/input/FOR2107_DF1-3_030320_rs_connectivity_matrices_QAed.p", "rb") as file:
    X = pickle.load(file)

graph_constructor = GraphConstructorThreshold(threshold=0.5, use_abs=False, fisher_transform=True)
X = graph_constructor.transform(X)