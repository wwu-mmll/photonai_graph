import networkx as nx
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt


def get_random_graphs(num_graphs=500, num_nodes=20, edge_prob=0.3):
    # returns a list of random networkx graphs
    graph_list = []
    for i in range(num_graphs):
        graph = nx.fast_gnp_random_graph(num_nodes, edge_prob)
        graph_list.append(graph)

    return graph_list


def get_sw_graph(num_graphs=500, num_nodes=20, knn=5, edge_rew=0.2):
    # returns a list of small world networkx graphs
    graph_list = []
    for i in range(num_graphs):
        graph = nx.watts_strogatz_graph(num_nodes, knn, edge_rew)
        graph_list.append(graph)

    return graph_list


def plot_nx_edge_count(graphs1, graphs2, label1="graphs_1", label2="graphs_2"):
    # plot the distribution of edges for 2 sets of graphs
    edges1 = []
    edges2 = []

    for graph in graphs1:
        edge_count = graph.number_of_edges()
        edges1.append(edge_count)

    for graph in graphs2:
        edge_count = graph.number_of_edges()
        edges2.append(edge_count)

    labels = ([label1] * len(edges1)) + ([label2] * len(edges2))
    edges = edges1 + edges2

    df = pd.DataFrame(list(zip(edges, labels)), columns=["edge_count", "Group"])

    sns.displot(df, x="edge_count", hue="Group", kind="kde", fill=True)
    plt.show()
