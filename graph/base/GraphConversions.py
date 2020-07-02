from networkx.drawing.nx_agraph import write_dot, read_dot
import networkx as nx
import os


def save_networkx_to_file(Graphs, path, output_format="dot", IDs=None):
    # Case 1: a list of networkx graphs as input
    if isinstance(Graphs, list):
        # Case 1.1: an ID is specified
        if isinstance(IDs, list):
            # check they have equal length
            if len(Graphs) == len(IDs):
                # run graph_writing with IDs
                if output_format == "dot":
                    for graph, i in zip(Graphs, IDs):
                        graph_filename = "graph_" + str(i)
                        graph_path = os.path.join(path, graph_filename)
                        write_dot(graph, graph_path)
                elif output_format == "AdjacencyList":
                    for graph, i in zip(Graphs, IDs):
                        graph_filename = "graph_" + str(i)
                        graph_path = os.path.join(path, graph_filename)
                        nx.write_adjlist(graph, graph_path)
                elif output_format == "MultilineAdjacencyList":
                    for graph, i in zip(Graphs, IDs):
                        graph_filename = "graph_" + str(i)
                        graph_path = os.path.join(path, graph_filename)
                        nx.write_multiline_adjlist(graph, graph_path)
                elif output_format == "EdgeList":
                    for graph, i in zip(Graphs, IDs):
                        graph_filename = "graph_" + str(i)
                        graph_path = os.path.join(path, graph_filename)
                        nx.write_edgelist(graph, graph_path)
                elif output_format == "WeightedEdgeList":
                    for graph, i in zip(Graphs, IDs):
                        graph_filename = "graph_" + str(i)
                        graph_path = os.path.join(path, graph_filename)
                        nx.write_edgelist(graph, graph_path)
                elif output_format == "GEXF":
                    for graph, i in zip(Graphs, IDs):
                        graph_filename = "graph_" + str(i)
                        graph_path = os.path.join(path, graph_filename)
                        nx.write_gexf(graph, graph_path)
                elif output_format == "pickle":
                    for graph, i in zip(Graphs, IDs):
                        graph_filename = "graph_" + str(i)
                        graph_path = os.path.join(path, graph_filename)
                        nx.write_gpickle(graph, graph_path)
                elif output_format == "GML":
                    for graph, i in zip(Graphs, IDs):
                        graph_filename = "graph_" + str(i)
                        graph_path = os.path.join(path, graph_filename)
                        nx.write_gml(graph, graph_path)
                elif output_format == "GraphML":
                    for graph, i in zip(Graphs, IDs):
                        graph_filename = "graph_" + str(i)
                        graph_path = os.path.join(path, graph_filename)
                        nx.write_graphml(graph, graph_path)
                elif output_format == "GraphML-XML":
                    for graph, i in zip(Graphs, IDs):
                        graph_filename = "graph_" + str(i)
                        graph_path = os.path.join(path, graph_filename)
                        nx.write_graphml_xml(graph, graph_path)
                elif output_format == "GraphML-LXML":
                    for graph, i in zip(Graphs, IDs):
                        graph_filename = "graph_" + str(i)
                        graph_path = os.path.join(path, graph_filename)
                        nx.write_graphml_lxml(graph, graph_path)
                elif output_format == "YAML":
                    for graph, i in zip(Graphs, IDs):
                        graph_filename = "graph_" + str(i)
                        graph_path = os.path.join(path, graph_filename)
                        nx.write_yaml(graph, graph_path)
                elif output_format == "graph6":
                    for graph, i in zip(Graphs, IDs):
                        graph_filename = "graph_" + str(i)
                        graph_path = os.path.join(path, graph_filename)
                        nx.write_graph6(graph, graph_path)
                elif output_format == "PAJEK":
                    for graph, i in zip(Graphs, IDs):
                        graph_filename = "graph_" + str(i)
                        graph_path = os.path.join(path, graph_filename)
                        nx.write_pajek(graph, graph_path)
                else:
                    raise Exception("The output format is not supported. Please check your output format.")
            else:
                # give back error cause unequal length
                raise Exception(
                    'The graph ID list and the list of Graphs are not of equal length. Please ensure that they have equal length.')
        # Case 1.2: no ID is specified
        else:
            # run the counter based method if no ID list is specified
            counter = 1
            if output_format == "dot":
                for graph in Graphs:
                    graph_filename = "graph_" + str(counter)
                    graph_path = os.path.join(path, graph_filename)
                    write_dot(graph, graph_path)
                    counter += 1
            elif output_format == "AdjacencyList":
                for graph in Graphs:
                    graph_filename = "graph_" + str(counter)
                    graph_path = os.path.join(path, graph_filename)
                    nx.write_adjlist(graph, graph_path)
                    counter += 1
            elif output_format == "MultilineAdjacencyList":
                for graph in Graphs:
                    graph_filename = "graph_" + str(counter)
                    graph_path = os.path.join(path, graph_filename)
                    nx.write_multiline_adjlist(graph, graph_path)
                    counter += 1
            elif output_format == "EdgeList":
                for graph in Graphs:
                    graph_filename = "graph_" + str(counter)
                    graph_path = os.path.join(path, graph_filename)
                    nx.write_edgelist(graph, graph_path)
                    counter += 1
            elif output_format == "WeightedEdgeList":
                for graph in Graphs:
                    graph_filename = "graph_" + str(counter)
                    graph_path = os.path.join(path, graph_filename)
                    nx.write_edgelist(graph, graph_path)
                    counter += 1
            elif output_format == "GEXF":
                for graph in Graphs:
                    graph_filename = "graph_" + str(counter)
                    graph_path = os.path.join(path, graph_filename)
                    nx.write_gexf(graph, graph_path)
                    counter += 1
            elif output_format == "pickle":
                for graph, i in zip(Graphs, IDs):
                    graph_filename = "graph_" + str(counter)
                    graph_path = os.path.join(path, graph_filename)
                    nx.write_gpickle(graph, graph_path)
                    counter += 1
            elif output_format == "GML":
                for graph in Graphs:
                    graph_filename = "graph_" + str(counter)
                    graph_path = os.path.join(path, graph_filename)
                    nx.write_gml(graph, graph_path)
                    counter += 1
            elif output_format == "GraphML":
                for graph in Graphs:
                    graph_filename = "graph_" + str(counter)
                    graph_path = os.path.join(path, graph_filename)
                    nx.write_graphml(graph, graph_path)
                    counter += 1
            elif output_format == "GraphML-XML":
                for graph in Graphs:
                    graph_filename = "graph_" + str(counter)
                    graph_path = os.path.join(path, graph_filename)
                    nx.write_graphml_xml(graph, graph_path)
                    counter += 1
            elif output_format == "GraphML-LXML":
                for graph in Graphs:
                    graph_filename = "graph_" + str(counter)
                    graph_path = os.path.join(path, graph_filename)
                    nx.write_graphml_lxml(graph, graph_path)
                    counter += 1
            elif output_format == "YAML":
                for graph in Graphs:
                    graph_filename = "graph_" + str(counter)
                    graph_path = os.path.join(path, graph_filename)
                    nx.write_yaml(graph, graph_path)
                    counter += 1
            elif output_format == "graph6":
                for graph in Graphs:
                    graph_filename = "graph_" + str(counter)
                    graph_path = os.path.join(path, graph_filename)
                    nx.write_graph6(graph, graph_path)
                    counter += 1
            elif output_format == "PAJEK":
                for graph in Graphs:
                    graph_filename = "graph_" + str(counter)
                    graph_path = os.path.join(path, graph_filename)
                    nx.write_pajek(graph, graph_path)
                    counter += 1
    # Case 2: the input is just a single graph
    if isinstance(Graphs, nx.classes.graph.Graph):
        if output_format == "dot":
            write_dot(Graphs, path)
        elif output_format == "AdjacencyList":
            nx.write_adjlist(Graphs, path)
        elif output_format == "MultilineAdjacencyList":
            nx.write_multiline_adjlist(Graphs, path)
        elif output_format == "EdgeList":
            nx.write_edgelist(Graphs, path)
        elif output_format == "WeightedEdgeList":
            nx.write_edgelist(Graphs, path)
        elif output_format == "GEXF":
            nx.write_gexf(Graphs, path)
        elif output_format == "pickle":
            nx.write_gpickle(Graphs, path)
        elif output_format == "GML":
            nx.write_gml(Graphs, path)
        elif output_format == "GraphML":
            nx.write_graphml(Graphs, path)
        elif output_format == "GraphML-XML":
            nx.write_graphml_xml(Graphs, path)
        elif output_format == "GraphML-LXML":
            nx.write_graphml_lxml(Graphs, path)
        elif output_format == "YAML":
            nx.write_yaml(Graphs, path)
        elif output_format == "graph6":
            nx.write_graph6(Graphs, path)
        elif output_format == "PAJEK":
            nx.write_pajek(Graphs, path)
        else:
            raise Exception("Output format not implemented or recognized. Please check your desired output format.")


def load_file_to_networkx(path, input_format="dot"):
    if isinstance(path, list):
        graph_list = []

        if input_format == "dot":
            for graph in path:
                G = read_dot(graph)
                graph_list.append(G)
        elif input_format == "AdjacencyList":
            for graph in path:
                G = nx.read_adjlist(graph)
                graph_list.append(G)
        elif input_format == "MultilineAdjacencyList":
            for graph in path:
                G = nx.read_multiline_adjlist(graph)
        elif input_format == "EdgeList":
            for graph in path:
                G = nx.read_edgelist(graph)
                graph_list.append(G)
        elif input_format == "WeightedEdgeList":
            for graph in path:
                G = nx.read_edgelist(graph)
                graph_list.append(G)
        elif input_format == "GEXF":
            for graph in path:
                G = nx.read_gexf(graph)
                graph_list.append(G)
        elif input_format == "pickle":
            for graph in path:
                G = nx.read_gpickle(graph)
                graph_list.append(G)
        elif input_format == "GML":
            for graph in path:
                G = nx.read_gml(graph)
                graph_list.append(G)
        elif input_format == "GraphML":
            for graph in path:
                G = nx.read_graphml(graph)
                graph_list.append(G)
        elif input_format == "YAML":
            for graph in path:
                G = nx.read_yaml(graph)
                graph_list.append(G)
        elif input_format == "graph6":
            for graph in path:
                G = nx.read_graph6(graph)
                graph_list.append(G)
        elif input_format == "PAJEK":
            for graph in path:
                G = nx.read_pajek(graph)
                graph_list.append(G)

    if isinstance(path, str):
        if input_format == "dot":
            graph_list = read_dot(path)
        elif input_format == "AdjacencyList":
            graph_list = nx.read_adjlist(path)
        elif input_format == "MultilineAdjacencyList":
            graph_list = nx.read_multiline_adjlist(path)
        elif input_format == "EdgeList":
            graph_list = nx.read_edgelist(path)
        elif input_format == "WeightedEdgeList":
            graph_list = nx.read_edgelist(path)
        elif input_format == "GEXF":
            graph_list = nx.read_gexf(path)
        elif input_format == "pickle":
            graph_list = nx.read_gpickle(path)
        elif input_format == "GML":
            graph_list = nx.read_gml(path)
        elif input_format == "GraphML":
            graph_list = nx.read_graphml(path)
        elif input_format == "YAML":
            graph_list = nx.read_yaml(path)
        elif input_format == "graph6":
            graph_list = nx.read_graph6(path)
        elif input_format == "PAJEK":
            graph_list = nx.read_pajek(path)
        else:
            raise Exception('Input format is not supported right now.')

    return graph_list
