# Graph Measures

Graph measures or metrics are values that capture graph properties like efficiency. As these measures capture information across the entire graph, or for nodes, edges or subgraphs, they can be used to study graph properties. These measures can also be used as a low-dimensional representation of the graph in machine learning tasks.

# GraphMeasureTransform

The GraphMeasureTransform class is a class for extracting graph measures from graphs. A range of networkx graph measures is available and can be used in a PHOTON pipeline or extracted and written to a csv file for further use.

| Parameter | type | Description |
| -----     | ----- | ----- |
| graph_functions | dict | a dict of graph functions to calculate with keys as the networkx function name and a dict of the arguments as the value. In this second dictionary, the keys are the functions arguments and values are the desired values for the argument. |
| verbosity | int, default=0 | The level of verbosity, 0 is least talkative and gives only warn and error, 1 gives adds info and 2 adds debug |
| logs | str, default=None | path to the log data |
