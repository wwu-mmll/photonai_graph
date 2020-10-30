# Dynamic Utilities

A module focused on functions and constructors for dynamic graph data.

!!! note "Warning" Currently under development and constantly evolving. Use with caution.

## CofluctTransform

Cofluctuation transformer, that takes a timeseries of node signals and calculates the cofluctuation between nodes for a selected quantile. Returns an adjacency matrix. Can also return edge time series.

| Parameter | type | Description |
| -----     | ----- | ----- |
| quantiles | tuple, default = (0, 1) | lower and upper bound of connection strength quantile to look at |
| return_mat | bool, default=True | Whether to return matrix (True) or dictionary (False). Matrix is required inside a PHOTONAI pipeline. |
| adjacency_axis | int, default=0 | position of the adjacency matrix, default being zero |

#### Example

```python
transformer = CofluctTransform(quantiles=(0.95, 1), return_mat=True)
```

## Cofluctuation function (cofluct)

Computes cofluctuation time-series (per edge) for a nodes x timepoints matrix. Based on https://www.pnas.org/content/early/2020/10/21/2005531117.

| Parameter | type | Description |
| -----     | ----- | ----- |
| quantiles | tuple, default=(0, 1) | tuple of lowest/highest quantile of edge events to use |
| return_mat | bool, default=True | Whether to return a connectivity matrix (True) or dictionary (False). The dict edge contains cofluctuation time-series (pairs_of_nodes x timepoints) and event timeseries. |
