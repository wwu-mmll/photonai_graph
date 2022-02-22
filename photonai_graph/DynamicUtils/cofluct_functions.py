import numpy as np
from scipy.stats import zscore
import warnings


def cofluct(X, quantiles: tuple = (0, 1), return_mat=True):
    """
    Computes cofluctuation time-series (per edge) for a nodes x timepoints matrix X.
    Based on https://www.pnas.org/content/early/2020/10/21/2005531117

    Parameters
    ----------
    quantiles: tuple,default=(0,1)
        list of lowest/highest quantile of edge events to use
        [0, 1]: all events = pearson corr; [0, .05]: bottom 5% of events; [.95, 1]: top 5% of events
    return_mat: bool,default=True
        Whether to return a connectivity matrix (True) or dictionary (False). The dict edge contains cofluctuation
        time-series (pairs_of_nodes x timepoints) and event timeseries.


    Returns
    -------
    float
        edge cofluctuation time-series dict (pairs_of_nodes x timepoints) and event timeseries as dict

    """
    X = np.float32(X)
    # get z-value time series (mean 0, std 1)
    z_ts = zscore(X, axis=1)

    # multiply z-scored time series to get edge time series (co-fluctuation)
    edge_ids = np.triu_indices(X.shape[0], k=1)
    edge_ts = np.einsum('ik,jk->ijk', z_ts, z_ts)[edge_ids]     # get upper triangular values

    # get event_ts as Root Mean Square over nodes
    event_ts = np.sqrt(np.mean(np.square(edge_ts), axis=0))

    # get correlations for quantile intervals of events
    q_low = np.quantile(np.abs(event_ts), q=quantiles[0])
    q_high = np.quantile(np.abs(event_ts), q=quantiles[1])
    edge_ts = edge_ts[:, (event_ts >= q_low) & (event_ts <= q_high)]

    # compute correlation/connectivity vector or matrix
    cors = np.mean(edge_ts, axis=1)
    if return_mat:
        tmp = np.eye(X.shape[0])
        tmp[np.triu_indices(tmp.shape[0], k=1)] = cors
        cors = tmp + tmp.T
        np.fill_diagonal(cors, 1)

    # return as dict
    else:
        warnings.warn("dicts can not be handled by photonai."
                      "Dicts are implemented for use outside of Photon.")
        edge_ids_str = [str(m) + '_' + str(n) for m, n in zip(edge_ids[0], edge_ids[1])]
        edge_ts_dict = {key: val for val, key in zip(edge_ts, edge_ids_str)}
        edge_ts_dict['event_ts'] = event_ts
        return edge_ts_dict

    return cors
