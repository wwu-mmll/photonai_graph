from scipy.linalg import svd, schur
import numpy as np


def modal_control(a_mat):
    # This method returns values of MODAL CONTROLLABILITY for each node
    # in a network, given the adjacency matrix for that network. Modal
    # controllability indicates the ability of that node to steer the
    # system into difficult-to-reach states, given input at that node.
    #
    # INPUT:
    #     a_mat is the structural (NOT FUNCTIONAL) network adjacency matrix,
    # 	such that the simple linear model of dynamics outlined in the
    # 	reference is an accurate estimate of brain state fluctuations.
    # 	Assumes all values in the matrix are positive, and that the
    # 	matrix is symmetric.
    #
    # OUTPUT:
    #     Vector of modal controllability values for each node
    #
    # Reference: Gu, Pasqualetti, Cieslak, Telesford, Yu, Kahn, Medaglia,
    #            Vettel, Miller, Grafton & Bassett, Nature Communications
    #            6:8414, 2015.
    u, s, vt = svd(a_mat)  # singluar value decomposition
    a_mat = a_mat / (1 + s[0])  # s is a eigen-value
    t_mat, u_mat = schur(a_mat, 'real')  # Schur stability
    eig_vals = np.diag(t_mat)
    n_mat = a_mat.shape[0]
    phi = np.zeros(n_mat, dtype=float)
    for i in range(n_mat):
        a_left = u_mat[i, ] * u_mat[i, ]  # element-wise multiplication
        a_right = (1.0 - np.power(eig_vals, 2)).transpose()
        phi[i] = np.matmul(a_left, a_right)
    return phi


def average_control(a_mat):
    # This program is a Python version of average controllability
    #     This function returns values of AVERAGE CONTROLLABILITY for
    #     each node in a network, given the adjacency matrix for that network.
    #     Average controllability measures the ease by which input at
    #     that node can steer the system into many easily-reachable states.
    #
    # INPUT:
    #     A is the structural (NOT FUNCTIONAL) network adjacency matrix,
    #     such that the simple linear model of dynamics outlined in the%
    #     reference is an accurate estimate of brain state fluctuations.
    #     Assumes all values in the matrix are positive, and that the
    #     matrix is symmetric.
    #
    # OUTPUT:
    #     Vector of average controllability values for each node
    #
    #     Reference: Gu, Pasqualetti, Cieslak, Telesford, Yu, Kahn, Medaglia,
    #             Vettel, Miller, Grafton & Bassett, Nature Communications
    #             6:8414, 2015.
    # %%
    n_mat = a_mat.shape[0]
    u, s, vt = svd(a_mat)  # singluar value decomposition
    a_mat = a_mat / (1 + s[0])  # s is a eigen-value
    t_mat, u_mat = schur(a_mat, 'real')  # Schur stability
    mid_mat = np.multiply(u_mat, u_mat).transpose()  # element-wise multiplication
    v = np.matrix(np.diag(t_mat)).transpose()

    # %%
    p_mat = np.diag(1 - np.matmul(v, v.transpose()))
    p_mat = np.tile(p_mat.reshape([a_mat.shape[0], 1]), (1, n_mat))
    values = sum(np.divide(mid_mat, p_mat))
    return values
