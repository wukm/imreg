#!/usr/bin/env python3

import numpy as np
import numpy.random
import scipy.sparse
import scipy.sparse.linalg

"""
All functions in graphs.py relate to generating sample graphs (and subgraphs)
as problems for graphmatch.
"""

def _row_norm_matrix(A, shape=None):
    """
    return a row norm matrix from A with the shape provided

    if A is an nxd matrix, return an nxr matrix that
    looks like

    [[ ||x1||^2 ||x1||^2 ... ||x1||^2 ]
     [ ||x2||^2 ||x2||^2 ... ||x2||^2 ]
     ...
     [ ||xn||^2 ||xn||^2 ... ||xn||^2 ]]

    where xi represents a row of X.
    
    shape provided must satisfy shape[0] == A.shape[0]

    used for computing edge distances between points in a graph
    """
   
    # really making shape a tuple is just to make the interface more foolproof
    assert shape[0] == A.shape[0], "can't broadcast to this shape!"

    n = shape[1]

    # a flattened list of row two norms of rows in A (data points)
    row_norms = [row.T.dot(row) for row in A]

    Q = np.array(row_norms, ndmin=2).T
    Q = np.tile(Q, (1, shape[1]))

    return Q

def random_vertices(N,n, clustered=False, transform=False, scale=False):
    """uniformly distributed points in xy-plane.
    a random subset of size n<=N is returned separately. 
    
    NOTE: scale X outside of this. scale parameter is deprecated
    """
    
    assert n <= N

    # A is a 2xN matrix representing N points in the xyplane between [0,100)
    A = 100*np.random.rand(2,N)
    
    if clustered:
        # get a random point
        x0 =  A[:,np.random.randint(A.shape[1])]
        # get n nearest points to x0
        x0 = x0.reshape(-1,1)
        s = ((A*A).sum(axis=0) - (x0*x0).sum(axis=0)).argsort()

        id_extracted = s[:n]
        X = A[:,id_extracted]
    
    else:
        # the following columns are to be extracted
        # these can serve as identifiers/labels for X
        id_extracted = np.random.permutation(N)[:n] 
        
        # this is now a 2xn matrix whose ith column
        X = A[:,id_extracted]
  
    if transform:
        # get a random theta
        theta = 2*np.pi*np.random.rand()
        T = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])

        X = T.dot(X)

        # the affine translation bit will literally just be zeroing out
        X -= X.min(axis=1).reshape(-1,1)
    
    if scale:
        scale = 100*np.random.rand()
        X *= scale

    return A, X, id_extracted 

def calculate_edges(M):
    """
    INPUT
    M:  a (2,m) array representing m points in R^2

    OUTPUT
    D:  a (m,m) array whose where Dij represents euclidean distance between
        Mi,Mj (two points corresponding to the ith and jth col. of M)

    Note: There is potentially a better way to construct these, since I am now
    interested in the sqrt. look at numpy.linalg.norm(a, ord=2, axis=0)
    """
    M = M.T # we want to sum along columns and the calculations below use rows

    n = M.shape[0] 
    D = _row_norm_matrix(M, (n,n))

    D = D - M.dot(M.T)
    D = D + D.T

    return np.sqrt(D)

def scale_range(D,d):
    """
    get a lower and upper bound for the scaling problem
   
    INPUT
    d: a matrix, subgraph edge attributes (distances)
    D: a matrix, graph edge attributes (distances)
    
    OUTPUT
    (kmin, kmax) - two real numbers representing the minimum and maximum
    possible scaling factors to make k*d represent a subgraph of D
    """

    # get minimum and maximum distance in the graph
    Amax = D.max()
    Amin = D[D > 0].min() # since D has zeros on diagonals

    # ditto for subgraph
    amax = d.max()
    amin = d[d>0].min() # d has zeros on diagonals

    return Amin/amin, Amax/amax

