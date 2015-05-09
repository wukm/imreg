#!/usr/bin/env/python3

import numpy as np
import numpy.random
import scipy.sparse
import scipy.sparse.linalg

def row_norm_matrix(A, shape=None):
    """
    needed for forming M
    if A is an nxd matrix, return an MxN matrix that
    looks like

    [[ ||x1||^2 ||x1||^2 ... ||x1||^2 ]
     [ ||x2||^2 ||x2||^2 ... ||x2||^2 ]
     ...
     [ ||xM||^2 ||xM||^2 ... ||xM||^2 ]]

    where xi represents a row of X.

    this subfunction is useful for making the gaussian kernel and
    testing whats-it matrix N
    """
   
    # really making shape a tuple is just to make the interface more foolproof
    assert shape[0] == A.shape[0], "can't broadcast to this shape!"

    n = shape[1]

    # a flattened list of row two norms of rows in A (data points)
    row_norms = [row.T.dot(row) for row in A]

    Q = np.array(row_norms, ndmin=2).T
    Q = np.tile(Q, (1, shape[1]))

    return Q


def random_vertices(N,n):
    """uniformly distributed points in xy-plane.
    a random subset of size n<=N is returned separately. 
    """
    
    assert n <= N

    # A is a 2xN matrix representing N points in the xyplane between [0,100)
    A = 100*np.random.rand(2,N)
    
    # the following columns are to be extracted
    # these can serve as identifiers/labels for X
    id_extracted = np.random.permutation(N)[:n] 
    
    # this is now a 2xn matrix whose ith column
    X = A[:,id_extracted]
   
    return A, X, id_extracted 


def calculate_edges(M):
    """
    M.shape == (2,m)
    returns a matrix D of shape mxm
    where D_ij = ||M_i - M_j||
    # i actually want these square rooted, so maybe there's a better way to
    # construct
    """
    M = M.T # we want to sum along columns and the calculations below use rows

    n = M.shape[0] 
    D = row_norm_matrix(M, (n,n))

    D = D - M.dot(M.T)
    D = D + D.T

    return np.sqrt(D)

def build_similarities(D,d):
    """ construct a similarity matrix Nn x Nn where each element represents the
    'cost' of matching (i -> i') in the two graphs.

    d -> edge values of small subgraph (X)
    D -> edge values of large graph (A)


    """
    W = np.tile(d, D.shape)

    W0 = np.kron(D, np.ones(d.shape))
    
    W = W - W0 

    return abs(W)

def build_P(N,n):

    # these are the N "one-to-one" constraints, shape (N,N*n)
    ii = [i//n for i in range(N*n)]
    jj = [i for i in range(n*N)]
    C = scipy.sparse.csc_matrix((np.ones(n*N),(ii,jj)))
    # okay, just undo it for now because i wanna simplify
    C = C.toarray()

    # now i'm just following notation from the paper
    b = np.ones((N,1))
    Ck = C[-1:]
    
    Ceq = np.eye(N-1,N).dot(C - b.dot(Ck))

    P = Ceq.dot(Ceq.T)
    P = np.linalg.inv(P)
    P = Ceq.T.dot(P)
    P = P.dot(Ceq)
    P = np.eye(N*n) - P

    return P, C
    
def discretize(p):

    guess = p.argmax(axis=0)

    if len(set(guess)) != p.shape[1]:
        print("there was a collision")
    return guess

def graph_match(N,n):

    """
    main function. add description

    returns A, X, ids, est

    s.t. A[:,ids] == X
    """
    assert N >= n

    A, X, ids = random_vertices(N,n)

    # these are now the graph attributes of each graph
    D = calculate_edges(A)
    # you could also just 'extract' this from D if you're clever
    d = calculate_edges(X)

    # at least on my chromebook, np.exp(W*W) just makes everything 0. i'm just gonna
    # invert this locally
    
    # note result is NOT inverted yet
    W = build_similarities(D,d)
    # invert it (gauss kernel seems too small, idk)
    W = 1 / (W + 1) 
    
    P, C = build_P(N,n)

    #S = PWP
    S = P.dot(W)
    S = S.dot(P) 
    
    # largest eigenvalue and eigenvector of the system
    # for some reason, it returns these as complex numbers but the imag part is
    # 0
    el, ev = scipy.sparse.linalg.eigs(S, 1)
   
    # now THIS is our solution, which much be normalized
    x = el*ev
    descale = C.dot(x)[0,0]
    x /= descale

    # reshape into an Nxn matrix, and we should be able to discretize that into
    # a permutation matrix

    xm = x.reshape((N,n)).real

    u, s, v = np.linalg.svd(xm,full_matrices=False)
    
    x_orth = u.dot(v.T)

    est = discretize(x_orth)

    print("N={}, n={}".format(N,n))
    #print("correct values:\n", set(ids))
    #print("the guess:\n", set(est))
    print("how'd we do:", set(ids) - set(est))
    
    return A, X, ids, est
if __name__ == "__main__":
    
    from visual import plot_system, compare_estimated

    N = 20
    n = 5
    
    A, X, ids, est = graph_match(N,n)

    #f1 = plot_system(A,X, fid=1)
    #Xest = A[:,est] 
    #f2 = plot_system(A,Xest, fid=2)

    fig = compare_estimated(A,ids,est)

