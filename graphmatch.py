#!/usr/bin/env/python3

import numpy as np
import numpy.random
import scipy.sparse
import scipy.sparse.linalg

from IPython.core.debugger import Tracer

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


def random_vertices(N,n, clustered=False, transform=False, scale=False):
    """uniformly distributed points in xy-plane.
    a random subset of size n<=N is returned separately. 
    """
    
    assert n <= N

    # A is a 2xN matrix representing N points in the xyplane between [0,100)
    A = 500*np.random.rand(2,N)
    
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
        X - X.min(axis=1).reshape(-1,1)
    
    if scale:
        scale = 100*np.random.rand()
        X *= scale
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
    """
    turn an Nxn matrix into a rectangular "permutation" matrix.
    discretization occurs by choosing the element with highest value compared to
    its column, and then disallowing further selection of elements in its row
    and column. 
    
    returns p, a Nxn matrix, which then can be reduced to an actual vertex to
    vertex association by p.argmax(axis=0)

    note: this implementation is probably very memory inefficent. there's
    probably a way to do this via masked arrays or something. idk
    """

    matches = []
    
    # normalize p (should grab best matches first now)
    p /= p.sum(axis=0)

    mask_val = -np.inf 
    for i in range(p.shape[1]):
        # find the highest remaining element
        w = np.unravel_index(p.argmax(), p.shape)
        # "zero out" its row and column in a noninterfering way (if p was
        # overwhelming negative this could be buggy  

        p[w[0],:] = mask_val
        p[:,w[1]] = mask_val
        matches.append(w) # these will be filled in later.
    
    p = np.zeros_like(p)
    for match in matches:

        p[match] = 1

    # then really these should be sorted by which x corresponds to what
    # just to make the matches clearer

    # of course i could do this in the array interface...
    #m = list(matches)
    #m.sort(key=(lambda x:x[1]))
    

    #return p
    return p


def graph_match(N,n, clustered=False, transform=False, scale=False):

    """
    main function. add description

    returns A, X, ids, est

    s.t. A[:,ids] == X
    """
    assert N >= n

    A, X, ids = random_vertices(N,n, clustered, transform, scale)

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
    x = ev
    descale = C.dot(x)[0,0]
    x /= descale

    # reshape into an Nxn matrix, and we should be able to discretize that into
    # a permutation matrix
    

    xm = x.reshape((N,n))
    #xm = x.reshape((n,N))
    #xm = xm.T

    # i don't know how we picked up the complex parts...
    
    xm = xm.real

    #u, s, v = np.linalg.svd(xm,full_matrices=False)
    
    #x_orth = u.dot(v.T)
    x_orth = xm

    p_est = discretize(x_orth)
    est = p_est.argmax(axis=0)
    
    
    return A, X, ids, est, D, d

def get_accuracy(real_vertices, estimated_vertices):
    """
    inputs are two 1D arrays of the same length, representing
    vertex numbers

    returns total amount of correct matches. note that this could be less than
    what appears visually. for example,
    real -> [2, 3, 4]
    est  -> [3, 2, 4]

    will result in only 33% accuracy reported, even though this graphs might
    look identical visually.
    """


    assert real_vertices.size == estimated_vertices.size

    hits = (real_vertices == estimated_vertices).sum()

    return 100*(hits / real_vertices.size)

if __name__ == "__main__":
    
    from visual import plot_system, compare_estimated
    
    N = 20
    n = 4
    clustered= False
    transform = True
    scale= False

    A, X, ids, est, D, d = graph_match(N,n, clustered=clustered,
            transform=transform, scale=scale)

    fig = compare_estimated(A,ids,est)

    if transform:
        figt = plot_system(None, X)

    accuracy = get_accuracy(ids, est)
    s_accuracy = 100*(1 - len(set(ids) - set(est)) / len(ids))

    print("N={}, n={}".format(N,n))
    #print("correct values:\n", list(ids))
    #print("the guess:\n", list(est))
    print('pairwise accuracy: {}%'.format(accuracy))
    print('setwise accuracy: {}%'.format(s_accuracy))
