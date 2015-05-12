#!/usr/bin/env/python3

import numpy as np
import numpy.random
import scipy.sparse
import scipy.sparse.linalg
from graphs import random_vertices, calculate_edges 


def build_similarities(D,d):
    """ construct a similarity matrix Nn x Nn where each element represents the
    'cost' of matching (i -> i') and (j -> j') in the two graphs.
    
    this is the matrix W

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

    return p


def graph_match(D,d):

    """
    performs the graph matching problem given two matrices of edge attributes
    D (larger graph), d (subgraph)
    
    returns est, a 1d array such that
        est[i] = j  <--> xi matches with aj

    also returns the energy to be maximized,j xT W x (see paper)
    """
    N = D.shape[0]
    n = d.shape[0]

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

    # i don't know how we picked up the complex parts, but they're zero
    xm = xm.real

    p_est = discretize(xm)
    est = p_est.argmax(axis=0)
    
    # now reshape back to a row vector (now discretized) and calculate energy
    x = p_est.reshape((N*n, 1))
    energy = x.T.dot(W).dot(x)
    
    return est, energy

if __name__ == "__main__":
    
    from tests import simple
    simple()
