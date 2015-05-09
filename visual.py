#!/usr/bin/env python3

"""
using matplotlib for visualizing the system
"""

import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_system(A,X, fid=None): 
    """
    returns a pyplot figure, which can be viewed with
    fig.show()
    """
    if fid is None:
        fig = plt.figure(1)
    else:
        fig = plt.figure(fid)
    plt.plot(A[0],A[1], 'o')
    
    #thanks to this http://stackoverflow.com/a/6834693
    plt.plot(
        *zip(*itertools.chain.from_iterable(itertools.combinations(X.T, 2))),
        marker = 'o')
    
    return fig

def compare_estimated(A,ids,est):
    """
    this should be wholly unnecessary, but here we are
    """
    
    X = A[:,ids]
    Xest = A[:,est]

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    for ax, ind in zip((ax1, ax2), (ids, est)):
        
        x = A[:,ind]
        ax.plot(A[0],A[1], 'o')

        #thanks to this http://stackoverflow.com/a/6834693
        ax.plot(
            *zip(*itertools.chain.from_iterable(itertools.combinations(x.T, 2))),
            marker = 'o')
   
    return fig
if __name__ == "__main__":

    from graphmatch import random_vertices
    
    N = 100; n=5
    A, X, ids = random_vertices(N,n)
   
    f = plot_system(A,X)
    f.show()

    
