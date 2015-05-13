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
        fid = 1

    fig = plt.figure(fid)
    
    if A is not None:
        plt.plot(A[0],A[1], 'o')
    
    #thanks to this http://stackoverflow.com/a/6834693
    plt.plot(
        *zip(*itertools.chain.from_iterable(itertools.combinations(X.T, 2))),
        marker = 'o')
    
    return fig

def compare_estimated(A,ids,est, fid=None):
    """
    this should be wholly unnecessary, but here we are
    """
    
    X = A[:,ids]
    Xest = A[:,est]

    if fid is None:
        fid = 1

    fig = plt.figure(fid)
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

def scaling_energies(scale_data):
    """
    from output of tests.scaling_problem()
    """

    fig, ax1 = plt.subplots()
    
    ax1.plot(scale_data[0], scale_data[1], 'b-')
    ax1.set_xlabel('scaling_factor (k)')
    ax1.set_ylabel('accuracy (percentage)', color='b')
    ax1.set_title('Graph matching with unknown scale factor')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')

    ax2 = ax1.twinx()
    ax2.plot(scale_data[0], scale_data[2], 'r-')
    ax2.set_ylabel('energy of estimated subgraph', color='r')

    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    return fig

def random_graph():
    """eye candy only"""

    from graphmatch import random_vertices
    
    N = 100; n=5
    A, X, ids = random_vertices(N,n)
   
    f = plot_system(A,X)
    f.show()

if __name__ == "__main__":
    
    random_graph()
