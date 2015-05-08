#!/usr/bin/env python3

"""
using matplotlib for visualizing the system
"""

import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_system(A,X): 
    """
    returns a pyplot figure, which can be viewed with
    fig.show()
    """
    fig = plt.figure(1)
    plt.plot(A[0],A[1], 'o')
    
    #thanks to this http://stackoverflow.com/a/6834693
    plt.plot(
        *zip(*itertools.chain.from_iterable(itertools.combinations(X.T, 2))),
        marker = 'o')
    
    return fig
if __name__ == "__main__":

    from graphmatch import random_vertices
    
    N = 100; n=5
    A, X, ids = random_vertices(N,n)
   
    f = plot_system(A,X)
    f.show()

    
