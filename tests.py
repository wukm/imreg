#!/usr/bin/env/python3

import numpy as np
from graphmatch import graph_match, match_energy, build_similarities
from graphs import *
from visual import compare_estimated, plot_system
from matplotlib.pyplot import close

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

def scaling_problem(make_figures=False):
    
    N = 23
    n = 4
    mesh_size = 50

    A, X_ex, ids = random_vertices(N, n, transform=True)

    # now scale X by some amount
    scale_by = np.random.uniform(low=1, high=10)
    X = X_ex / scale_by
    
    # test whether it works on the exact subgraph
    D = calculate_edges(A) 
    d = calculate_edges(X_ex)
    
    W = build_similarities(D,d)
    exact_energy = match_energy(W, ids, shape=(N,n))

    #del W
    print('exact energy:', exact_energy)

    est, energy = graph_match(D, d)
    accuracy = get_accuracy(ids, est)

    if make_figures or accuracy != 100:
        filename = 'figures/exact.png'
        fig = compare_estimated(A,ids,est, fid=1)
        fig.savefig(filename)
        close(fig)     

    print("N={}, n={}".format(N,n))
    print('correct scaling:', scale_by)
    print('pairwise accuracy (at correct scaling): {}%'.format(accuracy))
    print('energy:', energy)
   
    d_unscaled = calculate_edges(X) 
    # get the minimum and maximum possible scaling factors for the two graphs
    kmin, kmax = scale_range(D,d_unscaled) 

    for k in np.linspace(kmin,kmax, num=mesh_size):

        d = calculate_edges(k*X)

        est, energy = graph_match(D,d)
        accuracy = get_accuracy(ids, est)

        if make_figures:
            filename = 'figures/k_{}.png'.format(k)
            fig = compare_estimated(A,ids,est, fid=2)
            fig.savefig(filename)
            close(fig)     

        print('*'*40)
        print('k={}'.format(k))
        print('pairwise accuracy: {}%'.format(accuracy))
        print('energy:', energy)
    
def simple(make_figures=True): 
    """
    basic test, one random system
    """
    N = 30
    n = 10

    clustered= False
    transform = True
    scale = False

    assert N >= n

    A, X, ids = random_vertices(N,n, clustered, transform, scale)
    
    # these are now the graph attributes of each graph
    D = calculate_edges(A)
    d = calculate_edges(X)

    est, energy = graph_match(D,d)
    accuracy = get_accuracy(ids, est)

    if make_figures:
        fig = compare_estimated(A,ids,est, fid=2)
    
        if transform:
            figt = plot_system(None, X, fid=1)

    print("N={}, n={}".format(N,n))
    print('pairwise accuracy: {}%'.format(accuracy))
    

#def vary_sizes(figures=False):
#    """
#    finish this. this shows how the algorithm fares for different sizes of the
#    system, as well as size of the subgraph relative.
#    """
#    N = 30
#
#    percents = [.10,.33,.66,.90]
#
#
#    for p in percents:
#        n = max(int(N*p),1)
#    
#    assert N >= n
#
#    clustered= False
#    transform = True
#    scale= False
#
#    A, X, ids = random_vertices(N,n clustered,transform,scale)
#
#
#    # these are now the graph attributes of each graph
#    D = calculate_edges(A)
#    # you could also just 'extract' this from D if you're clever
#    d = calculate_edges(X)
#
#    est = graph_match(D,d)
#
#    #fig = compare_estimated(A,ids,est)
#
#    #if transform:
#    #    figt = plot_system(None, X)
#
#    accuracy = get_accuracy(ids, est)
#
#    print("N={}, n={}".format(N,n))
#    print('pairwise accuracy: {}%'.format(accuracy))

if __name__ == "__main__":

    scaling_problem()
