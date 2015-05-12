#!/usr/bin/env/python3

from graphmatch import graph_match
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

def scaling_problem():
    
    N = 23
    n = 5
    
    A, X_ex, ids = random_vertices(N, n, transform=True)

    # now scale X by some amount (could be random or fixed)
    X = .5 *X_ex
    
    # show it works
    D = calculate_edges(A) 
    d = calculate_edges(X_ex)
    est, energy = graph_match(D, d)
    fig = compare_estimated(A,ids,est, fid=1)

    #figt = plot_system(None, X, fid=1)
    filename = 'exact.png'
    fig.savefig(filename)
    close(fig)     

    accuracy = get_accuracy(ids, est)
    print("N={}, n={}".format(N,n))
    print('pairwise accuracy: {}%'.format(accuracy))
    print('energy:', energy)

    for k in [1., 2., 3., 4.]:
        D = calculate_edges(A)
        d = calculate_edges(k*X)

        est, energy = graph_match(D,d)

        fig = compare_estimated(A,ids,est, fid=2)

        #figt = plot_system(None, X, fid=1)
        filename = 'k_{}.png'.format(k)
        fig.savefig(filename)
        close(fig)     

        accuracy = get_accuracy(ids, est)
        print("N={}, n={}".format(N,n))
        print('pairwise accuracy: {}%'.format(accuracy))
        print('energy:', energy)
    
def simple(figures=True): 
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
    # you could also just 'extract' this from D if you're clever
    d = calculate_edges(X)

    est, energy = graph_match(D,d)

    fig = compare_estimated(A,ids,est, fid=2)
    
    if transform:
        figt = plot_system(None, X, fid=1)

    accuracy = get_accuracy(ids, est)
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
