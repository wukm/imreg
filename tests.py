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

def scaling_problem(summary_figure=False, make_figures=False):
    
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

    del W
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
    
    all_data = []

    best_accuracy = 0
    best_energy =  0
    estimated_scale = 0

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
        
        all_data.append([k, accuracy, energy])
        if energy > best_energy:
            best_energy=energy
            best_accuracy=accuracy
            estimated_scale=k

    print('*'*80)
    print("best energy: ", best_energy)
    print("corresp. accuracy: ", best_accuracy)
    print("estimated scale_factor", estimated_scale)
    print("real scale value:", scale_by)
    
    return np.array(all_data).T

def simple(make_figures=True): 
    """
    basic test, one random system
    """
    N = 60
    n = 5

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
    print('pairwise accuracy of match: {}%'.format(accuracy))
    print('energy of match:', energy)
   
    return accuracy, energy

def simple_average():
    
    accuracies = 0
    energies = 0
    trials = 200
    for i in range(trials):
        a, e = simple(make_figures=False)
        accuracies += a
        energies += e
    
    accuracies /= trials
    energies /= trials

    print("average accuracy over {} trials: {}%".format(trials, accuracies))
    print("average energy:", energies)

def vary_sizes(figures=False):
    """
    finish this. this shows how the algorithm fares for different sizes of the
    system, as well as size of the subgraph relative.
    """
    N = 60
    n = 5 
    
    assert N >= n

    clustered= False
    transform = True
    scale= False

    A, X, ids = random_vertices(N,n, clustered,transform,scale)


    # these are now the graph attributes of each graph
    D = calculate_edges(A)
    # you could also just 'extract' this from D if you're clever
    d = calculate_edges(X)

    est, energy = graph_match(D,d)

    #fig = compare_estimated(A,ids,est)

    #if transform:
    #    figt = plot_system(None, X)

    accuracy = get_accuracy(ids, est)

    print("N={}, n={}".format(N,n))
    print('pairwise accuracy: {}%'.format(accuracy))

if __name__ == "__main__":

    from visual import scaling_energies
    data = scaling_problem()
    f = scaling_energies(data)
