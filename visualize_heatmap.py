import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def ids_to_coo(ids):
    '''
    Convert cluster assignments to a coo matrix

    Args:
    ids (np.array): the cluster assignments

    Returns:
    coo (np.array): the coo matrix
    '''
    
    coo = np.zeros((len(ids), len(ids)))
    for u in np.unique(ids):
        mask = ids == u
        coo[np.ix_(mask, mask)] = 1
    return coo

def viz_heatmap(run, iter):
    '''
    Visualize the cluster assignments as a heatmap

    Args:
    run (str): the run number
    iter (int): the number of iterations

    Returns:
    best_assigns (pd.DataFrame): the best cluster assignments
    
    Saves a heatmap of the cluster assignments to images/heatmap.png
    '''

    if iter < 510:
        assigns = pd.read_csv(f"outputs/sim{run}_assigns.csv", header=None)
    else:
        assigns = pd.read_csv(f"outputs/sim{run}_assigns.csv", header=None).iloc[500:].reset_index(drop=True)

    coos = np.stack([ids_to_coo(x) for _, x in assigns.iterrows()])
    mean_coo = coos.mean(axis=0)
    dists = np.linalg.norm(coos - mean_coo, axis=(1, 2))
    min_dist_idx = np.argmin(dists)
    match_idxs = np.where(np.all(coos == coos[min_dist_idx], axis=(1, 2)))[0]

    # save to file
    best_assigns =  pd.DataFrame(assigns.loc[min_dist_idx])
    best_assigns.columns = ['cluster id']

    # rotate the df
    best_assigns = best_assigns.T

    reorder = assigns.loc[min_dist_idx].sort_values().index

    # plot cluster assignments
    plt.imshow(mean_coo[np.ix_(reorder, reorder)], cmap="coolwarm")
    plt.colorbar()

    N = len(assigns.columns)

    plt.yticks(np.arange(N), reorder, fontsize=5)
    plt.xticks(np.arange(N), reorder, fontsize=5, rotation=90)
    plt.tight_layout()
    if not os.path.exists('images/heatmap.png'):
        plt.savefig('images/heatmap.png', dpi = 500)
    else:
        plt.savefig('images/heatmap_2.png', dpi = 500)

    return best_assigns

