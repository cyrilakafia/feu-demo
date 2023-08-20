import matplotlib.pyplot as plt
import matplotlib as mpl

from utils.data import read_pickle, load_network
from utils.vis import reorder_cluster_ids, vis_heatmap

def visualize_heatmap(filename):
    out = read_pickle(filename)

    cluster_ids, cluster_params, n_clusters = reorder_cluster_ids(*out)

    burnin = 1000
    sim_mat, all_best_gibbs, all_sim_mats = vis_heatmap(cluster_ids)
    return sim_mat, all_best_gibbs, all_sim_mats