import numpy as np
import os

def microstates_distribution(reassign_trajs, num_clusters, idx1, idx2, x_initial, x_end,  y_initial, y_end, 
                             num_slices, output_dir="microstates_distribution_tica"):
    """
    Use the re-assigned conformations (belonging to every microstates) to get the mcirostate distribution;
    The simplest way is to discretize/grid the tICA space into bins and count the distribution of conformations
    in each bin; 
    The number of bins increase exponential o(N^3) with the dimensionality of tICA space; To simplify the input 
    of VAE,  we can visualize pathways on each pair of 2-D tICA space and concatenate the partial distributions;
    In this way, the complexity incease much slower o(N^2) with the dimensionarlity of tICA space.
    ---parameters---
    reassign_trajs: the ressigned tICA conformatiosn for every microstates
    num_clusters: the number of microstates (to calculate distribution)
    idx1: the dimensional index of the first embedding coordinate
    idx2: the dimensional index of the second embedding coordinate
    x_initial: the minimum value of the first embedding coordinate
    x_end: the maximum value of the first embedding coordinate
    y_initial: the minimum value of the second embedding coordinate
    y_end: the maximum value of the second embedding coordinate
    num_slices: the number of slices/bins to discretize along each coordinate
    output_dir: the output directory of the microstate distribution
    """

    if os.path.exists(output_dir):
        print("Attention! The ouput file already exists!")
    else:
        os.makedirs(output_dir)

    xdelta = (x_end - x_initial) / num_slices
    ydelta = (y_end - y_initial) / num_slices
    for i in range(num_clusters):
        dist = np.zeros((num_slices, num_slices))
        print("For No {} state, in total, there are {} frames;".format(i, len(reassign_trajs[i])))
        for j in range(0, len(reassign_trajs[i]), 1):
            x = (reassign_trajs[i][j, idx1]-x_initial) // xdelta
            y = (reassign_trajs[i][j, idx2]-y_initial) // ydelta
            if 0 < x < num_slices and 0 < y< num_slices:
                dist[num_slices-int(y)-1, int(x)] += 1
        dist = dist / (len(reassign_trajs[i]))
        np.save(output_dir + "/%03d_state_distribution.npy"%(i), dist)

        print("No {} state is completed for distribution calculation;".format(i))



## Load the reassigned trajectories (conformations belonging to every state)
reassign_trajs = np.load("reassign_tica2micro.npy", allow_pickle=True).item()
reassign_trajs = [i for i in reassign_trajs.values()]
microstates_distribution(reassign_trajs=reassign_trajs, num_clusters=500, idx1=0, idx2=1, x_initial=-5, x_end=3, y_initial=-5, y_end=3, 
                             num_slices=30, output_dir="microstates_distribution_tica01")