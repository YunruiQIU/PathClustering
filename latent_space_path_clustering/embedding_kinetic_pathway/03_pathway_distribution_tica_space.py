import numpy as np

"""
Concatenate pathway distribution in each pairs of tICs space into 1-D distribution vector (input for VAE);
Here is an example for 3-D tICA space recombination;
"""

## number of discretized slices along each collective variable 
num_slices = 30

## Load the pathways identified by Transition Path Theory, each pathway is a sequence of state indexes
paths = np.load("./167bps_6pdis_mirror_cut_0.25ns3tics_530kmeans_trans_2.5nstpt_paths.npy", allow_pickle=True)

## Input number of pathways with largest flux to embed
num_pathways = 5000

## The directories of state distribution (in each pair of collective variables space)
dirc1 = 'microstates_distribution_tica01'
dirc2 = 'microstates_distribution_tica02'
dirc3 = 'microstates_distribution_tica12'


for i in range(0, num_pathways):
    f = paths[i]
    dist = np.zeros((3*num_slices, num_slices))
    temp = 0
    for j in range(len(f)):
        mat1 = np.load("./"+dirc1+"/%03d_state_distribution.npy"%int(f[j]), allow_pickle=True)
        mat2 = np.load("./"+dirc2+"/%03d_state_distribution.npy"%int(f[j]), allow_pickle=True)
        mat3 = np.load("./"+dirc3+"/%03d_state_distribution.npy"%int(f[j]), allow_pickle=True)

        dist[:num_slices] = dist[:num_slices] + mat1 
        dist[num_slices:2*num_slices] = dist[num_slices:2*num_slices] + mat2 
        dist[2*num_slices:3*num_slices] = dist[2*num_slices:3*num_slices] + mat3 

    print("No {} transition pathway is calculated as distribution;".format(i))
    np.save("./tpt_path_distribution/No_%04d_path_distribution.npy"%i, dist)