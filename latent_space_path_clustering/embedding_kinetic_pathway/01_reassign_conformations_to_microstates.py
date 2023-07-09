import numpy as np
from tqdm import trange

"""
Re-assign MD conformations in tICA space back to microstates
"""

## Import the tICA projected trajectories or any trajectories in the space you want to perform path clustering
## The trajectories should have list format
distrajs = np.load("./tica_trajs.npy", allow_pickle=True).item()
distrajs = [i for i in distrajs.values()]

## Import the microstates based clustered trajectories
## The trajectories should have list format
ctrajs = np.load("./cluster_kmeans_trajs.npy", allow_pickle=True)

## Set the reassign step, number of microstates, dimensionality of tICA
step = 1  ## The frequency of reassign
num_clusters = 530 
dim_tica = 6

reassign_trajs = {}
for i in trange(len(ctrajs)):
    for j in range(0, len(ctrajs[i]), step):
        try:
            reassign_trajs[int(ctrajs[i][j])] = np.vstack((reassign_trajs[int(ctrajs[i][j])], distrajs[i][j]))
        except KeyError:
            reassign_trajs[int(ctrajs[i][j])] = distrajs[i][j]
np.save("reassign_tica2micro.npy", reassign_trajs)
