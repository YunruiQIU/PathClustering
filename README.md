PathClustering
==========
Cluster the transition pathways according to their kinetics;

## Markov State Model (MSM) 
MSM has power to elucidate the kinetic mechanism of complicated dynamical system.
It converts the high-dimensional time sequence to transitions between few microstates.

```console
Recommended references : J. Chem. Phys. 134, 174105 (2011); MRS Bulletin 47, 958–966 (2022);
```
## Transition Path Theory (TPT) 
TPT is able to identify transition pathways between source microstates and sink microstates based on constructed MSM.
It generates ensemble of paths going through microstates and their corresponding fluxes.
```console
Recommended references : Proc. Natl. Acad. Sci. USA 106, 19011-19016 (2009); Journal of Statistical Physics 123, 503-523 (2006);
```

## Path Lumping Algorithm
There usually exsits plenty of parallel pathways which share the comparable fluxes in multi-body systems. To better understand the sophisticated dynamics of these systems, it's necessary to cluster a large number of parallel pathways to obtain fine-grained metastable path channels. 

Path Lumping algorithm uses interfluxes between pathways as criteria to lump paths into few metastable channels by spectral clustering. Different from TPT, Path Lumping uses flux matrix to identify the pathways and specially considers the loopless flux matrix.
```console
Recommended references : J. Chem. Phys. 147, 044112 (2017)
```
#### Details procedues:
(1). Committor_Probability_Calculation: compute committor probabilities between each pair of microstates by solving linear equation;

(2). Flux_Matrix_Calculation: construct flux matrix according to net fluxes;

(3). Loopless_Flux_Calculation: construct loopless flux matrix by removing recrossing and loops; (geometric tricks)

(4). Merge-Pathways: merge repeated pathways;

(5). Inter_Path_Flux: scan the interfluxes between each pair of paths;

(6). Spectral_Cluster_Path_Channel : cluster paths based on interfluxes matrix by spectral-clustering
