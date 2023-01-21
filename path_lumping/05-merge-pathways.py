import numpy as np
import numba as nb

def __path_compare(path1, path2):

    samebool = 1
    if len(path1) != len(path2):
        samebool = 0
    else:
        for i in range(len(path1)):
            if int(path1[i]) != int(path2[i]):
                samebool = 0
                break
    return samebool


def get_merged_pathway(pathways, fluxes):
    
    count = 0
    visitbool = np.zeros(len(pathways))
    finalflux = {}
    for i in range(len(pathways)):
        finalflux[i] = 0.0
    finalpath = {}
    for i in range(len(pathways)):
        if visitbool[i] == 1:
            continue
        finalflux[count] += fluxes[i]
        for j in range(i+1, len(pathways), 1):
            if len(pathways[j]) == len(pathways[i]) and visitbool[j] == 0 and __path_compare(pathways[i], pathways[j]) == 1:
                finalflux[count] += fluxes[j]
                visitbool[1]
        finalpath[count] = pathways[i]
        count += 1
    
    # sort the merged pathways by their new merged fluxes;
    sortfinalflux = []
    sortfinalpath = []
    __finalflux = np.array([v for v in finalflux.values()])
    __idx = np.argsort(__finalflux)
    __pathnum = len(__idx)
    for i in range(len(__idx)):
        sortfinalpath.append(finalpath[int(__idx[__pathnum-1-i])])
        sortfinalflux.append(finalflux[int(__idx[__pathnum-1-i])])
    return sortfinalpath, sortfinalflux



pathways = np.load("tpt_dijkstra_pathways.npy", allow_pickle=True)
fluxes = np.load("tpt_dijkstra_pathways_fluxes.npy", allow_pickle=True)

mergepath, mergeflux = get_merged_pathway(pathways, fluxes)


