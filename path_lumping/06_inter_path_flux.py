"""
Author: Yunrui QIU     email: yunruiqiu@gmail.com
2022-Dec-20th
"""

import numpy as np
import numba as nb

def __flux_matrix(num_states, pathways, fluxes):
    fluxmatrix = np.zeros((num_states, num_states))
    for i in range(len(pathways)):
        for j in range(len(pathways[i])-1):
            fluxmatrix[int(pathways[i][j])][int(pathways[i][j+1])] += fluxes[i]
    return fluxmatrix


def __state_path_flux(num_states, pathways, fluxes, fluxmatrix):
    stateflux = np.sum(fluxmatrix, axis=1)
    statepathmatrix = np.zeros((num_states, len(pathways)))
    for i in range(num_states):
        for j in range(len(pathways)):
            if float(i) in pathways[j] and i != int(pathways[j][-1]):
                statepathmatrix[i, j] = fluxes[j] / stateflux[i]
    return stateflux, statepathmatrix


def interpathflux(num_states, pathways, fluxes, source, sink):
    f = fluxes / np.sum(fluxes)
    fluxmatrix = __flux_matrix(num_states=num_states, pathways=pathways, fluxes=f)
    stateflux, statepathmatrix = __state_path_flux(num_states, pathways, f, fluxmatrix)
    tpm = np.zeros((len(pathways), len(pathways)))
    for i in range(len(pathways)):
        for j in range(len(pathways)):
            for k in range(1, len(pathways[i])-1):
                for p in range(0, len(pathways[j])):
                    tpm[i,j] += fluxmatrix[int(pathways[i][k])][int(pathways[j][p])] * statepathmatrix[int(pathways[j][p])][j] * statepathmatrix[int(pathways[i][k])][i]
        print("No {} pathway inter-flux is completed to be computed;".format(i))
    return tpm


num_states = 900
pathways = np.load("tpt_dijkstra_pathways.npy", allow_pickle=True)
fluxes = np.load("tpt_dijkstra_pathways_fluxes.npy", allow_pickle=True)
source = np.loadtxt("./testdata/iniState.dat")
sink = np.loadtxt("./testdata/finalState.dat")

interflux = interpathflux(pathways=pathways, fluxes=fluxes, num_states=num_states, source=source, sink=sink)

