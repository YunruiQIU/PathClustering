"""
Author: Yunrui QIU     email: yunruiqiu@gmail.com
2022-Dec-20th
"""
import numpy as np
import numba as nb
import scipy.linalg
import gc


def __getmaxcurrent(maxcurrent, maxcurrentnode, reachbool):

    maxvalue = -1
    for i in range(len(maxcurrent)):
        if reachbool[i] == 0:
            continue
        if maxvalue < maxcurrent[i]:
            maxvalue = maxcurrent[i]
            fromnode = i
            tonode = maxcurrentnode[i]
    return fromnode, int(tonode)


def __updatemaxcurrent(matrix, reachbool, maxcurrent, maxcurrentnode):
    
    num_states = len(matrix)
    for i in range(num_states):
        if reachbool[int(maxcurrentnode[i])] == 1:
            maxcurrent[i] = -1
            maxcurrentnode[i] = i
            for j in range(num_states):
                if reachbool[j] != 1 and maxcurrent[i] < matrix[i][j]:
                    maxcurrent[i] = matrix[i, j]
                    maxcurrentnode[i] = j
    return maxcurrent, maxcurrentnode


def __searchpath(flux_matrix, pathvisitbool, tonode, source):
    
    ## Trace one pathway from final sink state to initial source state (choose the paraent states and connect)
    flux = 100000000
    num_states = len(flux_matrix)
    upnode = int(tonode)
    tempnode = int(tonode)
    pathway = [tonode]
    while not float(upnode) in source:
        check = 0
        for j in range(num_states):
            if pathvisitbool[j][tempnode] == 1:
                upnode = j
                check += 1
                if check == 2:
                    raise ValueError("Errors in the path searching (pathvisitbool matrix construction error)")
        if flux_matrix[upnode][tempnode] < flux:
            flux = flux_matrix[upnode][tempnode]
        pathway.append(upnode)
        tempnode = upnode
    pathway = np.array(pathway)
    return pathway[::-1], flux


def get_pathways(flux_matrix, source, sink, num_ways):

    num_states = len(flux_matrix)
    idx_way = 1
    finish = 1
    maxflux = -1
    pathway_ensemble = []
    flux_ensemble = []

    while finish:

    ## Initialization of reachbool vector, which will be used to judge which state can transfer in next;
        reachbool = np.zeros(num_states)
        for i in range(len(source)):
            reachbool[int(source[i])] = 1. ## The first transfer-in state should be source state;
    
    ## Initialization of pathvisitbool, which will be used to trace back the pathway from final to initial
        pathvisitbool = np.zeros((num_states, num_states))
        for i in range(len(source)):
            for j in range(len(source)):
                pathvisitbool[int(source[i])][int(source[j])] = 1.
    
    ## Initialization of maxcurrent and maxcurrentnode for every state, key point for widest path algorithm, Dijkstra's algorithm
        maxcurrent = np.ones(num_states) * -1
        maxcurrentnode = np.zeros(num_states)
        for i in range(num_states):
            for j in range(num_states):
                if reachbool[j] != 1 and maxcurrent[i] < flux_matrix[i][j]:
                    maxcurrent[i] = flux_matrix[i, j]
                    maxcurrentnode[i] = j
        
        ## To go from initial source state to final sink state
        while True:
            
            fromnode, tonode = __getmaxcurrent(maxcurrent, maxcurrentnode, reachbool)
#            print(fromnode, tonode)
            if flux_matrix[int(fromnode)][int(tonode)] < 1e-18:
                finish = 0
                break
            
            reachbool[int(tonode)] = 1
            pathvisitbool[int(fromnode)][int(tonode)] = 1

            if not float(tonode) in sink:
                maxcurrent, maxcurrentnode = __updatemaxcurrent(flux_matrix, reachbool, maxcurrent, maxcurrentnode)
            else:
                pathway, flux = __searchpath(flux_matrix, pathvisitbool, tonode, source)
                if flux > maxflux:
                    maxflux = flux
                    fluxlimit = 0.1 * maxflux
                if flux > fluxlimit:
                    flux = fluxlimit
                for k in range(len(pathway)-1):
                    flux_matrix[int(pathway[k]), int(pathway[k+1])] -= flux
                pathway_ensemble.append(pathway)
                flux_ensemble.append(flux)
                idx_way += 1
                break
        if idx_way % 20 == 0:
            print("No {} Transition Pathway is identified by the TPT and Dijkstra Algorithm;".format(idx_way))
        if idx_way > num_ways:
            finish = 0
    return pathway_ensemble, flux_ensemble
                

 
matrix = np.load("yunrui_loopless_flux_matrix.npy", allow_pickle=True)
source = np.loadtxt("./testdata/iniState.dat")
sink = np.loadtxt("./testdata/finalState.dat")               

print("Initial source states are chosen as: ", source)
print("Final sink states are chosen as ", sink)
pathways, fluxes = get_pathways(flux_matrix=matrix, source=source, sink=sink, num_ways=1000)
print(pathways)
print(fluxes)



