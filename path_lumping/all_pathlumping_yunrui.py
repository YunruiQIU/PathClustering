import numpy as np
import numba as nb
import scipy.linalg
import gc
from tqdm import tqdm, trange
import warnings
warnings.filterwarnings("ignore")


@nb.jit
def __tcm_2_tpm(tcm):
    tpm = np.zeros(np.shape(tcm))
    tpm = (tcm + tcm.T) / 2
    for i in range(len(tpm)):
        tpm[i] = tpm[i] / np.sum(tpm[i])
    return tpm

@nb.jit
def __committor_calculator(TCM, mergedim, statemergemap):
    mergetcm = np.zeros((mergedim, mergedim))
    commitprob = np.zeros(len(TCM))
    for i in range(len(statemergemap)):
        for j in range(len(statemergemap)):
            mergetcm[int(statemergemap[i]), int(statemergemap[j])] += TCM[i, j]
    mergetpm = __tcm_2_tpm(tcm=mergetcm)
    submergetpm = mergetpm[2:, 2:] - np.identity(mergedim-2)
    rhs = -mergetpm[2:, 1]
    result = np.linalg.solve(submergetpm, rhs)
    for i in range(len(commitprob)):
        if statemergemap[i] == 0:
            commitprob[i] = 0
        if statemergemap[i] == 1:
            commitprob[i] = 1
        if statemergemap[i] > 1:
            commitprob[i] = result[int(statemergemap[i]-2)]
    return commitprob


def __stationary_population(tpm):
    eigval, eigvec, vr = scipy.linalg.eig(tpm, left=True)
    for i in range(len(eigval)):
        if eigval[i] > 0.99999999999:
            break
    return eigvec[:, i] / np.sum(eigvec[:, i])


@nb.jit
def __getprob(matrix, staterow, statecol, selftransition):

    ## This method more or less is like self-consistent sloving, when compute all loop-less probability between any i state and j state, it use the mathematical induction to go from 1 to inifinite(so do the induction twice in the pathlumping, first time to derive the express for loopless probability, second time to compute this prob(i->j) for all loopless pathways); and finally get probability prob = 1 - (Pmi*(1-Pm_prim)^-1*Pjm) - Pmm

    dim = len(matrix)
    infinitematrix = np.linalg.inv((np.identity(dim)-matrix))
    alpha = 0
    for i in range(dim):
        for j in range(dim):
            alpha += staterow[i] * infinitematrix[i, j] * statecol[j]
    alpha += selftransition
    return (1-alpha)



def committor_probability(TCM, source, sink):
    if not type(TCM) is np.ndarray:
        raise TypeError("The input Transition Count Matrix should be numpy.ndarray format.")
    if not len(np.shape(TCM)) == 2:
        raise IOError("The input Transition Count Matrix should be a 2-dimensional numpy.ndarray.")
#    if not isinstance(source, list) or not isinstance(sink, list):
#        raise TypeError("The input source/sink states should be list format.")

    num_states = np.shape(TCM)[0]
    statemergemap = np.ones(num_states)*(-1)
    mergedim = num_states - len(source) - len(sink) + 2

    for i in range(len(source)):
        ## set the state id for initial states is 0;
        statemergemap[int(source[i])] = 0
    for i in range(len(sink)):
        ## set the state id for the final states is 1;
        statemergemap[int(sink[i])] = 1
    temp = 2
    for i in range(len(statemergemap)):
        if statemergemap[i] == -1:
            statemergemap[i] = temp
            temp += 1
    cp = __committor_calculator(TCM=TCM, mergedim=mergedim, statemergemap=statemergemap)
    return cp



def flux_matrix(TCM, commitprob):
    fluxmatrix = np.zeros(np.shape(TCM))
    tpm = __tcm_2_tpm(TCM)
    equipop = __stationary_population(tpm)
    for i in range(len(TCM)):
        for j in range(len(TCM)):
            fluxmatrix[i, j] = equipop[i] * (1-commitprob[i]) * tpm[i, j] * commitprob[j]
    return fluxmatrix


def loopless_flux(fluxmatrix, source, sink):

    num_states = np.shape(fluxmatrix)[0]
    lessdim = len(source) + len(sink) + 1
    ### Normalize the flux matrix to be the probability matrix
    rowsum = np.sum(fluxmatrix, axis=1)
    originfluxmatrix = fluxmatrix.copy()
    for i in range(len(fluxmatrix)):
        if rowsum[i] > 0.0000000000001:
            fluxmatrix[i] = fluxmatrix[i] / rowsum[i]
        else:
            fluxmatrix[i] = np.zeros(len(fluxmatrix[i]))
    prob = np.zeros(num_states)
    for i in tqdm(range(num_states), desc='weigth for states'):
        if float(i) in source or float(i) in sink:
            prob[i] = 1
        else:
            staterow = np.zeros(num_states-lessdim)
            statecol = np.zeros(num_states-lessdim)
            matrix = np.zeros((num_states-lessdim, num_states-lessdim))
            temp = 0
            for j in range(num_states):
                if not float(j) in source and not float(j) in sink and j != i:
                    staterow[temp] = fluxmatrix[i][j]
                    statecol[temp] = fluxmatrix[j][i]
                    temp2 = 0
                    for k in range(num_states):
                        if not float(k) in source and not float(k) in sink and k != i:
                            matrix[temp][temp2] = fluxmatrix[j][k]
                            temp2 += 1
                    temp += 1
            prob[i] = __getprob(matrix=matrix, staterow=staterow, statecol=statecol, selftransition=fluxmatrix[i, i])
        rowsum[i] *= prob[i]
    llmatrix = np.zeros(np.shape(fluxmatrix))
    for i in range(num_states):
        if float(i) in source:
            for j in range(num_states):
                if rowsum[j] < 0.0000000000001:
                    llmatrix[i, j] = 0
                else:
                    llmatrix[i, j] = fluxmatrix[i, j] * rowsum[j]
        else:
            for j in range(num_states):
                if float(j) in sink:
                    llmatrix[i, j] = originfluxmatrix[i, j]
                else:
                    llmatrix[i, j] = fluxmatrix[i, j] * rowsum[i]
    del originfluxmatrix, fluxmatrix, matrix, staterow, statecol, rowsum
    gc.collect()
    return llmatrix


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
    for i in tqdm(range(len(pathways)), desc='pathways-interfluxes'):
        for j in range(len(pathways)):
            for k in range(1, len(pathways[i])-1):
                for p in range(0, len(pathways[j])):
                    tpm[i,j] += fluxmatrix[int(pathways[i][k])][int(pathways[j][p])] * statepathmatrix[int(pathways[j][p])][j] * statepathmatrix[int(pathways[i][k])][i]
        print("No {} pathway inter-flux is completed to be computed;".format(i))
    return tpm


def laplacian_2_data(matrix, num_clusters):

    matrix = (matrix + matrix.T) / 2
    rowsum = np.sum(matrix, axis=1)
    matrix = np.diag(rowsum) - matrix
    rowsum = np.sqrt(rowsum)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            matrix[i, j] = matrix[i, j] / rowsum[i] / rowsum[j]
    eigen, leftvec = scipy.linalg.eig(matrix, left=True, right=False)
    _idx = np.argsort(eigen)
    eigen = eigen[_idx]
    leftvec = leftvec[:, _idx].T
    _normalfactor = np.sum((leftvec*leftvec), axis=1)
    _normalfactor = np.sqrt(_normalfactor)

    ### Use NCut algorithm to divide the networks
    for i in range(len(leftvec)):
        leftvec[i] = leftvec[i] / _normalfactor[i]
    data = leftvec[0:num_clusters].T
    _normalfactor = np.sqrt(np.sum((data*data), axis=1))
    for i in range(len(data)):
        data[i] = data[i] / _normalfactor[i]
    return matrix, eigen, data


def __get_nearest_center(data, centeridlist, centernum):

    centerdist = np.zeros(len(data))
    for i in range(len(data)):
        dist = np.zeros(int(centernum))
        for j in range(int(centernum)):
            dist[j] = np.sum((data[i] - data[int(centeridlist[j])]) * (data[i] - data[int(centeridlist[j])]))
        centerdist[i] = np.min(dist)
    return centerdist

def __get_new_assign(data, centeraverage):

    newassign = np.zeros(len(data))
    for i in range(len(data)):
        dist = np.zeros(len(data[0]))
        for j in range(len(data[0])):
            dist[j] = np.sum((data[i] - centeraverage[j]) * (data[i] - centeraverage[j]))
        newassign[i] = np.argmin(dist)
    return newassign


def __get_new_average(data, assign):

    centeraverage = np.zeros((len(data[0]), len(data[0])))
    count = np.zeros(len(data[0]))
    for i in range(len(data)):
        centeraverage[int(assign[i])] += data[i]
        count[int(assign[i])] += 1
    for i in range(len(centeraverage)):
        centeraverage[i] /= count[i]
    return centeraverage

def __compare_assign(oldassign, newassign):

    samebool = 1
    for i in range(len(oldassign)):
        if int(oldassign[i]) != int(newassign[i]):
            samebool = 0
            break
    return samebool


def spectral_clustering(data, num_clusters, maxiter=1000):

    if len(data[0]) != num_clusters:
        raise ValueError("The input data length should be consistent with the number of clusters")

    mindistsum = 1e10
    ### Scan all the datapoint and find out the best initial seed
    for initseed in tqdm(range(len(data)), desc='bestseeds'):
        centeridlist = np.zeros(num_clusters); centeridlist[0] = initseed;
        for i in range(1, num_clusters):
            centerdist = __get_nearest_center(data=data, centeridlist=centeridlist, centernum=int(i))
            centeridlist[i] = np.argmax(centerdist)
        centerdist = __get_nearest_center(data=data, centeridlist=centeridlist, centernum=num_clusters)

        ### Get initial center coordinates
        centeraverage = np.zeros((num_clusters, num_clusters))
        for i in range(num_clusters):
            centeraverage[i] = data[int(centeridlist[i])]

        ### Do the loop until the clustering results do not change or the maximum iteration number is reached;
        _assign = __get_new_assign(data=data, centeraverage=centeraverage)
        for count in range(0, maxiter):
            centeraverage = __get_new_average(data=data, assign=_assign)
            assign = __get_new_assign(data=data, centeraverage=centeraverage)
            convergebool = __compare_assign(_assign, assign)
            if convergebool == 0:
                _assign = assign
            else:
                break
        distsum = 0
        for i in range(len(data)):
            distsum += np.sum((data[i] - centeraverage[int(assign[i])]) * (data[i] - centeraverage[int(assign[i])]))
#        print(distsum)
        if distsum < mindistsum:
            mindistsum = distsum
            bestseed = initseed
            print("*****The coresponding minimum distance for bestseed is descreased to : ", mindistsum)
#        print("No {} datapoint has been scanned with the initialization seed;".format(initseed))


    initseed = bestseed
    centeridlist = np.zeros(num_clusters); centeridlist[0] = initseed;
    for i in range(1, num_clusters):
        centerdist = __get_nearest_center(data=data, centeridlist=centeridlist, centernum=i)
        centeridlist[i] = np.argmax(centerdist)
    centerdist = __get_nearest_center(data=data, centeridlist=centeridlist, centernum=num_clusters)
    centeraverage = np.zeros((num_clusters, num_clusters))
    for i in range(num_clusters):
        centeraverage[i] = data[int(centeridlist[i])]
    _assign = __get_new_assign(data=data, centeraverage=centeraverage)
    for count in range(0, maxiter):
        centeraverage = __get_new_average(data=data, assign=_assign)
        assign = __get_new_assign(data=data, centeraverage=centeraverage)
        convergebool = __compare_assign(_assign, assign)
        if convergebool == 0:
            _assign = assign
        else:
            break
    return assign






### 00 The input data and parameters for path lumping algorithm
tcm = np.loadtxt("./testdata/countMatrix.dat")  ## Transition Count Matrix for the microstates trajectories
source = np.loadtxt("./testdata/iniState.dat")  ## source for TPT
sink = np.loadtxt("./testdata/finalState.dat")  ## sink for TPT
num_states = 900                                ## Number of microstates for the trajectories
num_pathways = 1000                             ## Number of pathways will be indentified by Transition Path Theory
num_clusters = 4                                ## Number of clusters/channels want to be lumping to
maxiter = 1000                                  ## Maximum iteration used to k-means clustering algorithm


print("********************************************************************************************")
print("#### Pahtlumping algorithm is starting...")
### 00 Show the shape or details about input data
print("The shape of the input TCM data is : ", np.shape(tcm))
print("The shape of the input source states is: ", np.shape(source))
print("The shape of the input sink states is: ", np.shape(sink))
print("Path Lumping algorithm will lump {} microstates model Transition Pathways into {} path channels;".format(num_states, num_clusters))



print("\n\n********************************************************************************************")
### 01 Calculate the committor probabilities for each state and construct corresponding flux matrix
print("#### Committor probabilities and flux matrix is computing...")
committor_prob = committor_probability(tcm, source, sink)
print("The shape of the calculated committor probability is: ", committor_prob.shape)
fluxmat = flux_matrix(TCM=tcm, commitprob=committor_prob)
print("The calculation of flux matrix is complete and shape is: ", np.shape(fluxmat))


print("\n\n********************************************************************************************")
### 02 Calculate the loopless fluxmatrix to exclude the looped path and abandon the network flux matrix
print("#### loopless fluxmatrix is computing...")
llfluxmat = loopless_flux(fluxmatrix=fluxmat, source=source, sink=sink)
print("The calculation of loopless flux matrix ix complete and shape is: ", np.shape(llfluxmat))



print("\n\n********************************************************************************************")
### 03 Find out the pathways and corresponding fluxes using Dijkstra’s algorithm
print("#### Dijkstra algorithm is used to find out pathways and corresponding fluxes...")
pathways, fluxes = get_pathways(flux_matrix=llfluxmat, source=source, sink=sink, num_ways=num_pathways)



print("\n\n********************************************************************************************")
### 04 Merge some pathways if they are indentical (because loopless flux matrix is used and didn't totally move the bottleneck)
print("#### Indentical pathways are merging and combining...")
mergepath, mergeflux = get_merged_pathway(pathways, fluxes)



print("\n\n********************************************************************************************")
### 05 Calculate the interfluxes between different pathways / similarities
print("#### The inter-fluxes / similarities between different pathways are computing...")
interflux = interpathflux(pathways=mergepath, fluxes=mergeflux, num_states=num_states, source=source, sink=sink)



print("\n\n********************************************************************************************")
### 06 Use interfluxes between pathways combined with specral clustering algorithm to classfy different path channels
print("#### Spectral Clustering algorithm is used to classify path channels in accordance with the inter-fluxes/similarities")
matrix, eigenval, data = laplacian_2_data(interflux, num_clusters=num_clusters)
assign = spectral_clustering(data=data, num_clusters=num_clusters, maxiter=maxiter)
