import numpy as np
import numba as nb
import scipy.linalg
import gc


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
    for i in range(num_states):
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


tcm = np.loadtxt("./testdata/countMatrix.dat")
print("The shape of the input TCM data is : ", np.shape(tcm))
source = np.loadtxt("./testdata/iniState.dat")
print("The shape of the input source states is: ", np.shape(source))
sink = np.loadtxt("./testdata/finalState.dat")
print("The shape of the input sink states is: ", np.shape(sink))
committor_prob = committor_probability(tcm, source, sink)
print("The shape of the calculated committor probability is: ", committor_prob.shape)
fluxmat = flux_matrix(TCM=tcm, commitprob=committor_prob)
print("The calculation of flux matrix is complete and shape is: ", np.shape(fluxmat))
llfluxmat = loopless_flux(fluxmatrix=fluxmat, source=source, sink=sink)
print("The calculation of loopless flux matrix ix complete and shape is: ", np.shape(llfluxmat))

        

