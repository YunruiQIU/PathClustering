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
