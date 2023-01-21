"""
Author: Yunrui QIU     email: yunruiqiu@gmail.com
2022-Dec-20th
"""

import numpy as np
import numba as nb
import scipy.linalg


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
    for initseed in range(len(data)):
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
        print("No {} datapoint has been scanned with the initialization seed;".format(initseed))

    
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




interflux = np.loadtxt("interflux.txt")
matrix, eigenval, data = laplacian_2_data(interflux, num_clusters=4)
assign = spectral_clustering(data=data, num_clusters=4, maxiter=1000)
