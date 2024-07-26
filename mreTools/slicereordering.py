import sys
import numpy as np
from scipy.spatial.distance import pdist
from .seriate import seriate
from numpy.linalg import norm

"""Orders the given phase images to represent a wave
Given an unordered list of phase images, this function sorts the images to minimize the euclidean distance between all neighboring slices. This means that the final ordering will represent a sampled wave progression. 
The input image has to respect the niquist theorem for correct results.

Parameters
----------
arr : np.ndarray
    A 3D numpy array with dimensions (sliceIndex, y, x) containing unwrapped phase images.
ret_arr_ordered : boolean
    Should the ordered array be returned?
ret_reordering : boolean
    Should the reordering list be returned?

Returns
-------
np.ndarray
    Numpy array of same dimensions as input containing the reordered slices.
"""
def sort_phases(arr, no_phases, ret_arr_ordered=True, ret_reordering=True):
    # init some containers
    arr_ordered = np.empty_like(arr)
    arr_isreverse = np.zeros_like(arr)
    prev_last_slice = np.zeros_like(arr_ordered[:no_phases])
    reordered = np.ndarray(arr.shape[0], dtype=np.int64)
    
    # main loop
    for slice_idx in range(0, arr.shape[0], no_phases):
        print(f'processing slice {slice_idx} - {slice_idx+no_phases-1}')
        
        # get all phases of current slice
        ar = arr[slice_idx:slice_idx+no_phases]

        # sort current phases
        sort_idx = seriate.seriate(pdist(ar.reshape([no_phases, -1]),
            metric='euclidean').astype(int))
        
        d_orig = list()
        d_rev = list()
        
        # # test phase shift and reverse direction
        for shift_ in range(no_phases):

            sort_arr = np.array(sort_idx)
            sort_arr = np.roll(sort_arr, -shift_, axis=0)
            sort_rev = sort_arr[::-1] # reverse indexing

            # detect if wave cycles forward or backward
            ar_orig = ar[sort_arr]
            ar_rev = ar[sort_rev]

            d_orig.append(norm(prev_last_slice - ar_orig))
            d_rev.append(norm(prev_last_slice - ar_rev))

        sort_arr = np.array(sort_idx)

        if min(d_orig) < min(d_rev):
            # original orientation
            shift_ = d_orig.index(min(d_orig))
            sort_arr = np.roll(sort_arr, -shift_, axis=0)

        else: 
            # reverse
            shift_ = d_rev.index(min(d_rev))
            sort_arr = np.roll(sort_arr, -shift_, axis=0)
            sort_arr = sort_arr[::-1] # reverse indexing
            
        arr_ordered[slice_idx:slice_idx+no_phases] = prev_last_slice = ar[sort_arr]
        reordered[slice_idx:slice_idx+no_phases] = sort_arr + slice_idx

    # unstack phase
    arr_ordered = np.moveaxis(arr_ordered.reshape(
        [-1, no_phases, arr_ordered.shape[1], arr_ordered.shape[2]]), 
        1, -1)

    # return desired outputs
    if ret_arr_ordered and ret_reordering:
        return arr_ordered, reordered
    elif ret_arr_ordered:
        return arr_ordered
    elif ret_reordering:
        return reordered
    else:
        return

