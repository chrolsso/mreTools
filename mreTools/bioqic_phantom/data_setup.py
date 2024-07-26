import sys
import matplotlib.pyplot as plt
import h5py
import scipy
import numpy as np
import os 

from mreTools.mrepinn.data import bioqic

def data_setup():
    '''
    Setup function that needs to be run once to create the data necessary for the phantom benchmark
    '''
    data_path = ""
    

    # get the path to data folder as a command line argument
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        print("Missing argument data_path")

    # store the data_path in a file in the current working directory
    with open("data_path.txt", "w") as file:
        file.write(data_path)

    # create a single mask with different values for the different rois
    rois = scipy.io.loadmat("./phantom_ROIs.mat")
    matrix = np.transpose(rois['ROImatrix'], (1, 0)) * 10
    roi1 = np.transpose(rois['ROI1'], (1, 0)) * 1
    roi2 = np.transpose(rois['ROI2'], (1, 0)) * 2
    roi3 = np.transpose(rois['ROI3'], (1, 0)) * 3
    roi4 = np.transpose(rois['ROI4'], (1, 0)) * 4

    mask = np.ndarray((128, 80), dtype=np.int8)
    mask = np.maximum.reduce([matrix, roi1, roi2, roi3, roi4])

    # expand the mask for the 25 slices of the image
    mask = np.expand_dims(mask, axis=-1)
    mask = np.repeat(mask, 25, axis=-1)

    # save the mask for later use
    np.save(os.path.join(data_path, "mask.npy"), mask)

    # Assign values from Meyer et al for each target region
    stiffness = {
        'matrix': 10100,
        'roi1': 35300,
        'roi2': 4900,
        'roi3': 5600,
        'roi4': 14400
    }
    mu = np.copy(mask)
    mu = mu.astype(np.uint16)
    mu[mask == 10] = stiffness['matrix']
    mu[mask == 1] = stiffness['roi1']
    mu[mask == 2] = stiffness['roi2']
    mu[mask == 3] = stiffness['roi3']
    mu[mask == 4] = stiffness['roi4']
    mu = np.expand_dims(mu, axis=0)
    mu = np.repeat(mu, 8, axis=0)
    np.save(os.path.join(data_path, "storage_modulus_gt.npy"), mu)

if __name__ == "__main__":
    data_setup()