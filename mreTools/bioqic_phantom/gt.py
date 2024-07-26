import numpy as np
import os

def getDataPath():
    '''Reads the path to the data directory set at setup
    '''
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    config_file = os.path.join(current_directory, "data_path.txt")
    with open(config_file, "r") as file:
        data_path = file.read()
    return data_path

def getStorageModulusGt():
    '''Returns a map containing the ground truth storage modulus values in Pa. Where no info is available, the map has value 0.
    '''
    return np.load(os.path.join(getDataPath(), "storage_modulus_gt.npy"))[0]

def getSwsGt():
    '''Returns a map containing the ground truth shear wave speed values in m/s. Where no info is available, the map has value 0.
    '''
    mask = getMask()
    
    c = np.zeros_like(mask, dtype=np.float32)
    c[mask == 10] = 3.18 # m/s
    c[mask == 1] = 5.94 # m/s
    c[mask == 2] = 2.21 # m/s
    c[mask == 3] = 2.38 # m/s
    c[mask == 4] = 3.80 # m/s

    return c

def getMask():
    '''Returns a mask of the phantom. The mask has values 1 to 4 for each of the regions of interest and 10 for the surrounding matrix.
    '''
    return np.load(os.path.join(getDataPath(), "mask.npy"))

def getGtSwsForRoi(roi):
    '''Returns the ground truth sws values in m/s for the given region of interest. Possible values are 'roi1', 'roi2', 'roi3', 'roi4' and 'matrix'.
    '''

    assert roi in ['roi1', 'roi2', 'roi3', 'roi4', 'matrix'], "Possible values are 'roi1', 'roi2', 'roi3', 'roi4' and 'matrix'"
    if roi == 'matrix':
        return 3.18
    if roi == 'roi1':
        return 5.94
    if roi == 'roi2':
        return 2.21
    if roi == 'roi3':
        return 2.38
    if roi == 'roi4':
        return 3.80
    
def getGtStorageModulusForRoi(roi):
    '''Returns the ground truth storage modulus values in Pa for the given region of interest. Possible values are 'roi1', 'roi2', 'roi3', 'roi4' and 'matrix'.
    '''

    assert roi in ['roi1', 'roi2', 'roi3', 'roi4', 'matrix'], "Possible values are 'roi1', 'roi2', 'roi3', 'roi4' and 'matrix'"
    if roi == 'matrix':
        return 10100
    if roi == 'roi1':
        return 35300
    if roi == 'roi2':
        return 4900
    if roi == 'roi3':
        return 5600
    if roi == 'roi4':
        return 14400