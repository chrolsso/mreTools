import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom as dicom
from tqdm import tqdm

from . import unwrapping
from . import slicereordering
from . import utils

"""Sorts phase images into the correct order and writes them to the output_folder

Parameters
----------
partsOfSlice : list
    All filenames that contain phase images for the current slice in numeric order
timesteps : int
    The number of acquired timesteps
data_folder : string
    Path to the image folder
output_folder : string
    Path to the folder where the new image should be written
"""
def sort_phases_for_slice(partsOfSlice, timesteps, data_folder, output_folder):
    # get slice dimensions
    ds = dicom.dcmread(os.path.join(data_folder, str(partsOfSlice[0])))
    size_x = ds.Columns
    size_y = ds.Rows

    # create empty array that will hold the image
    loadedSlice = np.ndarray((len(partsOfSlice), size_y, size_x), dtype=np.float64)
    
    # go through the partsOfSlice list and load the phase images
    dicom_objects = []
    for i in tqdm(range(len(partsOfSlice))):
        dicom_objects.append(dicom.dcmread(os.path.join(data_folder, str(partsOfSlice[i]))))
        loadedSlice[i, :, :] = unwrapping.spatialUnwrapping(dicom_objects[-1].pixel_array * dicom_objects[-1].RescaleSlope + dicom_objects[-1].RescaleIntercept)

    reordering = slicereordering.sort_phases(loadedSlice, timesteps, ret_arr_ordered=False)
    del loadedSlice

    # save old images with new name to reorder them
    for i_old, i_new in enumerate(reordering):
        dicom_objects[i_old].save_as(os.path.join(output_folder, str(partsOfSlice[i_new])))
    
    
"""Order phases of dicom files in given folder
Reads all files in the given folder, and creates copies of the files in another folder, but in the correct order.
This only works if the input image is in the format as it is returned by the scanner, because it makes some assumptions on the order of dimensions.

Parameters
----------
input_path : string
    Path to a folder with dicom files for each slice
output_path : string
    Path to empty folder
timesteps : int
    number of imaged timesteps
megs : int
    number of motion encoding gradients
"""
def order_dicom(input_path, output_path, numSlices, timesteps, megs):
    # get all files for this sample
    files = sorted(utils.get_files_in_folder(input_path))
    
    # write all possible slice locations in sorted order to sorted_locations
    locations = set()
    for i, file in enumerate(files):
        ds = dicom.dcmread(os.path.join(input_path, file))
        locations.add(float(ds.SliceLocation))
    sorted_locations = sorted(locations)
    
    # create a list of file indices that saves for each file its index in the image
    index = np.zeros((numSlices, 2), dtype=int)
    fileIndices = []
    for i, file in enumerate(files):
        ds = dicom.dcmread(os.path.join(input_path, file))
        if i < len(files)//2:
            phase = 0 # magnitude
        else:
            phase = 1 # phase
        currentSliceIndex = sorted_locations.index(ds.SliceLocation) % (len(files)//2)
        currentIndex = index[currentSliceIndex, phase]
        fileIndices.append([currentSliceIndex, currentIndex, phase])
        index[currentSliceIndex, phase] += 1
    
    # this goes through the fileIndices list to find all files corresponding to the given slice
    def find(lst, slc, ph):
        return [(i+1) for i, x in enumerate(lst) if x[0] == slc and x[2] == ph]

    # go through all slices and sort them
    for i in range(len(locations)):
        partsOfSlice = find(fileIndices, i, 1)
        sort_phases_for_slice(partsOfSlice, timesteps, input_path, output_path)