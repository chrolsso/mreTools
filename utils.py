import os
import scipy
import numpy as np
import pydicom as dicom

"""Get a list of all files in a folder

Parameters
----------
folder_path : string
    Path to the folder that should be listed

Returns
-------
list
    List with filenames for all files in folder_path
"""
def get_files_in_folder(folder_path):
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

"""Load folder with single slice dicom files as ndarray

Parameters
----------
folder_path : string
    Path to the folder holding all image slices. The names of all files in the folder should represent the position of the slice.

dimensions : np.ndarray
    list of dimensions in which the slices are ordered. x and y should be included at the start, magnitude / phase should not be included

type : string
    "complex" if the image is complex, "real" if the image is real

Returns
--------
np.ndarray
    complex numpy array containing the pixel data


"""
def load(folder_path, dimensions, type="complex"):
    files = get_files_in_folder(folder_path)
    files = sorted(files)
    img = None
    if type == "complex":
        img = np.empty((dimensions[0], dimensions[1], dimensions[2:].prod()), dtype=np.complex64)
        for i in range(len(files) // 2):
            ds = dicom.dcmread(os.path.join(folder_path, files[i]))
            img.real[:, :, i] = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
            ds = dicom.dcmread(os.path.join(folder_path, files[i+len(files) // 2]))
            img.imag[:, :, i] = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
    else:
        img = np.empty((dimensions[0], dimensions[1], dimensions[2:].prod()), dtype=np.float64)
        for i in range(len(files)):
            ds = dicom.dcmread(os.path.join(folder_path, files[i]))
            img[:, :, i] = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
    return img.reshape(dimensions)

"""Saves the image in the given folder path in a matlab file as img
Takes image in mayo format (slices, components, timesteps) and saves it in bioqic format (slices, timesteps, components, frequencies)

Parameters
----------
file_path : string
    .mat file in which to store the image

pixel_data : np.ndarray
    complex numpy array with magnitude as real part and phase as imaginary

spacing : np.ndarray
    numpy array with voxel sizes in [0] x, [1] y, [2] z directions in meters

frequencies : np.ndarray
    numpy array with frequencies in Hz in the same order as in pixel_data

"""
def to_bioqic(file_path, pixel_data, spacing, frequencies):
    img = dict()
    pixel_data = pixel_data.swapaxes(3, 4)
    pixel_data = np.expand_dims(pixel_data, 5)
    img["magnitude"] = pixel_data.real
    img["phase"] = pixel_data.imag
    img["info"] = dict()
    img["info"]["dx_m"] = spacing[0] / 1000
    img["info"]["dy_m"] = spacing[1] / 1000
    img["info"]["dz_m"] = spacing[2] / 1000
    img["info"]["frequencies_Hz"] = frequencies.tolist()
    dims = list(pixel_data.shape)
    # swap x and y because of matlab format
    x = dims[0]
    dims[0] = dims[1]
    dims[1] = x
    img["info"]["size"] = dims

    scipy.io.savemat(file_path, img, oned_as="column")