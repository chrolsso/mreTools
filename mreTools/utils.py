import os
import scipy
import numpy as np
import pydicom as dicom
import nibabel as nib
from functools import wraps

def get_files_in_folder(folder_path):
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
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

def load(folder_path, dimensions, type="complex"):
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

def to_nibabel(pixel_data):
    """Wraps the given pixel data in a nibabel image object

    Parameters
    ----------
    pixel_data : np.ndarray
        2D numpy array with pixel data

    Returns
    -------
    nibabel.nifti1.Nifti1Image
        Nifti1Image object containing the pixel data
    """
    return nib.Nifti1Image(pixel_data, np.eye(4))

def to_bioqic(file_path, pixel_data, spacing, frequencies, time=True):
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

    time : bool
        set to false if input image has no time dimension
    """
    img = dict()
    # add empty dimensions for frequencies and time if not given
    if not time:
        pixel_data = np.expand_dims(pixel_data, 3)
    pixel_data = np.expand_dims(pixel_data, 5)
    img["magnitude"] = np.abs(pixel_data)
    img["phase"] = np.angle(pixel_data)
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

def brain_extraction_pre(image_path, image):
    """Not really used at the moment. This can be handy for using a brain extraction tool, but i did not do that yet."""
    # load nifti image from given path
    img = nib.load(os.path.join(image_path, image))
    # get pixel data from image
    img_shape = img.get_fdata().shape
    data = img.get_fdata().reshape((img_shape[0], img_shape[1], 48, 24))
    for i in range(data.shape[3]):
        # create output folder if not exists
        os.makedirs(os.path.join(image_path, "magnitude3d"), exist_ok=True)
        to_nibabel(data[:, :, :, i]).to_filename(os.path.join(os.path.join(image_path, "magnitude3d"), "mag_" + str(i) + ".nii.gz"))

def strcat(s1,s2,s3="",s4="",s5="",s6="",s7="",s8=""):
    """Concatenate up to 6 strings
    """
    l=[s1,s2,s3,s4,s5,s6,s7,s8]
    s=""
    for el in l:
        s=s+el
    return s

def storageLossToSws(complex_shear_modulus, rho):
    '''Calculates a map of shear wave speed values from a map of complex shear modulus values and the material density rho
    Returns sws as c = sqrt(2*|G*| / ρ * (1 + cos(φ(G*)))

    Arguments:
    ----------
    complex_shear_modulus : complex np.ndarray (magnitude = storage modulus, phase = loss modulus)
        complex shear modulus map
    rho : float
        material density in kg/m³
    '''
    return np.sqrt(2*np.abs(complex_shear_modulus) / rho * (1 + np.cos(np.angle(complex_shear_modulus))))

def apparentStiffnessToSws(apparent_stiffness, rho):
    '''Calculates a map of shear wave speed values from a map of apparent stiffness (as) values and the material density rho. The apparent stiffness is the stiffness value returned by LFE inversion.
    Returns sws as c = sqrt(as / ρ)

    Arguments:
    ----------
    apparent_stiffness : np.ndarray
        apparent stiffness map
    rho : float
        material density in kg/m³
    '''
    return np.sqrt(apparent_stiffness / rho)

def shearMagnitudeAngleToSws(G_star, phi, rho):
    '''Calculates a map of shear wave speed values from a map of absolute shear modulus values, a map of shear angle values and the material density rho
    Returns sws as c = sqrt((2*|G*|) / (ρ * (1 + cos(𝜑(G*)))))

    Arguments:
    ----------
    G_star : np.ndarray
        absolute shear modulus (magnitude) map
    phi : np.ndarray
        shear angle (phase) map
    rho : float
        material density in kg/m³
    '''
    return np.sqrt(2*G_star / rho * (1 + np.cos(phi)))

def complex_operator(f):
    """
    Taken from Ragoza et al.
    A decorator that applies a complex operator to a function.

    This decorator checks if the input array `u` is of complex dtype. If it is,
    the function `f` is applied separately to the real and imaginary parts of `u`,
    and the results are combined to form a complex output. If `u` is not complex,
    the function `f` is applied directly to `u`.

    Args:
        f (function): The function to apply the complex operator to.

    Returns:
        function: The decorated function.

    """
    @wraps(f)
    def wrapper(u, x, *args, **kwargs):
        if u.dtype.is_complex:
            f_real = f(u.real, x, *args, **kwargs)
            f_imag = f(u.imag, x, *args, **kwargs)
            return f_real + 1j * f_imag
        else:
            return f(u, x, *args, **kwargs)
    return wrapper