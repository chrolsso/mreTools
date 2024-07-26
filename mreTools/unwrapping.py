import numpy as np
import scipy
import nibabel as nib
from fsl import wrappers
import matplotlib.pyplot as plt
import os, sys

from mreTools import utils

"""Construct discrete Laplacian operator
Constructs a discrete Laplacian operator to calculate a gradient image using matrix multiplication. The resulting matrix can be multiplied to a flattened image to get the same result as you would get using a
2D convolution with the 2D laplacian operator [[0, 1, 0], [1, -4, 1], [0, 1, 0]].

Parameters
----------
factor_size_x : int
    The width of the image that the operator should be applied on
factor_size_y : int
    The height of the image that the operator should be applied on

Returns
-------
np.ndarray
    Discrete laplacian operator as array of size (factor_size_x*factor_size_y, factor_size_x*factor_size_y)

Example
-------
>>> img = np.random.randint(0, 256, (3, 5))
... grad_img = np.matmul(constructDiscreteLaplacian(5, 3), img.flatten())
"""
def constructDiscreteLaplacian(factor_size_x, factor_size_y):
    size = factor_size_x * factor_size_y
    operator = np.zeros((size, size), dtype=np.int64)
    # corners:
    # upper left
    operator[0, :2] = [-4, 1]
    operator[0, factor_size_x] = 1
    # upper right
    operator[factor_size_x-1, factor_size_x-2:factor_size_x] = [1, -4]
    operator[factor_size_x-1, 2*factor_size_x-1] = 1
    # lower left
    operator[-factor_size_x, -factor_size_x:-factor_size_x+2] = [-4, 1]
    operator[-factor_size_x, -(2*factor_size_x)] = 1
    # lower right
    operator[-1, -2:] = [1, -4]
    operator[-1, -factor_size_x-1] = 1

    # edges
    for i in range(1, factor_size_x-1):
        # top
        operator[i, (i-1):(i-1)+3] = [1, -4, 1]
        operator[i, factor_size_x+i] = 1
        # bottom
        operator[-i-1, factor_size_x*factor_size_y-i+1-3:factor_size_x*factor_size_y-i+1] = [1, -4, 1]
        operator[-i-1, -factor_size_x-1-i] = 1
    for i in range(1, factor_size_y-1):
        # left
        operator[i*factor_size_x, (i-1)*factor_size_x] = 1
        operator[i*factor_size_x, i*factor_size_x:i*factor_size_x+2] = [-4, 1]
        operator[i*factor_size_x, (i+1)*factor_size_x] = 1
        # right
        operator[(i+1)*factor_size_x-1, i*factor_size_x-1] = 1
        operator[(i+1)*factor_size_x-1, (i+1)*factor_size_x-2:(i+1)*factor_size_x] = [1, -4]
        operator[(i+1)*factor_size_x-1, (i+2)*factor_size_x-1] = 1
        
    # remaining
    for i in range(1, factor_size_y-1):
        for j in range(1, factor_size_x-1):
            operator[i*factor_size_x+j, (i-1)*factor_size_x+j] = 1
            operator[i*factor_size_x+j, i*factor_size_x+(j-1):i*factor_size_x+(j-1)+3] = [1, -4, 1]
            operator[i*factor_size_x+j, (i+1)*factor_size_x+j] = 1
    
    return operator

"""Apply smoothing to the given image

Parameters
----------
image : np.ndarray
    2D image data

Returns
-------
np.ndarray
    2D blurred image

"""
def smoothImage(image):
    sigma = 2
    radius = 2
    return scipy.ndimage.gaussian_filter(image, sigma, radius=radius)

def spatialUnwrapping(image, method = "fsl", preprocessing = lambda x: x):
    """Perform unwrapping on the wrapped input image
    Supported methods:
    - FSL PRELUDE (default)
    Needs complex image as input

    - Laplacian (currently faulty implementation)
    Reference: Hirsch et al., Magnetic Resonance Elastography
    Only phase image is necessary, both complex images or real images holding the phase information are supported
    Laplacian unwrapping is based on the idea that the gradients are similar in wrapped and unwrapped images. Therefore we can inverse the laplace operator to find a possible original image without wrapping.

    Parameters
    ----------
    image : np.ndarray
        2D wrapped image, either real or complex
    method : string
        The method to use for unwrapping. Valid inputs are "fsl" and "laplacian" (laplacian has some errors though), default is "fsl"
    preprocessing : function
        A function that is applied to the image before unwrapping (e.g. smoothing)

    Returns
    -------
    np.ndarray
        2D unwrapped image
    """
    image = preprocessing(image)
    if method == "fsl":
        if not np.iscomplexobj(image):
            print("FSL unwrapping requires complex image as input, returning original image")
            return image
        # scale phase to 2pi range
        mag = np.abs(image)
        phase = np.angle(image)
        # print("imag", phase.min(), phase.max())
        image.imag = ((phase - phase.min()) / (phase.max() - phase.min())) * 2*np.pi
        # print("imag", phase.min(), phase.max())
        # print("real", mag.min(), mag.max())
        magn = utils.to_nibabel(mag)
        phase = utils.to_nibabel(phase)
        # fslpy is unfortunately just a wrapper to the FSL command line tools, so we need to save the image to disk and load it back
        wrappers.prelude(phase = phase, abs = magn, out = 'fsl_temp')
        unwrapped = nib.load('fsl_temp.nii.gz').get_fdata()
        os.remove('fsl_temp.nii.gz')
        return unwrapped

    elif method == "laplacian":
        if np.iscomplexobj(image):
            image = image.imag
        L = scipy.sparse.csc_matrix(constructDiscreteLaplacian(image.shape[1], image.shape[0]))
        x = scipy.sparse.linalg.spsolve(L, image.flatten())
        return x.reshape(image.shape)
    
    else:
        print("Method not recognized, returning wrapped image")
        return image


def frequencySelection(image, timeDimension):
    plt.imshow(np.abs(image[4, 1, 4, 12, :, :]))
    plt.show()
    image = np.moveaxis(image, timeDimension, -1)
    dims = np.array(image.shape)
    print("before unwrapping", dims)
    image = image.reshape((dims[:-1].prod(), dims[-1]))
    for i in range(image.shape[0]):
        image[i, :] = scipy.fft.fft(image[i, :])
    print("after unwrapping", dims, dims[:-1])
    return image[:, 1].reshape(dims[:-1])
