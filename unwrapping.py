import numpy as np
import scipy

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

"""Perform laplacian unwrapping on the wrapped input image
Reference: Hirsch et al., Magnetic Resonance Elastography
Laplacian unwrapping is based on the idea that the gradients are similar in wrapped and unwrapped images. Therefore we can inverse the laplace operator to find a possible original image without wrapping.

Parameters
----------
image : np.ndarray
    2D wrapped image

Returns
-------
np.ndarray
    2D unwrapped image
"""
def spatialUnwrapping(image, preprocessing = smoothImage):
    image = preprocessing(image)
    L = scipy.sparse.csc_matrix(constructDiscreteLaplacian(image.shape[1], image.shape[0]))
    x = scipy.sparse.linalg.spsolve(L, image.flatten())
    return x.reshape(image.shape)