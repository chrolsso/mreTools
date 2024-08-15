import xarray as xr
import torch
import numpy as np

from torch.utils.data import Dataset

class KthBrainMreDatasetSingle(Dataset):
    """Dataset containing the pixel data of a single patient from the KTH Brain MRE Dataset.
    
    Arguments
    ----------
    data_file : str
        Path to the xarray file (.nc) containing the data. The file should contain two 'parts' which are 'real' and 'imag'
    verbose : bool
        If True, print debug information
    """

    def __init__(self, data_file, verbose=False):
        self.verbose = verbose
        self.image = np.array(self.load_xarray_file(data_file))

    def load_xarray_file(self, nc_file):
        """Load an xarray file and return the data as a numpy array. If the file contains a 'part' dimension, the real and imaginary parts are combined into a complex array."""
        if self.verbose: print(f'Loading {nc_file}')
        array = xr.open_dataarray(nc_file)
        if self.verbose: print(array.dims)
        if 'part' in array.dims:
            if 'part' not in array.coords:
                array['part'] = xr.DataArray(['real', 'imag'], dims='part')
            real = array.sel(part='real')
            imag = array.sel(part='imag')
            if self.verbose: print(real.shape)
            return real + 1j * imag
        else:
            return array

    def image_info(self):
        print(f'Image type: {self.image.dtype}')
        print(f'Image shape: {self.image.shape}')

class KthBrainMreDatasetPixel(KthBrainMreDatasetSingle):
    """Dataset for KTH Brain MRE data that returns single pixel data from the same image
    """
    
    def __len__(self):
        return np.prod(self.image.shape)

    def __getitem__(self, idx):
        idx = np.unravel_index(idx, self.image.shape)
        real = torch.tensor(self.image[idx].real, dtype=torch.float32)
        imag = torch.tensor(self.image[idx].imag, dtype=torch.float32)
        return torch.stack((real, imag), dim=0)

class KthBrainMreDatasetImage(KthBrainMreDatasetSingle):
    """Dataset for KTH Brain MRE data that returns the same image every time

    Arguments
    ----------
    data_file : str
        Path to the xarray file (.nc) containing the data. The file should contain two 'parts' which are 'real' and 'imag'
    outputMagnitude : bool
        If True, __getitem__ will return the complex image as magnitude and phase, otherwise as real and imaginary parts
    """

    def __init__(self, data_file, outputMagnitude = False, verbose=False):
        super().__init__(data_file, verbose)
        self.outputMagnitude = outputMagnitude
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        if self.outputMagnitude:
            magn = torch.from_numpy(np.abs(self.image))
            phase = torch.from_numpy(np.angle(self.image))
            return torch.stack((magn, phase), dim=0).float()
        else:
            real = torch.from_numpy(self.image.real)
            imag = torch.from_numpy(self.image.imag)
            return torch.stack((real, imag), dim=0).float()