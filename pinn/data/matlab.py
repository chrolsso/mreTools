import scipy, sys
import xarray as xr
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from ..utils import print_if, as_xarray
from .dataset import MREDataset

class MatlabSample(object):
    '''
    Class to load a sample from a Matlab file in BIOQIC format.
    '''

    def __init__(self, filepath, frequency, resolution=[1.0, 1.0, 1.0]):
        """The data in the given .mat file is supposed to have shape (cols, rows, slices, timesteps, components, frequencies).
        Parameters
        ----------
        filepath : str
            Path to the .mat file.
        frequency : int
            Acquisition frequency in Hz.
        resolution : list
            Resolution of the image in mm for lines, cols, slices.
        """
        self.mat_file = filepath
        self.mask_file = None
        self.bin_mask_file = None
        self.freq = frequency
        self.resolution = resolution

    @property
    def wave_var(self):
        return 'u_ft'

    def load_mat(self, verbose=True): 
        """Load the image from the initialized .mat file"""
        data, rev_axes = MatlabSample._load_mat_file(self.mat_file, verbose)
        # move all image values so that the smallest value is 0
        # data['magnitude'] = data['magnitude'] - np.min(data['magnitude'])
        # data['phase'] = data['phase'] - np.min(data['phase'])
        wave = np.multiply(data['magnitude'], np.exp(1j * data['phase'])) # construct complex image from magnitude and phase
        if (len(wave.shape) == 6):
            wave = wave[:, :, :, 0, :, :] # remove time dimension
        wave = self.add_metadata(wave)
        self.arrays = xr.Dataset(dict(wave=wave))

        print_if(verbose, self.arrays)

    def add_metadata(self, array):
        "Adds info on resolution, spatial dimensions, frequencies, and components to the array"
        dims = ['x', 'y', 'z', 'component', 'frequency']
        coords = {
            'frequency': np.array([self.freq]), # Hz
            'x': np.arange(array.shape[0])  * self.resolution[0],
            'y': np.arange(array.shape[1]) * self.resolution[1],
            'z': np.arange(array.shape[2])  * self.resolution[2],
            'component': ['x', 'y', 'z'], # is this really the correct order???
            # 'timesteps': np.arange(array.shape[3]),
        }
        array = xr.DataArray(array, dims=dims, coords=coords)
        return array.transpose('frequency', 'x', 'y', 'z', 'component')

    def preprocess(self, storage_mod_file, loss_mod_file, verbose=True):
        """Add masks and ground truth data"""
        self.storage_mod_file = storage_mod_file
        self.loss_mod_file = loss_mod_file
        self.create_elastogram(verbose)   ## after this         self.arrays['mu'] = mu
        self.segment_regions(verbose) # after this         self.arrays['spatial_region'] = mask

    def segment_regions(self, verbose=True):
        """Add mask data"""
        self.bin_var = None
        self.anat_var = None
        print_if(verbose, 'Segmenting spatial regions')
        u = (self.arrays.wave.mean(['frequency', 'component']))
        if self.mask_file is not None:
            print_if(verbose, 'Adding mask from file')
            perc_mask=self.file_path_mask 
            mask_mat = scipy.io.loadmat(perc_mask) # It should be (80,80,40) directly from MATLAB
            mask_mat=mask_mat['img_seg']  #mask_mat=list(mask_mat)[3]
            print('u shape and mask mat shape:')
            print(u.shape,mask_mat.shape) #(160, 160, 80) (80, 160, 160)
            mask_mat=as_xarray(mask_mat,like=u)  
            mask_mat.name = 'spatial_region'
            self.arrays['spatial_region'] = mask_mat
            print(self.arrays.spatial_region)
            self.arrays = self.arrays.assign_coords(spatial_region=self.arrays.spatial_region)  #additional dimension added
        else:
            mask = np.ones(u.shape, dtype=bool)
            mask = as_xarray(mask, like=u)
            mask.name = 'spatial_region'
            self.arrays['spatial_region'] = mask

        if self.bin_mask_file is not None:
            self.bin_var = 'img_seg_filled'
            print_if(verbose, 'Adding binary mask')
            perc_fill=self.file_path_filled
            mask_fill= scipy.io.loadmat(perc_fill) #DOVREBBE ESSERE  (80,80,40) DIRETTAMENTE DA MATLAB
            mask_fill=mask_fill['img_seg_filled']  #mask_mat=list(mask_mat)[3]
            mask_fill=as_xarray(mask_fill,like=u)  
            mask_fill.name = 'binary_region'
            self.arrays['binary_region'] = mask_fill
            self.arrays = self.arrays.assign_coords(spatial_region=self.arrays.spatial_region)  #additional dimension added 
        else:
            self.arrays['binary_region'] = np.abs(self.arrays['mu']) > 0
            self.arrays = self.arrays.assign_coords(spatial_region=self.arrays.spatial_region)  #additional dimension added
            self.bin_var = '>0'
            
    def create_elastogram(self, verbose=True):
        """Add ground truth data"""
        print_if(verbose, 'Creating ground truth elastogram')
        u=self.arrays.wave.mean(['component'])

        # complex_array = np.multiply(nib.load(self.storage_mod_file).get_fdata(), np.exp(1j * nib.load(self.loss_mod_file).get_fdata()))
        storage = nib.load(self.storage_mod_file).get_fdata()
        loss = nib.load(self.loss_mod_file).get_fdata()
        plt.imshow(storage[:, :, 24])
        plt.show()
        plt.imshow(loss[:, :, 24])
        plt.show()
        complex_array = storage + 1j * loss
        print_if(verbose, 'complex array shape:')
        print_if(verbose, complex_array.shape) #(80, 160, 160, 3, 1)
        mu_mat = complex_array
        mu_mat = np.expand_dims(mu_mat, axis=0)
        print_if(verbose, 'u shape and stiff mat shape:')
        print_if(verbose, u.shape, mu_mat.shape)
        mu = as_xarray(mu_mat, like=u)
        mu.name = 'elastogram'  # CREATES mre (mu)
        self.arrays['mu'] = mu #mu is a NumPy array

    def to_dataset(self):
        print("to dataset") 
        return MREDataset.from_matlab(self)
    
    def _load_mat_file(mat_file, verbose=False):
        '''
        Load data set from MATLAB file.
        Args:
            mat_file: Filename, typically .mat.
            verbose: Print some info about the
                contents of the file.
        Returns:
            Loaded data in a dict-like format.
            Flag indicating MATLAB axes order.
        '''
        mat_file = str(mat_file)
        print_if(verbose, f'Loading {mat_file}')
        try:
            data = scipy.io.loadmat(mat_file)
            rev_axes = True
        except NotImplementedError as e:
            # Please use HDF reader for matlab v7.3 files
            import h5py
            data = h5py.File(mat_file)
            rev_axes = False
        except:
            print(f'Failed to load {mat_file}', file=sys.stderr)
            raise
        if verbose:
            MatlabSample._print_mat_info(data, level=1)
        return data, rev_axes

    def _print_mat_info(data, level=0, tab=' '*4):
        '''
        Recursively print information
        about the contents of a data set
        stored in a dict-like format.
        '''
        for k, v in data.items():
            if hasattr(v, 'shape'):
                print(tab*level + f'{k}: {type(v)} {v.shape} {v.dtype}')
            else:
                print(tab*level + f'{k}: {type(v)}')
            if hasattr(v, 'items'):
                MatlabSample._print_mat_info(v, level+1)