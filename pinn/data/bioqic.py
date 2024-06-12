
import sys, pathlib, urllib
import numpy as np
import xarray as xr
import skimage.draw
import scipy.io

from ..utils import print_if, as_xarray
from .dataset import MREDataset
from ..visual import XArrayViewer


class BIOQICSample(object):
    '''
    An MRE sample dataset from https://bioqic-apps.charite.de/.
    '''
    @property
    def mat_name(self):
        raise NotImplementedError
    
    @property
    def mat_base(self):
        return self.mat_name + '.mat'

    @property
    def mat_file(self):
        return self.download_dir / self.mat_base

    def download(self, verbose=True):
        url = f'https://bioqic-apps.charite.de/DownloadSamples?file={self.mat_base}'
        print_if(verbose, f'Downloading {url}')
        self.download_dir.mkdir(exist_ok=True)
        urllib.request.urlretrieve(url, self.mat_file)

    def load_mat(self, verbose=True):
        data, rev_axes = load_mat_file(self.mat_file, verbose)

        wave = data[self.wave_var].T if rev_axes else data[self.wave_var]
        wave = self.add_metadata(wave)
        self.arrays = xr.Dataset(dict(wave=wave))

        if self.anat_var is not None:
            anat = data[self.anat_var].T if rev_axes else data[self.anat_var]
            anat = self.add_metadata(anat)
            self.arrays['anat'] = anat

        print_if(verbose, self.arrays)

    def preprocess(self, verbose=True):
        self.segment_regions(verbose)
        self.create_elastogram(verbose)
        self.preprocess_wave_image(verbose)

    def select_data_subset(self, frequency, xyz_slice, verbose=True):
        self.arrays, ndim = select_data_subset(
            self.arrays, frequency, xyz_slice, verbose=verbose
        )

    def spatial_downsample(self, factor, verbose=True):
        print_if(verbose, 'Spatial downsampling')
        factors = {d: factor for d in self.arrays.field.spatial_dims}
        arrays = self.arrays.coarsen(boundary='trim', **factors).mean()
        arrays['spatial_region'] = self.arrays.spatial_region.coarsen(
            boundary='trim', **factors
        ).max()
        self.arrays = arrays

    def view(self, *args, **kwargs):
        if not args:
            args = self.arrays.keys()
        for arg in args:
            viewer = XArrayViewer(self.arrays[arg], **kwargs)

    def to_dataset(self):
        return MREDataset.from_bioqic(self)
    
class BIOQICFEMBox(BIOQICSample):

    def __init__(self, download_dir):
        self.download_dir = pathlib.Path(download_dir)

    @property
    def mat_name(self):
        return 'four_target_phantom'

    @property
    def anat_var(self):
        return None

    @property
    def wave_var(self):
        return 'u_ft'

    def add_metadata(self, array):
        resolution = 1e-3 # meters
        dims = ['frequency', 'component', 'z', 'x', 'y']
        coords = {
            'frequency': np.arange(50, 101, 10), # Hz
            'x': np.arange(80)  * resolution,
            'y': np.arange(100) * resolution,
            'z': np.arange(10)  * resolution,
            'component': ['y', 'x', 'z'],
        }
        array = xr.DataArray(array, dims=dims, coords=coords)
        return array.transpose('frequency', 'x', 'y', 'z', 'component')

    def segment_regions(self, verbose=True):
        
        print_if(verbose, 'Segmenting spatial regions')
        u = self.arrays.wave.mean(['frequency', 'component'])

        matrix_mask = np.ones((80, 100), dtype=int)
        disk_mask = np.zeros((80, 100), dtype=int)
        disks = [
            skimage.draw.disk(center=(39.8, 73.6), radius=10),
            skimage.draw.disk(center=(39.8, 49.6), radius=5),
            skimage.draw.disk(center=(39.8, 31.8), radius=3),
            skimage.draw.disk(center=(39.8, 18.6), radius=2),
        ]
        disk_mask[disks[0]] = 1
        disk_mask[disks[1]] = 2
        disk_mask[disks[2]] = 3
        disk_mask[disks[3]] = 4

        mask = (matrix_mask + disk_mask)[:,:,np.newaxis]
        mask = as_xarray(np.broadcast_to(mask, u.shape), like=u)
        mask.name = 'spatial_region'
        self.arrays['spatial_region'] = mask
        self.arrays = self.arrays.assign_coords(
            spatial_region=self.arrays.spatial_region
        )

    def create_elastogram(self, verbose=True):

        print_if(verbose, 'Creating ground truth elastogram')
        spatial_region = self.arrays.spatial_region
        wave = self.arrays.wave

        # ground truth physical parameters
        mu = np.array(
            [0, 3e3, 10e3, 10e3, 10e3, 10e3]
        )[spatial_region][np.newaxis,...,]

        axes = tuple(range(1, mu.ndim))
        omega = 2 * np.pi * wave.frequency
        omega = np.expand_dims(omega, axis=axes)

        # Voigt model
        eta = 1 # Pa·s
        mu = mu + 1j * omega * eta
        mu = as_xarray(mu, like=wave.mean(['component']))
        mu.name = 'elastogram'
        self.arrays['mu'] = mu

    def preprocess_wave_image(self, verbose=True):
        pass

def load_mat_file(mat_file, verbose=False):
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
        print_mat_info(data, level=1)
    return data, rev_axes

def print_mat_info(data, level=0, tab=' '*4):
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
            print_mat_info(v, level+1)

def select_data_subset(
    data,
    frequency=None,
    xyz_slice=None,
    downsample=None,
    verbose=True
):
    '''
    Args:
        data: An xarray dataset with the dimensions:
            (frequency, x, y, z, component)
        frequency: Single frequency to select.
        x_slice, y_slice, z_slice: Indices of spatial dimensions to subset,
            resulting in 2D or 1D.
        downsample: Spatial downsampling factor.
    Returns:
        data: An xarray containing the data subset.
        ndim: Whether the subset is 1D, 2D, or 3D.
    '''
    # spatial downsampling
    if downsample and downsample > 1:
        downsample = {'x': downsample, 'y': downsample, 'z': downsample}
        data = data.coarsen(**downsample).mean()
        data['spatial_region'] = data.spatial_region.coarsen(**downsample).max()

    # single frequency
    if frequency and frequency not in {'all', 'multi'}:
        print_if(verbose, 'Single frequency', end=' ')
        data = data.sel(frequency=[frequency])
    else:
        print_if(verbose, 'Multi frequency', end=' ')

    x_slice, y_slice, z_slice = parse_xyz_slice(xyz_slice)

    # single x slice
    if x_slice is not None:
        data = data.isel(x=x_slice)

    # single y slice
    if y_slice is not None:
        data = data.isel(y=y_slice)

    # single z slice
    if z_slice is not None:
        data = data.isel(z=z_slice)

    # number of spatial dimensions
    ndim = (x_slice is None) + (y_slice is None) + (z_slice is None)
    assert ndim > 0, 'no spatial dimensions'
    print_if(verbose, f'{ndim}D')

    # subset the displacement components
    data = data.sel(component=['z', 'y', 'x'][:ndim])

    return data, ndim

def parse_xyz_slice(xyz_slice):
    if not xyz_slice:
        return (None, None, None)
    if isinstance(xyz_slice, str):
        xyz_slice = xyz_slice.upper()
        if xyz_slice == '3D':
            return (None, None, None)
        elif xyz_slice == '2D':
            return (None, None, 0)
        elif xyz_slice == '1D':
            return (None, 75, 0)
        else:
            return map(int, xyz_slice.split('-'))
    return xyz_slice