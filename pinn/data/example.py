import pathlib
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from ..utils import print_if, as_xarray, delete_folder_contents
from ..extendxarray import fields
from ..visual import XArrayViewer

class MREExample(object):
    '''
    A single instance of preprocessed MRE imaging sequences.
    '''
    def __init__(self, example_id, wave, mre, mre_mask, anat=None,bin_mask=None):
        self.example_id = example_id
        wave = wave.assign_coords(region=mre_mask)
        mre = mre.assign_coords(region=mre_mask)
        self.arrays = {'wave': wave, 'mre': mre, 'mre_mask': mre_mask}
        if anat is not None:
            self.arrays['anat'] = anat.assign_coords(region=mre_mask)
        if bin_mask is not None:
            self.arrays['bin_mask'] = bin_mask.assign_coords(region=mre_mask)

    @classmethod
    def from_matlab(cls, matlab, frequency):  #extracts x arrays from the bioqic object using the frequency and creates and on this creates  MREExample instance with the following attributes, xarray, with a sèecific frequency
        example_id = str(frequency.item())
        arrays = matlab.arrays.sel(frequency=frequency) # three variables, each with 4 dimensions (frequency selected)
        example = MREExample(
            example_id, #frequency
            wave=arrays['wave'],
            mre=arrays['mu'],
            mre_mask=arrays['spatial_region']
        )
        if matlab.bin_var is not None:
            example['bin_mask'] = arrays['binary_region']
        if matlab.anat_var is not None:
            example['anat'] = arrays['anat']
            example['anat_mask'] = arrays['spatial_region']
        return example
    
    @classmethod
    def from_bioqic(cls, bioqic, frequency):
        example_id = str(frequency.item())
        arrays = bioqic.arrays.sel(frequency=frequency)
        example = MREExample(
            example_id,
            wave=arrays['wave'],
            mre=arrays['mu'],
            mre_mask=arrays['spatial_region']
        )
        example['bin_mask'] = arrays['binary_region']
        return example

    @classmethod #the first parametre is the class itself, not one istance. A class method is associated with the class itself, rather than an instance of the class.
    def from_patient(cls, patient):
        example_id = patient.patient_id
        arrays = patient.convert_images()
        sequences = ['t1_pre_in', 't1_pre_out', 't1_pre_water', 't1_pre_fat', 't2']
        new_dim = xr.DataArray(sequences, dims=['component'])
        anat = xr.concat([arrays[a] for a in sequences], dim=new_dim)
        anat = anat.transpose('x', 'y', 'z', 'component')
        example = MREExample(
            example_id,
            wave=arrays['wave'],
            mre=arrays['mre'],
            mre_mask=arrays['mre_mask'],
            anat=anat
        )
        example['anat_mask'] = arrays['anat_mask']
        return example

    @classmethod
    def from_xarrays(cls, xarray_dir, example_id, anat=False, bin_mask=False, verbose=True):
        xarray_dir  = pathlib.Path(xarray_dir) #../data/BIOQIC/fem_box/
        example_dir = xarray_dir / str(example_id) #../data/BIOQIC/fem_box/100
        wave = load_xarray_file(example_dir / 'wave.nc', verbose)   #../data/BIOQIC/fem_box/100/wave.nc
        mre  = load_xarray_file(example_dir / 'mre.nc',  verbose)   #../data/BIOQIC/fem_box/100/mre.nc
        mre_mask  = load_xarray_file(example_dir / 'mre_mask.nc',  verbose) #../data/BIOQIC/fem_box/100/mre_mask.nc
        if anat:
            print('anat')
            anatomy = load_xarray_file(example_dir / 'anat.nc', verbose) #../data/BIOQIC/fem_box/100/anat.nc
            anat_mask = load_xarray_file(example_dir / 'anat_mask.nc', verbose)#../data/BIOQIC/fem_box/100/anat_mask.nc
            if 'sequence' in anatomy.coords:
                anatomy = anatomy.rename(sequence='component')
            if bin_mask:
                binary_mask = load_xarray_file(example_dir / 'bin_mask.nc', verbose) #../data/BIOQIC/fem_box/100/anat.nc
                return MREExample(example_id, wave, mre, mre_mask, anat=anatomy,bin_mask=binary_mask)
            else:
                return MREExample(example_id, wave, mre, mre_mask, anat=anat)
        else:
            if bin_mask:
                print('loading binary mask')
                binary_mask = load_xarray_file(example_dir / 'bin_mask.nc', verbose)
                return MREExample(example_id, wave, mre, mre_mask, bin_mask=binary_mask)
            return MREExample(example_id, wave, mre, mre_mask)

    def save_xarrays(self, xarray_dir, verbose=True):   #Converts the xarray_dir to a pathlib.Path object.
        #Creates a subdirectory named after the example_id within the xarray_dir.
        # Saves the following Xarray files within the subdirectory

        xarray_dir  = pathlib.Path(xarray_dir)
        example_dir = xarray_dir / str(self.example_id)  #data/BIOQUIC/fem_box_matlab
        #if not example_dir.exists():
        example_dir.mkdir(parents=True, exist_ok=True)
        delete_folder_contents(example_dir)
        save_xarray_file(example_dir / 'wave.nc', self.wave, verbose)
        save_xarray_file(example_dir / 'mre.nc',  self.mre,  verbose)  # all images within a frequency have the same mre_mask and same mre
        save_xarray_file(example_dir / 'mre_mask.nc',  self.mre_mask,  verbose)
        print("saving folder related to frequency:")
        print(self.example_id)

        if 'anat' in self:
            save_xarray_file(example_dir / 'anat.nc', self.anat, verbose)
            save_xarray_file(example_dir / 'anat_mask.nc', self.anat_mask, verbose)
        if 'bin_mask' in self:
            save_xarray_file(example_dir / 'bin_mask.nc', self.bin_mask, verbose)

    def __getitem__(self, key):
        return self.arrays[key]

    def __setitem__(self, key, val):
        self.arrays[key] = val

    def __contains__(self, key):
        return key in self.arrays

    def __getattr__(self, key):
        if key in self.arrays:
            return self.arrays[key]
        raise AttributeError(f"'MREExample' object has no attribute '{key}'")

    def vars(self):
        return self.arrays.keys()

    @property
    def metadata(self):
        index_cols = ['variable', 'dimension']
        df = pd.DataFrame(columns=index_cols).set_index(index_cols) #It initializes an empty DataFrame with columns 'variable' and 'dimension', and sets these columns as the index.
        print(self.arrays.keys())
        for var, array in self.arrays.items():  # iterates over the variables in the xarray dataset (self.arrays.items()), where each variable is associated with an array.
            print(var)
            print(array.shape)
            shape = array.field.spatial_shape
            print(shape)
            res = array.field.spatial_resolution
            origin = array.field.origin
            for i, dim in enumerate(array.field.spatial_dims):
                df.loc[(var, dim), 'size'] = shape[i]
                df.loc[(var, dim), 'spacing'] = res[i]
                df.loc[(var, dim), 'origin'] = origin[i]
        df['size'] = df['size'].astype(int)
        df['limit'] = df['origin'] + (df['size'] - 1) * df['spacing']
        df['center'] = df['origin'] + (df['size'] - 1) / 2 * df['spacing']
        df['extent'] = df['limit'] - df['origin'] + df['spacing']
        return df

    def describe(self):
        index_cols = ['variable', 'component']
        df = pd.DataFrame(columns=index_cols).set_index(index_cols)
        for var, array in self.arrays.items():
            if not array.field.has_components:
                comp = 'scalar'
                values = array.values
                df.loc[(var, comp), 'dtype'] = values.dtype
                df.loc[(var, comp), 'count'] = values.size
                df.loc[(var, comp), 'mean'] = values.mean()
                df.loc[(var, comp), 'std'] = values.std()
                df.loc[(var, comp), 'min'] = values.min()
                #df.loc[(var, comp), '25%'] = np.percentile(values_real, 25)
                #df.loc[(var, comp), '50%'] = np.percentile(values, 50)
                #df.loc[(var, comp), '75%'] = np.percentile(values, 75)
                df.loc[(var, comp), 'max'] = values.max()
                continue
            for comp in array.component.values:
                values = array.sel(component=comp).values
                df.loc[(var, comp), 'dtype'] = values.dtype
                df.loc[(var, comp), 'count'] = values.size
                df.loc[(var, comp), 'mean'] = values.mean()
                df.loc[(var, comp), 'std'] = values.std()
                df.loc[(var, comp), 'min'] = values.min()
                #df.loc[(var, comp), '25%'] = np.percentile(values_real, 25)
                #df.loc[(var, comp), '50%'] = np.percentile(values, 50)
                #df.loc[(var, comp), '75%'] = np.percentile(values, 75)
                df.loc[(var, comp), 'max'] = values.max()
        df['count'] - df['count'].astype(int)
        return df

    def downsample(self, **factors):
        arrays = {}
        for var_name in self.vars():
            array = self[var_name].coarsen(boundary='trim', **factors)
            if var_name in {'mre_mask', 'anat_mask', 'spatial_region'}:
                array = array.max()
            else:
                array = array.mean()
            arrays[var_name] = array
        return MREExample(self.example_id, **arrays)

    def add_gaussian_noise(self, noise_ratio, axis=None):
        self.arrays['wave'] = add_gaussian_noise(self.wave, noise_ratio, axis)

    def view(self, *args, mask=0, **kwargs): #visualizing Xarray data, possibly applying a mask if mask is provided.



        for var_name in (args or self.arrays):
            array = self.arrays[var_name]
            if mask > 0:
                if var_name == 'anat':
                    m = self.arrays['mre_mask'] # ERRORE !
                else:
                    m = self.arrays['mre_mask']
                m = ((m > 0) - 1) * mask + 1
                array = as_xarray(array * m, like=array)
            XArrayViewer(array, **kwargs)


def save_xarray_file(nc_file, array, verbose=True):
    print_if(verbose, f'Writing {nc_file}')
    if np.iscomplexobj(array):
        print('object complex')
        new_dim = xr.DataArray(['real', 'imag'], dims=['part'])
        array = xr.concat([array.real, array.imag], dim=new_dim)
        print(new_dim)
        print(array)
    array.to_netcdf(nc_file)


def load_xarray_file(nc_file, verbose=True):
    print_if(verbose, f'Loading {nc_file}')
    array = xr.open_dataarray(nc_file)
    print(array.dims)
    if 'part' in array.dims:
        if 'part' not in array.coords:
            array['part'] = xr.DataArray(['real', 'imag'], dims='part')
        real = array.sel(part='real')
        imag = array.sel(part='imag')
        print(real.shape)
        return real + 1j * imag
    else:
        return array


def complex_normal(loc, scale, size):
    radius = np.random.randn(*size) * scale 
    angle = np.random.rand(*size) * 2 * np.pi
    return radius * np.exp(1j * angle) + loc


def add_gaussian_noise(array, noise_ratio, axis=None):
    array_abs = np.abs(array)
    array_mean = np.mean(array_abs, axis=axis, keepdims=True)
    array_variance = np.var(array_abs, axis=axis, keepdims=True)
    array_power = array_mean**2 + array_variance
    noise_power = noise_ratio * array_power
    noise_std = np.sqrt(noise_power).values
    if np.iscomplexobj(array):
        noise = complex_normal(loc=0, scale=noise_std, size=array.shape)
    else:
        noise = np.random.normal(loc=0, scale=noise_std, size=array.shape)
    return array + noise
