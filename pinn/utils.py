import sys, tqdm, glob, pathlib
from braceexpand import braceexpand
from functools import wraps
import numpy as np
import xarray as xr
import torch
import sys, inspect, argparse
import os
import shutil


def identity(x):
    """
    Returns the unchanged input.

    Parameters:
    x : object
        The input to be returned.

    Returns:
    object
        The unchanged input.
    """
    return x

def exists(x):
    """
    Checks if the input is not None.

    Parameters:
    x : object
        The input to be checked.

    Returns:
    bool
        True if the input is not None, False otherwise.
    """
    return x is not None


def print_if(verbose, *args, **kwargs):
    """
    Prints the arguments if verbose is True.

    Parameters:
    verbose : bool
        If True, the arguments are printed.
    *args : list
        The arguments to be printed.
    **kwargs : dict
        Keyword arguments for the print function.
    """
    if verbose:
        print(*args, **kwargs)


def progress(*args, **kwargs):
    """
    Returns a tqdm progress bar.

    Parameters:
    *args : list
        Arguments for the tqdm function.
    **kwargs : dict
        Keyword arguments for the tqdm function.

    Returns:
    tqdm.tqdm
        A tqdm progress bar.
    """
    return tqdm.tqdm(*args, **kwargs, file=sys.stdout)


def is_iterable(obj, string_ok=False):
    """
    Checks if the object is iterable.

    Parameters:
    obj : object
        The object to be checked.
    string_ok : bool, optional
        If True, strings are considered iterable. Default is False.

    Returns:
    bool
        True if the object is iterable, False otherwise.
    """
    if isinstance(obj, str):
        return string_ok
    return hasattr(obj, '__iter__')


def as_iterable(obj, length=1, string_ok=False):
    """
    Returns the object as an iterable.

    Parameters:
    obj : object
        The object to be made iterable.
    length : int, optional
        The length of the iterable if the object is not already iterable. Default is 1.
    string_ok : bool, optional
        If True, strings are considered iterable. Default is False.

    Returns:
    iterable
        The object as an iterable.
    """
    if not is_iterable(obj, string_ok):
        return [obj] * length
    return obj


def parse_iterable(obj, sep='-', type=None):
    """
    Parses an iterable object into a list.

    Args:
        obj (iterable or str): The object to be parsed. If it is a string, it will be split using the specified separator.
        sep (str, optional): The separator used to split the string. Defaults to '-'.
        type (type, optional): The type to which each element of the parsed object should be converted. Defaults to None.

    Returns:
        list: The parsed object as a list. If a type is specified, the elements will be converted to that type.

    """
    if isinstance(obj, str):
        obj = obj.split(sep)
    if type is not None:
        return [type(x) for x in obj]
    return obj


def as_matrix(a):
    '''
    Reshape an array or tensor as a matrix.
    '''
    if a.ndim > 2:
        return a.reshape(-1, a.shape[-1])
    elif a.ndim == 2:
        return a
    elif a.ndim == 1:
        return a.reshape(-1, 1)
    else:
        return a.reshape(1, 1)


def as_complex(a, interleave=True, polar=False):
    '''
    Combine the even and odd indices of a real
    array or tensor into the real and imaginary
    parts of a complex array or tensor.

    Args:
        a: (..., 2M) real-valued array/tensor.
    Returns:
        An (..., M) complex-valued array/tensor.
    '''
    if a.dtype.is_complex:
        return a
    if interleave:
        assert a.shape[-1] % 2 == 0, a.shape
        if polar:
            return a[...,0::2] * torch.exp(1j * a[...,1::2])
        else:
            return a[...,0::2] + 1j * a[...,1::2]
    else:
        assert a.shape[-1] == 2, a.shape
        if polar:
            return a[...,0] * torch.exp(1j * a[...,1])
        else:
            return a[...,0] + 1j * a[...,1]


def as_real(a, interleave=True, polar=False):
    '''
    Interleave the real and imaginary parts of a
    complex array or tensor into the even and odd
    indices of a real array or tensor.

    Args:
        a: (N, M) complex-valued array/tensor.
    Returns:
        An (N, 2M) real-valued array/tensor.
    '''
    if not a.dtype.is_complex:
        return a
    if isinstance(a, torch.Tensor):
        if polar:
            a = torch.stack([torch.abs(a), torch.angle(a)], dim=-1)
        else:
            a = torch.stack([a.real, a.imag], dim=-1)
    else: # numpy array
        if polar:
            a = np.stack([np.abs(a), np.angle(a)], axis=-1)
        else:
            a = np.stack([a.real, a.imag], axis=-1)

    if interleave and a.ndim > 1:
        return a.reshape(*a.shape[:-2], -1)
    else:
        return a


def as_xarray(a, like, suffix=None):
    '''
    Convert an array to an xarray, copying the dims and coords
    of a reference xarray.

    Args:
        a: An array to convert to xarray format.
        like: The reference xarray.
    Returns:
        An xarray with the given array values.
    '''
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if suffix is not None:
        name = like.name + suffix
    else:
        name = like.name
    return xr.DataArray(a, dims=like.dims, coords=like.coords, name=name)


def copy_metadata(func, suffix=None):
    """
    Decorator that copies metadata from a reference xarray to the output xarray.

    Parameters:
    func : function
        The function to be decorated.
    suffix : str, optional
        The suffix to be added to the name of the output xarray. Default is None.

    Returns:
    function
        The decorated function.
    """
    @wraps(func)
    def wrapper(a, *args, **kwargs):
        """
        Wrapper function that copies metadata from a reference xarray to the output xarray.

        Parameters:
        a : xarray.DataArray or array-like
            The input data.
        *args : list
            Additional positional arguments for the decorated function.
        **kwargs : dict
            Additional keyword arguments for the decorated function.

        Returns:
        xarray.DataArray or array-like
            The output data with copied metadata.
        """
        ret = func(a, *args, **kwargs)
        if isinstance(a, xr.DataArray):
            return as_xarray(ret, like=a, suffix=suffix)
        return ret
    return wrapper


def concat(args, dim=0):
    """
    Concatenates a list of arrays or tensors along a specified dimension.

    Parameters:
    args : list
        The list of arrays or tensors to be concatenated.
    dim : int, optional
        The dimension along which to concatenate the arrays or tensors. Default is 0.

    Returns:
    torch.Tensor or numpy.ndarray
        The concatenated array or tensor.
    """
    try:
        return torch.cat(args, dim=dim)
    except TypeError:
        return np.concatenate(args, axis=dim)


def minibatch(method):
    """
    Decorator that applies a method to input data in mini-batches.

    Parameters:
    method : function
        The method to be decorated.

    Returns:
    function
        The decorated method.
    """
    @wraps(method)
    def wrapper(self, *args, batch_size=None, **kwargs):
        """
        Wrapper function that applies a method to input data in mini-batches.

        Parameters:
        self : object
            The object instance.
        *args : list
            The input data.
        batch_size : int, optional
            The size of each mini-batch. Default is None.
        **kwargs : dict
            Additional keyword arguments for the method.

        Returns:
        torch.Tensor or numpy.ndarray
            The output data.
        """
        N = args[0].shape[0]
        assert N > 0

        if batch_size is None or batch_size >= N:
            return method(self, *args, **kwargs)

        outputs = []
        for i in range(0, N, batch_size):
            batch_args = [a[i:i + batch_size] for a in args]
            batch_output = method(self, *batch_args, **kwargs)
            outputs.append(batch_output)

        if isinstance(batch_output, tuple):
            return map(concat, zip(*outputs))

        return concat(outputs)

    return wrapper


def as_bool(s):
    """
    Converts a string or boolean value to a boolean.

    Parameters:
    s : str or bool
        The input value.

    Returns:
    bool
        The converted boolean value.
    """
    if isinstance(s, str):
        s = s.lower()
        if s in {'true', 't', '1'}:
            return True
        elif s in {'false', 'f', '0'}:
            return False
        else:
            raise ValueError(f'{repr(s)} is not a valid bool string')
    else:
        return bool(s)


def main(func):
    """
    Decorator that adds command line argument parsing to a main function.

    Parameters:
    func : function
        The main function to be decorated.

    Returns:
    function
        The decorated main function.
    """

    parent_frame = inspect.stack()[1].frame
    __name__ = parent_frame.f_locals.get('__name__')

    if __name__ == '__main__':

        # get full argument specification
        argspec = inspect.getfullargspec(func)
        args = argspec.args or []
        defaults = argspec.defaults or ()
        undefined = object() # sentinel object
        n_undefined = len(args) - len(defaults)
        defaults = (undefined,) * n_undefined + defaults

        # automatically generate argument parser
        parser = argparse.ArgumentParser()
        for name, default in zip(argspec.args, defaults):
            type_ = argspec.annotations.get(name, None)

            if default is undefined: # positional argument
                parser.add_argument(name, type=type_)

            elif default is False and type_ in {bool, None}: # flag
                parser.add_argument(
                    '--' + name, default=False, type=as_bool, help=f'[{default}]'
                )
            else: # optional argument
                if type_ is None and default is not None:
                    type_ = type(default)
                parser.add_argument(
                    '--' + name, default=default, type=type_, help=f'[{default}]'
                )

        # parse and display command line arguments
        kwargs = vars(parser.parse_args(sys.argv[1:]))
        print(kwargs)

        # call the main function
        func(**kwargs)

    return func


def braced_glob(pattern):
    """
    Expands a brace pattern and returns a list of matching file paths.

    Parameters:
    pattern : str
        The brace pattern.

    Returns:
    list
        The list of matching file paths.
    """
    results = []
    for sub_pattern in braceexpand(str(pattern)):
        results.extend(glob.glob(sub_pattern))
    return sorted([pathlib.Path(p) for p in results])


def as_path_list(lst):
    """
    Converts a list of strings to a list of pathlib.Path objects.

    Parameters:
    lst : list
        The list of strings.

    Returns:
    list
        The list of pathlib.Path objects.
    """
    return sorted([pathlib.Path(p) for p in lst])

def delete_folder_contents(folder_path):
    """
    Deletes all files and directories within the specified folder.

    Args:
        folder_path (str): The path to the folder whose contents should be deleted.

    Returns:
        None
    """
    # Iterate over all files and directories in the folder
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        # Check if the item is a file
        if os.path.isfile(item_path):
            # Delete the file
            os.remove(item_path)
        # If it's a directory, delete it recursively
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
