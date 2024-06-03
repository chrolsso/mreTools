import numpy as np
import xarray as xr
import os
import pickle

def load_or_save_fem(exampleObject, path):
    # check if exampleObject has attribute fem
    if hasattr(exampleObject, 'fem'):
        # save fem
        with open(path, 'wb') as file:
            pickle.dump(exampleObject.fem, file, protocol=-1)
    else:
        # load fem
        with open(path, 'rb') as file:
            exampleObject.fem = pickle.load(file)
