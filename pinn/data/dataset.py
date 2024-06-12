import os, pathlib, glob
from functools import cache
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from .example import MREExample

class MREDataset(object):
    '''
    A set of preprocessed MRE imaging sequences in xarray format.
    '''
    def __init__(self, example_ids, examples):
        self.example_ids = np.array(example_ids)  # contains frequencies of each xarray
        self.examples = examples  # contains xarray

    @classmethod #    def to_dataset(self): return MREDataset.from_bioqic(self)
    def from_matlab(cls, matlab):
        examples = {} #dict: value = MRE example, key: frequency 
        example_ids = []  #list of frequency
        for frequency in matlab.arrays.frequency:  # for EACH frequency 
            ex = MREExample.from_matlab(matlab, frequency) #creates an MREExample instance with the following attributes  ( xarray)

            example_ids.append(ex.example_id)  # at the end will be ['30' '50' ecc]
            examples[ex.example_id] = ex
        return MREDataset(example_ids, examples)
    
    @classmethod
    def from_bioqic(cls, bioqic):
        examples = {}
        example_ids = []
        for frequency in bioqic.arrays.frequency:
            ex = MREExample.from_bioqic(bioqic, frequency)
            example_ids.append(ex.example_id)
            examples[ex.example_id] = ex
        return MREDataset(example_ids, examples)
    
    def save_xarrays(self, xarray_dir, verbose=True):
        for xid in self.example_ids:  # saves each example at a time, example_ids # contains frequencies of each xarray
            self.examples[xid].save_xarrays(xarray_dir, verbose)