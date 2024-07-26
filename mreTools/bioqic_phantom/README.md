# BIOQIC Phantom Benchmark

This benchmark uses the phantom data available on [the bioqic-apps website](https://bioqic-apps.charite.de/downloads).

The ground truth Storage Modulus values used here have been published in Meyer et al (see below).
The ground truth SWS and Penetration values used here have been published in Papazoglou et al (see below).

## Setup
1. Download the phantom_raw dataset from the bioqic website.
2. Move the file to a unique folder
3. If you haven't done it before, install the mreTools package in your environment: `python -m pip install -e <path to mreTools dir>`
3. Open a command prompt in the bioqic_phantom folder and run `python data_setup.py <path to data folder>`

### Corresponding publications
* [Papazoglou et al: Comparison of inversion methods in MR elastography: An
open-access pipeline for processing multifrequency
shear-wave data and demonstration in a phantom, human
kidneys, and brain](https://iopscience.iop.org/article/10.1088/0031-9155/57/8/2329#pmb418159eqn03)
* [Meyer et al: Multifrequency inversion in magnetic resonance elastography](https://onlinelibrary.wiley.com/doi/epdf/10.1002/mrm.29320)
* [Ingolf Sack: Magnetic resonance elastography 
from fundamental soft-tissue 
mechanics to diagnostic imaging](https://www.nature.com/articles/s42254-022-00543-2.epdf?sharing_token=Y2QaEjQYLuwRku3o2nZMINRgN0jAjWel9jnR3ZoTv0NVxsCMTUhkl-ROXMIy-bvMCRaVSuvNTuEAx18-L6OKQZeOSCttr0K4BAbbQ88FwlwJmP4PqAfxuA5OELYcAb013AXcbYPQMCxs3zm4Oe03QtgSLUVR-zm_ZI7F7GxUINs%3D)
* [Manduca et al: MR elastography: Principles, guidelines, and terminology](https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/mrm.28627)