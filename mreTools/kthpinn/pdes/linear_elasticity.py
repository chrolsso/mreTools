import torch
import numpy as np

'''
    Functions for calculating the Navier-Cauchy equation for steady-state elastic wave vibration.
'''

def body_forces(self, omega, u, rho=1000, detach=False):
    '''computes the formula: ρ*ω²*u, where 
        ρ is the material density in kg/m³, 
        ω is the angular frequency in rad/s, 
        u is the displacement field

        Arguments:
        ----------
        omega : float
            constant angular frequency
        u : torch.Tensor | np.ndarray
            displacement field
        rho : float
            constant material density
        detach : bool
            whether to detach the tensor from the computational graph. This is only useful when working with torch tensors. 
            Setting detach to true will prevent the tensor from being tracked by autograd, which is useful when you want to exclude it from gradient computation.
    '''
    assert isinstance(u, torch.Tensor) or isinstance(u, np.ndarray)
    if isinstance(u, torch.Tensor) and detach:
        u = u.detach()
    return rho * omega**2 * u

