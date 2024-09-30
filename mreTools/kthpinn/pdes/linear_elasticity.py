import torch
import numpy as np
from mreTools.utils import complex_operator

'''
    Functions for calculating the Navier-Cauchy equation for steady-state elastic wave vibration.
'''

def laplacian(u, x):
    return divergence(jacobian(u, x), x)

def divergence(u, x):
    # jac = torch.stack([jacobian(u[..., i, :], x) for i in range(u.shape[-2])], dim=1)
    # components = jac.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    # return components

    components = torch.zeros((x.shape[-2], u.shape[-2]), dtype=torch.complex64)
    for i in range(u.shape[-2]):
        jac = jacobian(u[...,i,:], x)
        component = 0
        for j in range(jac.shape[-1]):
            component += jac[...,j,j]
        components[:, i] = component
    return components

def jacobian(u, x):
    # components = torch.stack([gradient(u[..., i:i+1], x) for i in range(u.shape[-1])], dim=2)
    # return components

    components = torch.zeros((x.shape[-2], u.shape[-1], x.shape[-1]), dtype=torch.complex64)
    for i in range(u.shape[-1]):
        components[:, i, :] = gradient(u[...,i:i+1], x)
    return components

@complex_operator
def gradient(u, x):
    '''
    In parts taken from Ragoza et al.
    Continuous gradient operator, which maps a
    scalar field to a vector field of partial
    derivatives.

    Args:
        u: (..., 1) output tensor.
        x: (..., K) input tensor.
    Returns:
        D: (..., K) gradient tensor, where:
            D[...,i] = ∂u[...,0] / ∂x[...,i]
    '''

    out = torch.ones_like(u)
    grad = torch.autograd.grad(u, x, create_graph=True, grad_outputs=out)[0]

    return grad

def body_forces(omega, u, rho=1000, detach=False):
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

def get_traction_force_func(equation='helmholtz'):
    '''Returns a function to calculate the traction forces based on the given equation. Valid values are:
        - helmholtz
        - hetero
    '''
    if equation == 'helmholtz':
        def traction_force_func(x, u, mu, detach=False):
            laplace_u = laplacian(u, x)
            if detach:
                laplace_u = laplace_u.detach()
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                laplace_u = laplace_u.to(device)
            return mu * laplace_u.to(torch.complex64)
        return traction_force_func
    
    elif equation == 'hetero':
        pass
    
    else:
        raise ValueError("Invalid equation type")


def pde_residual(x, u, mu, omega, rho=1000, equation='helmholtz'):
    '''Computes the error between mu and the pde based on the given equation and displacement mu at position x.'''
    func_body_force = body_forces
    func_traction_force = get_traction_force_func(equation)

    omega_rad = omega * 2 * np.pi
    f_trac = func_traction_force(x, u, mu)
    f_body = func_body_force(omega_rad, u, rho)
    return f_trac + f_body

def pde_image_residuals(u, mu, omega, rho=1000, equation='helmholtz'):
    '''Computes the error between mu and the pde based on the given equation at all positions in the image. Expects the shape of u and mu to be like (2, z, n, m)'''
    shape = u.shape
    assert mu.shape == shape
    func_body_force = body_forces

    f_body = func_body_force(omega, u, rho)

    return f_body