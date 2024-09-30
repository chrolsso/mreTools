import numpy as np
import torch


def normalized_l2_loss_fn(y):
    '''This function is directly taken from MREPINN by Ragoza et al.'''
    norm = np.linalg.norm(y, axis=1).mean()
    def loss_fn(y_true, y_pred):
        return torch.mean(
            torch.norm(y_true - y_pred, dim=1) / norm
        )
    return loss_fn


def standardized_msae_loss_fn(y):
    '''This function is directly taken from MREPINN by Ragoza et al.'''
    variance = torch.var(torch.as_tensor(y))
    def loss_fn(y_true, y_pred):
        return torch.mean(
            torch.abs(y_true - y_pred)**2 / variance
        )
    return loss_fn


def msae_loss(y_true, y_pred):
    '''This function is directly taken from MREPINN by Ragoza et al.'''
    return torch.mean(
        torch.abs((y_true - y_pred))**2
    )

def complex_mse_loss(y_true, y_pred, batch_dim=None):
    '''Combines the first dimension of the input image representing real and complex parts into a complex image before calculating the loss.
    Expects input of shape (batch_size, 2, height, width, ...)
    If batch dim is specified, the losses for each sample are computed separately and the result is returned as a list.'''
    complex_y_true = y_true[...,0] + 1j * y_true[...,1]
    complex_y_pred = y_pred[...,0] + 1j * y_pred[...,1]
    abs = torch.abs(complex_y_true - complex_y_pred)**2

    if batch_dim is None: 
        return torch.mean(abs)
    
    shape = abs.shape
    num_of_dims = len(shape)

    if num_of_dims == 1:
        return abs
    
    dims_to_reduce = list(range(num_of_dims))
    dims_to_reduce.remove(batch_dim)
    return torch.mean(abs, dim=dims_to_reduce)

def normalized_complex_mse_loss(y_true, y_pred, batch_dim=None):
    '''Computes the mse loss seperately for the first and second item of the first component. Then normalizes both results with the value range of the true image and returns the sum of both.
    Expects input of shape (batch_size, 2, height, width, ...)'''
    real_loss_no_mean = (y_true[...,0] - y_pred[...,0])**2
    imag_loss_no_mean = (y_true[...,1] - y_pred[...,1])**2
    real_loss = 0
    imag_loss = 0

    shape = y_true[..., 0].shape
    num_of_dims = len(shape)
    
    if batch_dim is None:
        real_loss = torch.mean(real_loss_no_mean)
        imag_loss = torch.mean(imag_loss_no_mean)
    else:
        if num_of_dims == 1:    # there is a batch dim given but only one dimension exists. In this case, we do not take the mean
            real_loss = real_loss_no_mean
            imag_loss = imag_loss_no_mean
        else:
            dims_to_reduce = list(range(num_of_dims))
            dims_to_reduce.remove(batch_dim)

            real_loss = torch.mean(real_loss_no_mean, dim=dims_to_reduce)
            imag_loss = torch.mean(imag_loss_no_mean, dim=dims_to_reduce)
    
    # return real_loss / torch.max(y_true[...,0]) + imag_loss / torch.max(y_true[...,1])
    return real_loss + imag_loss

def normalized_complex_mae_loss(y_true, y_pred):
    '''Computes the mae loss seperately for the first and second item of the first component. Then normalizes both results with the value range of the true image and returns the sum of both.
    Expects input of shape (batch_size, 2, height, width, ...)'''
    mse = torch.nn.L1Loss()
    real_loss = mse(y_true[:,0], y_pred[:,0])
    imag_loss = mse(y_true[:,1], y_pred[:,1])
    print(real_loss, imag_loss)
    return real_loss / torch.max(y_true[:,0]) + imag_loss / torch.max(y_true[:,1])