from functools import cache
import numpy as np
import xarray as xr
import torch
import deepxde
import time
import matplotlib.pyplot as plt

from ..utils import minibatch, as_xarray
from ..pde import laplacian
from .losses import msae_loss

class MREPINNData(deepxde.data.Data):

    def __init__(
        self,
        example,
        pde,
        loss_weights,
        pde_warmup_iters=10000,
        pde_init_weight=1e-19,
        pde_step_iters=5000,
        pde_step_factor=10,
        n_points=4096,
        device='cuda'
    ):
        self.example = example
        self.pde = pde

        self.anatomical = ('anat' in example)
        if example.wave.field.has_components:
            self.wave_dims = example.wave.field.n_components
        else:
            self.wave_dims = 1

        self.loss_weights = loss_weights
        self.pde_warmup_iters = pde_warmup_iters
        self.pde_init_weight = pde_init_weight
        self.pde_step_iters = pde_step_iters
        self.pde_step_factor = pde_step_factor
        self.n_points = n_points
        self.device = device
        self.pde_residual=0
        self.pde_loss=0
        self.pde_pesii=[0]
        self.pde_weight=pde_init_weight
    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        x, = inputs
        u_true, mu_true, a_true = (
            targets[...,0:self.wave_dims],
            targets[...,self.wave_dims:self.wave_dims + 1],
            targets[...,self.wave_dims + 1:]
        )
        u_pred, mu_pred, a_pred = outputs

        u_loss  = loss_fn(u_true, u_pred)
        mu_loss = loss_fn(mu_true, mu_pred)
        a_loss  = loss_fn(a_true, a_pred) if self.anatomical else u_loss * 0

        pde_residual = self.pde(x, u_pred, mu_pred)
        pde_loss = loss_fn(0, pde_residual)
        self.pde_residual=pde_residual
        self.pde_loss=pde_loss
        u_weight, mu_weight, a_weight, pde_weight = self.loss_weights
        pde_iter = model.train_state.step - self.pde_warmup_iters
        if pde_iter < 0: # warmup phase (only train wave model)
            pde_weight = 0

        else: # PDE training phase
            n_steps = pde_iter // self.pde_step_iters
            pde_factor = self.pde_step_factor ** n_steps
            pde_weight=self.pde_init_weight * pde_factor
            #pde_weight = min(pde_weight, self.pde_init_weight * pde_factor)
        if self.pde_pesii is not None:
                self.pde_pesii.append(pde_weight)
        self.pde_weight=pde_weight
        self.u_weight=u_weight
        self.a_weight=a_weight
        self.mu_weight=mu_weight
        self.pde_loss=pde_loss
        self.u_loss=u_loss
        self.a_loss=a_loss
        self.mu_loss=mu_loss
        #print('pde loss:',pde_loss,'u loss:',u_loss,'a loss:',a_loss,'mu loss:',mu_loss)
        #print("PDE weight at current iter=",self.pde_weight)
        return [
            u_weight   * u_loss,
            mu_weight  * mu_loss,
            a_weight   * a_loss,
            pde_weight * pde_loss
        ]

    @cache
    def get_raw_tensors(self, device):
        example = self.example

        # get numpy arrays from data example
        x = example.wave.field.points()
        u = example.wave.field.values()
        mu = example.mre.field.values()
        #mu_mask = example.bin_mask.field.values()
        if hasattr(example, 'bin_mask'):
            mu_mask = example.bin_mask.transpose('x','y','z').field.values()
        else:
            mu_mask = example.mre_mask.values.reshape(-1)

        mu_mask = mu_mask.reshape(-1)    #shape: from (160, 160, 80) to (2048000,); unique_mu_mask_np=np.unique(mu_mask) is an array([  0,   4,  11,...207]) len 133,.
        #Before reshape: lots of 0 background+ some numbers (for example slice 6:  #array([  0,  11,  35,  38,  39,  40,  41,  73, 132], dtype=uint8) , len 9))
        
        # convert arrays to tensors on appropriate device
        x = torch.tensor(x, device=device, dtype=torch.float32)
        u = torch.tensor(u, device=device)
        mu = torch.tensor(mu, device=device)
        mu_mask = torch.tensor(mu_mask, device=device, dtype=torch.bool) #torch.Size([2048000])
        # unique_mu_mask_torch=torch.unique(mu_mask) --> tensor([False,  True], device='cuda:0')
        if self.anatomical:
            a = example.anat.field.values()
            a = torch.tensor(a, device=device, dtype=torch.float32)
        else:
            a = u[:,:0]
        return x, u, mu, mu_mask, a

    def get_tensors(self, use_mask=True):
        x, u, mu, mu_mask, a = self.get_raw_tensors(self.device)

        if use_mask: # apply mask and subsample points
            x, u, mu = x[mu_mask], u[mu_mask], mu[mu_mask]
            sample = torch.randperm(x.shape[0])[:self.n_points]
            x, u, mu = x[sample], u[sample], mu[sample]
            a = a[mu_mask][sample]

        input_ = (x,)
        target = torch.cat([u, mu, a], dim=-1)
        aux_var = ()
        return input_, target, aux_var

    def train_next_batch(self, batch_size=None, **kwargs):
        '''
        Returns:
            inputs: Tuple of input tensors.
            targets: Target tensor.
            aux_vars: Tuple of auxiliary tensors.
        '''
        return self.get_tensors(**kwargs)

    def test(self, **kwargs):
        return self.get_tensors(**kwargs)


class MREPINNModel(deepxde.Model):

    def __init__(self, example, net, pde, **kwargs):

        # initialize the training data
        data = MREPINNData(example, pde, **kwargs)

        super().__init__(data, net)

    def benchmark(self, n_iters=100):

        print(f'# iterations: {n_iters}')
        data_time = 0
        model_time = 0
        loss_time = 0
        for i in range(n_iters):
            t_start = time.time()
            inputs, targets, aux_vars = self.data.train_next_batch()
            t_data = time.time()
            x, = inputs
            x.requires_grad = True
            outputs = self.net(inputs)
            t_model = time.time()
            losses = self.data.losses(targets, outputs, msae_loss, inputs, self)
            t_loss = time.time()
            data_time += (t_data - t_start) / n_iters
            model_time += (t_model - t_data) / n_iters
            loss_time += (t_loss - t_model) / n_iters

        iter_time = data_time + model_time + loss_time
        print(f'Data time/iter:  {data_time:.4f}s ({data_time/iter_time*100:.2f}%)')
        print(f'Model time/iter: {model_time:.4f}s ({model_time/iter_time*100:.2f}%)')
        print(f'Loss time/iter:  {loss_time:.4f}s ({loss_time/iter_time*100:.2f}%)')
        print(F'Total time/iter: {iter_time:.4f}s')

        total_time = iter_time * n_iters
        print(f'Total time: {total_time:.4f}s')
        print(f'1k iters time: {iter_time * 1e3 / 60:.2f}m')
        print(f'10k iters time: {iter_time * 1e4 / 60:.2f}m')
        print(f'100k iters time: {iter_time * 1e5 / 3600:.2f}h')

    @minibatch
    def predict(self, x):
        x.requires_grad = True
        u_pred, mu_pred, a_pred = self.net.forward(inputs=(x,))
        lu_pred = laplacian(u_pred, x)
        f_trac, f_body = self.data.pde.traction_and_body_forces(x, u_pred, mu_pred)
        return (
            u_pred.detach().cpu(),
            mu_pred.detach().cpu(),
            a_pred.detach().cpu(),
            lu_pred.detach().cpu(),
            f_trac.detach().cpu(),
            f_body.detach().cpu()
       )

    def test(self):
        
        # get input tensors
        inputs, targets, aux_vars = self.data.test(use_mask=False)

        # get model predictions as tensors
        u_pred, mu_pred, a_pred, lu_pred, f_trac, f_body = \
            self.predict(*inputs, batch_size=self.data.n_points)

        # get ground truth xarrays
        u_true = self.data.example.wave
        mu_true = self.data.example.mre
        if 'anat' in self.data.example:
            a_true = self.data.example.anat
        else:
            a_true = u_true * 0
            a_pred = u_pred * 0
        mu_direct = self.data.example.direct
        mu_fem = self.data.example.fem
        if hasattr(self.data.example, 'bin_mask'):
            mu_mask = self.data.example.bin_mask
        else:
            mu_mask = self.data.example.mre_mask
        Lu_true = self.data.example.Lu
        # apply mask level
        mask_level = 1.0
        #mu_mask = ((mu_mask > 0) - 1) * mask_level + 1   #(160, 160, 80) dtype('float64'), 1 dentro e 0 fuori!
        #mu_mask=xr.ones_like(mu_mask)
        # convert predicted tensors to xarrays
        u_shape, mu_shape, a_shape = u_true.shape, mu_true.shape, a_true.shape
        u_pred  = as_xarray(u_pred.reshape(u_shape), like=u_true)
        lu_pred = as_xarray(lu_pred.reshape(u_shape), like=u_true)
        f_trac  = as_xarray(f_trac.reshape(u_shape), like=u_true)
        f_body  = as_xarray(f_body.reshape(u_shape), like=u_true)
        mu_pred = as_xarray(mu_pred.reshape(mu_shape), like=mu_true)
        a_pred  = as_xarray(a_pred.reshape(a_shape), like=a_true)

        a_vars = ['a_pred', 'a_diff', 'a_true']
        a_dim = xr.DataArray(a_vars, dims=['variable'])
        a = xr.concat([
            mu_mask * a_pred,
            mu_mask * (a_true - a_pred),
            mu_mask * a_true
        ], dim=a_dim)
        a.name = 'anatomy'

        u_vars = ['u_pred', 'u_diff', 'u_true']
        u_dim = xr.DataArray(u_vars, dims=['variable'])
        u = xr.concat([
            mu_mask * u_pred,
            mu_mask * (u_true - u_pred),
            mu_mask * u_true
        ], dim=u_dim)
        u.name = 'wave field'

        lu_vars = ['lu_pred', 'lu_diff', 'Lu_true']
        lu_dim = xr.DataArray(lu_vars, dims=['variable'])
        lu = xr.concat([
            mu_mask * lu_pred,
            mu_mask * (Lu_true - lu_pred),
            mu_mask * Lu_true
        ], dim=lu_dim)
        lu.name = 'Laplacian'

        pde_vars = ['pde_grad', 'pde_diff', 'mu_diff']
        pde_dim = xr.DataArray(pde_vars, dims=['variable'])
        pde_grad = -((f_trac + f_body) * lu_pred * 2)
        if 'component' in pde_grad.sizes:
            pde_grad = pde_grad.sum('component')
        pde_grad *= self.data.loss_weights[2]
        mu_diff = mu_true - mu_pred
        pde = xr.concat([
            mu_mask * pde_grad,
            mu_mask * (mu_diff - pde_grad),
            mu_mask * mu_diff
        ], dim=pde_dim)
        pde.name = 'PDE'

        mu_vars = ['mu_pred', 'mu_diff', 'mu_true']
        mu_dim = xr.DataArray(mu_vars, dims=['variable'])
        mu = xr.concat([
            mu_mask * mu_pred,
            mu_mask * mu_diff,
            mu_mask * mu_true
        ],dim=mu_dim)
        mu.name = 'elastogram'

        direct_vars = ['direct_pred', 'direct_diff', 'mu_true']
        direct_dim = xr.DataArray(direct_vars, dims=['variable'])
        direct = xr.concat([
            mu_mask * mu_direct,
            mu_mask * (mu_true - mu_direct),
            mu_mask * mu_true
        ], dim=direct_dim)
        direct.name = 'direct'

        fem_vars = ['fem_pred', 'fem_diff', 'mu_true']
        fem_dim = xr.DataArray(fem_vars, dims=['variable'])
        fem = xr.concat([
            mu_mask * mu_fem,
            mu_mask * (mu_true - mu_fem),
            mu_mask * mu_true
        ], dim=fem_dim)
        fem.name = 'FEM'

        return 'train', (a, u, lu, pde, mu, direct, fem)
    
    def plot_loss(self, name="sum", mode="train"):
        """Plots the loss history of the requested loss function.
            The plots can be shown with plt.show() or saved with plt.savefig().
            Parameters
            ----------
            name : str
                Name of the loss function to plot. Options are "sum", "u", "mu", "a", "pde".
            mode : str
                Mode of the loss function to plot. Options are "train" and "test".
        """
        # check that name is only one of "sum", "u", "mu", "a", "pde"
        assert name in ["sum", "u", "mu", "a", "pde"], f"Invalid name: {name}"
        assert mode in ["train", "test"], f"Invalid mode: {mode}"

        loss = self.losshistory.loss_train if mode == "train" else self.losshistory.loss_test
        epochs = self.losshistory.steps

        if name == "sum":
            ltrain_array = np.array(loss)
            sum_ltrain = np.sum(ltrain_array, axis=1)
            sum_list_ltrain = sum_ltrain.tolist()
            plt.plot(epochs, sum_list_ltrain, label='Training Loss sum')
        else:
            index = 0
            if name == "u":
                index = 0
            elif name == "mu":
                index = 1
            elif name == "anat":
                index = 2
            elif name == "pde":
                index = 3
            single_loss = [inner_list[index] for inner_list in loss]
            plt.plot(epochs, single_loss, label=f'Training Loss {name}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()