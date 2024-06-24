import numpy as np
import torch

from .utils import as_complex
from .generic import get_activ_fn

# Convert non-numeric elements to None
class MREPINN(torch.nn.Module):  # #torch.nn.Module is base class for all neural network modules-->Your models should also subclass this class.
    
    def __init__(self, example, omega, activ_fn='ss', siren_init=True, u_dropout=False, mu_dropout=False, **kwargs):
        super().__init__()  
        metadata = example.metadata
        metadata['center'] =  metadata['center'].astype('float32') 
        metadata['extent'] =  metadata['extent'].astype('float32')
        x_center = torch.tensor(metadata['center'].wave, dtype=torch.float32) #numeric_values_center,  #torch.Size([3]) tensor([0.0395, 0.0495, 0.0045], device='cuda:0'
        x_extent = torch.tensor(metadata['extent'].wave, dtype=torch.float32) #numeric_values_extent torch.Size([3])tensor([0.0800, 0.1000, 0.0100], device='cuda:0')

        stats = example.describe()
        self.u_loc = torch.tensor(stats['mean'].wave)   #input_loc sono le coordinate  #torch.Size([3]) device='cuda:0', dtype=torch.complex128)
        self.u_scale = torch.tensor(stats['std'].wave) ##torch.Size([3]) tensor([0.0002, 0.0003, 0.0049], device='cuda:0', dtype=torch.float64)
        self.mu_loc = torch.tensor(stats['mean'].mre) # #torch.Size([1])   tensor([3382.3750+565.4867j], device='cuda:0', dtype=torch.complex128)
        self.mu_scale = torch.tensor(stats['std'].mre) # #torch.Size([1])tensor([1590.7276], device='cuda:0', dtype=torch.float64)
        self.omega = torch.tensor(omega)   # torch.Size([]) tensor(90, device='cuda:0')

        if 'anat' in example:
            self.a_loc = torch.tensor(stats['mean'].anat)
            self.a_scale = torch.tensor(stats['std'].anat)
        else:
            self.a_loc = torch.zeros(0) #torch.Size([0]) tensor([], device='cuda:0')
            self.a_scale = torch.zeros(0) #torch.Size([0])   tensor([], device='cuda:0')

        self.input_loc = x_center #torch.Size([3]) tensor([0.0395, 0.0495, 0.0045], device='cuda:0'
        self.input_scale = x_extent #torch.Size([3])tensor([0.0800, 0.1000, 0.0100], device='cuda:0')
        u_droupout_rate = 0
        mu_dropout_rate = 0
        if u_dropout:
            u_dropout_rate = 5
        if mu_dropout:
            mu_dropout_rate = 5

        self.u_pinn = PINN(
            n_input=len(self.input_loc), #3
            n_output=len(self.u_loc), #3
            complex_output=example.wave.field.is_complex, #whether the output of the neural network should be treated as complex numbers or not.
            polar_output=False, #whether the output should be represented in polar coordinates or not. Polar coordinates consist of a magnitude (or amplitude) and a phase angle
            activ_fn=activ_fn[0],
            siren_weight_init=siren_init,
            **kwargs
        )
        self.mu_pinn = PINN(
            n_input=len(self.input_loc), #3
            n_output=len(self.mu_loc) + len(self.a_loc), #1+0=1
            complex_output=example.mre.field.is_complex,
            polar_output=True,
            activ_fn=activ_fn[1],
            siren_weight_init=siren_init,
            **kwargs
        )
        self.regularizer = None

    def forward(self, inputs):
        x, = inputs
        x = (x - self.input_loc) / self.input_scale  # normalises
        x = x * self.omega  

        u_pred = self.u_pinn(x)  #applies the network
        u_pred = u_pred * self.u_scale + self.u_loc  # renormalises

        mu_a_pred = self.mu_pinn(x)
        mu_pred = mu_a_pred[:,:len(self.mu_loc)]
        a_pred = mu_a_pred[:,len(self.mu_loc):]

        mu_pred = mu_pred * self.mu_scale + self.mu_loc
        a_pred = a_pred * self.a_scale + self.a_loc

        return u_pred, mu_pred, a_pred


class PINN(torch.nn.Module):  #torch.nn.Module is base class for all neural network modules-->Your models should also subclass this class.
    """
    Physics-Informed Neural Network (PINN) model.
    Parameters
    ----------
    n_input : int
        Number of input features.
    n_output : int
        Number of output features.
    n_layers : int
        Number of hidden layers.
    n_hidden : int
        Number of hidden units per layer.
    activ_fn : str
        Activation function for hidden layers.
    dense : bool
        Whether to use dense connections between layers.
    polar_input : bool
        Whether the input should be represented in polar coordinates.
    complex_output : bool
        Whether the output should be treated as complex numbers.
    polar_output : bool
        Whether the output should be represented in polar coordinates.
    siren_weight_init : bool
        Whether to use SIREN weight initialization.
    dropout_layer_rate : float
        Rate in which to use dropout layers between dense layers
    """
    def __init__(
        self,
        n_input,
        n_output,
        n_layers,
        n_hidden,
        activ_fn='s',
        dense=True,
        polar_input=False,
        complex_output=False,
        polar_output=False,
        siren_weight_init=True,
        dropout_layer_rate=0
    ):
        assert n_layers > 0
        super().__init__()

        if polar_input:
            n_input += 3

        self.hiddens = []
          #It initializes hidden layers (self.hiddens) based on the specified parameters using linear transformations (torch.nn.Linear).

        for i in range(n_layers - 1):

            hidden = torch.nn.Dropout(p=0.5) if (dropout_layer_rate > 0 and i % dropout_layer_rate == 0) else torch.nn.Linear(n_input, n_hidden)
            if dense:
                n_input += n_hidden
            else:
                n_input = n_hidden
            self.hiddens.append(hidden)
            self.add_module(f'hidden{i}', hidden)
        #It initializes the output layer (self.output) using linear transformation based on the number of input features.
        if complex_output:
            self.output = torch.nn.Linear(n_input, n_output * 2)
        else:
            self.output = torch.nn.Linear(n_input, n_output)
        #It initializes the activation function (self.activ_fn) based on the provided activation function string.
        self.activ_fn = get_activ_fn(activ_fn)
        self.dense = dense
        self.polar_input = polar_input
        self.complex_output = complex_output
        self.polar_output = polar_output

        if siren_weight_init:
            self.init_weights_siren()
        else:
            self.init_weights_random()

    def forward(self, x):
        '''
        Args:
            x: (n_points, n_input)
        Returns:
            u: (n_points, n_output)
        '''
        if self.polar_input: # polar coordinates
            x, y, z = torch.split(x, 1, dim=-1)
            r = torch.sqrt(x**2 + y**2)
            sin, cos = x / (r + 1), y / (r + 1)
            x = torch.cat([x, y, z, r, sin, cos], dim=-1)

        # hidden layers
        for i, hidden in enumerate(self.hiddens):
            if i == 0:
                y = torch.sin(hidden(x))
            else:
                y = self.activ_fn(hidden(x))
            if self.dense:
                x = torch.cat([x, y], dim=1)
            else:
                x = y
        
        # output layer
        if self.complex_output:
            return as_complex(self.output(x), polar=self.polar_output)
        else:
            return self.output(x)

    def init_weights_siren(self, c=6):
        '''
        SIREN weight initialization.
        '''
        for i, module in enumerate(self.children()):
            if not hasattr(module, 'weight'):
                continue
            n_input = module.weight.shape[-1]

            if i == 0: # first layer
                w_std = 1 / n_input
            else:
                w_std = np.sqrt(c / n_input)

            with torch.no_grad():
                module.weight.uniform_(-w_std, w_std)

    def init_weights_random(self):
        '''
        Random weight initialization.
        '''
        for module in self.children():
            if hasattr(module, 'weight'):
                torch.nn.init.xavier_normal_(module.weight)
                torch.nn.init.zeros_(module.bias)