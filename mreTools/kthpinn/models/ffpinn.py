import torch
from torch import nn
import numpy as np

class FfPinn(nn.Module):
    """Feed-Forward Pinn model inspired by Ragoza et al.
    """
    def __init__(
        self,
        device,
        n_input = 2,
        n_output = 2,
        output_polar=False,
        weight_init_scheme='siren'
    ):
        assert n_input > 0
        assert n_output > 0

        super().__init__()
        self.p_dropout = 0.01
        self.num_hidden_nodes = 128

        self.n_input = n_input
        self.n_output = n_output
        self.output_polar = output_polar
        self.weight_init_scheme = weight_init_scheme

        self.input_layer = torch.nn.Linear(self.n_input, 128, device=device)
        self.input_activation = torch.sin

        num_input_nodes = 128
        self.hidden_1 = torch.nn.Linear(num_input_nodes, self.num_hidden_nodes, device=device)
        num_input_nodes += self.num_hidden_nodes
        self.hidden_2 = torch.nn.Linear(num_input_nodes, self.num_hidden_nodes, device=device)
        num_input_nodes += self.num_hidden_nodes
        self.hidden_3 = torch.nn.Linear(num_input_nodes, self.num_hidden_nodes, device=device)
        num_input_nodes += self.num_hidden_nodes
        self.layer_activation = torch.sin
        self.dropout_layer = torch.nn.Dropout(p=self.p_dropout)

        self.output_layer = torch.nn.Linear(num_input_nodes, n_output, device=device)
        self.output_activation = torch.sin

        if self.weight_init_scheme == 'siren':
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

        # input layer
        x = self.input_layer(x)
        x = self.input_activation(x)

        # hidden layers
        y = self.hidden_1(x)
        y = self.layer_activation(y)
        x = torch.cat([x, y], dim=1)

        # dropout layer
        x = self.dropout_layer(x)
        x = self.layer_activation(x)

        y = self.hidden_2(x)
        y = self.layer_activation(y)
        x = torch.cat([x, y], dim=1)

        y = self.hidden_3(x)
        y = self.layer_activation(y)
        x = torch.cat([x, y], dim=1)

        x = self.output_layer(x)
        x = self.output_activation(x)
        
        return x

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