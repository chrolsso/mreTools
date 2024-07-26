import torch
import numpy as np

from MREpark.mreTools.pinn.utils import as_complex

class MrePinn(torch.nn.Module):
    def __init__(
        self,
        n_input,
        n_output,
        output_polar=False,
        weight_init_scheme='siren'
    ):
        assert n_input > 0
        assert n_output > 0

        super().__init__()
        self.num_hidden_layers = 3
        self.p_dropout = 0.5
        self.num_hidden_nodes = 128

        self.n_input = n_input
        self.n_output = n_output
        self.output_polar = output_polar
        self.weight_init_scheme = weight_init_scheme

        self.input_layer = torch.nn.Linear(self.n_input, 128)
        self.input_activation = torch.sin

        num_input_nodes = n_input
        self.layers = []
        for i in range(self.num_hidden_layers):
            self.layers.append(torch.nn.Linear(num_input_nodes, self.num_hidden_nodes))
            num_input_nodes += self.num_hidden_nodes
        self.layer_activation = torch.sin
        self.dropout_layer = torch.nn.Dropout(p=self.p_dropout)

        self.output_layer = torch.nn.Linear(num_input_nodes, n_output * 2)  # this *2 is the only thing that makes this network complex. Why don't we split much earlier between real and imaginary parts? 
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
        if (len(self.layers) > 0):
            y = self.layers[0](x)
            y = self.layer_activation(y)
            x = torch.cat([x, y], dim=1)

            # dropout layer
            x = self.dropout_layer(x)
            x = self.layer_activation(x)

            for i in range(1, self.num_hidden_layers):
                y = self.layers[i](x)
                y = self.layer_activation(y)
                x = torch.cat([x, y], dim=1)

        x = self.output_layer(x)
        x = self.output_activation(x)
        
        return as_complex(x, polar=self.output_polar)

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