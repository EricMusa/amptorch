import torch
import torch.nn as nn
from torch.autograd import grad
from torch.nn import Tanh
from torch_scatter import scatter


class LatentMLP(nn.Module):
    def __init__(
        self,
        n_input_nodes,
        n_layers,
        n_hidden_size,
        activation,
        batchnorm,
        n_output_nodes=1,
    ):
        super(LatentMLP, self).__init__()
        if isinstance(n_hidden_size, int):
            n_hidden_size = [n_hidden_size] * (n_layers)
        self.n_neurons = [n_input_nodes] + n_hidden_size + [n_output_nodes]
        self.activation = activation
        layers = []
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(self.n_neurons[_], self.n_neurons[_ + 1]))
            layers.append(activation())
            if batchnorm:
                layers.append(nn.BatchNorm1d(self.n_neurons[_ + 1]))
        self.model_net = nn.Sequential(*layers)
        self.output_layer = nn.Linear(self.n_neurons[-2], self.n_neurons[-1])
        self.n_latent = self.n_neurons[-2]

        # TODO: identify optimal initialization scheme
        # self.reset_parameters()

    def reset_parameters(self):
        for m in self.model_net:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

    def forward(self, inputs):
        latent = self.model_net(inputs)
        return self.output_layer(latent), latent
