import torch
import torch.nn as nn
from torch.autograd import grad
from torch.nn import Tanh
from torch_scatter import scatter


class MLP(nn.Module):
    def __init__(
        self,
        n_input_nodes,
        n_layers,
        n_hidden_size,
        activation,
        batchnorm,
        n_output_nodes=1,
        latent=False,
    ):
        super(MLP, self).__init__()
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
        layers.append(nn.Linear(self.n_neurons[-2], self.n_neurons[-1]))
        self.model_net = nn.Sequential(*layers)
        self.latent = latent
        self._forward = self._lat_forward if latent else self._reg_forward

        # TODO: identify optimal initialization scheme
        # self.reset_parameters()

    def reset_parameters(self):
        for m in self.model_net:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

    def forward(self, inputs):
        return self._forward(inputs)

    def _reg_forward(self, inputs):
        return self.model_net(inputs)

    def _lat_forward(self, inputs):
        latents = self.model_net[:-1](inputs)
        return self.model_net[-1](latents), latents


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


class ElementMask(nn.Module):
    def __init__(self, elements):
        super(ElementMask, self).__init__()
        nelems = len(elements)
        weights = torch.zeros(100, nelems)
        weights[elements, range(nelems)] = 1.0

        self.mask = nn.Embedding(100, nelems)
        self.mask.weight.data = weights

    def forward(self, atomic_numbers):
        return self.mask(atomic_numbers)


class BPNN(nn.Module):
    def __init__(
        self,
        elements,
        input_dim,
        num_nodes,
        num_layers,
        get_forces=True,
        batchnorm=False,
        activation=Tanh,
        latent=False,
    ):
        super(BPNN, self).__init__()
        self.get_forces = get_forces
        self.activation_fn = activation

        n_elements = len(elements)
        self.elementwise_models = nn.ModuleList()
        self.latent = latent
        for element in range(n_elements):
            self.elementwise_models.append(
                MLP(
                    n_input_nodes=input_dim,
                    n_layers=num_layers,
                    n_hidden_size=num_nodes,
                    activation=activation,
                    batchnorm=batchnorm,
                    latent=latent,
                )
            )

        self.element_mask = ElementMask(elements)

    def forward(self, batch):
        if isinstance(batch, list):
            batch = batch[0]
        with torch.enable_grad():
            atomic_numbers = batch.atomic_numbers
            fingerprints = batch.fingerprint
            fingerprints.requires_grad = True
            image_idx = batch.image_idx
            sorted_image_idx = torch.unique_consecutive(image_idx)
            mask = self.element_mask(atomic_numbers)

            nets_energy_predictions = []
            nets_latents = []
            for net in self.elementwise_models:
                results = net(fingerprints)
                if self.latent:
                    net_e, net_latent = results
                else:
                    net_e, net_latent = results, torch.tensor([], device=results.device)
                nets_energy_predictions.append(net_e)
                net_latent = torch.FloatTensor(net_latent, device=net_e.device)
                # print(i, net_latent.size())
                nets_latents.append(torch.flatten(net_latent))  # flatten first

            o = torch.sum(
                mask * torch.cat(nets_energy_predictions, dim=1),
                dim=1,
            )
            latent = torch.cat(
                nets_latents, dim=0
            )  # dim=0 --> flat, dim=1 --> rectangular

            energy = scatter(o, image_idx, dim=0)[sorted_image_idx]

            if self.get_forces:
                gradients = grad(
                    energy,
                    fingerprints,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0].view(1, -1)

                forces = -1 * torch.sparse.mm(batch.fprimes.t(), gradients.t()).view(
                    -1, 3
                )

            else:
                forces = torch.tensor([], device=energy.device)

            return energy, forces, latent

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class CustomLoss(nn.Module):
    def __init__(self, force_coefficient=0, loss="mae"):
        super(CustomLoss, self).__init__()
        self.alpha = force_coefficient
        self.loss = loss

        if self.loss == "mae":
            self.loss = nn.L1Loss()
        elif self.loss == "mse":
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError(f"{self.loss} loss not available!")

    def forward(self, prediction, target):

        energy_pred = prediction[0]
        energy_target = target[0]
        energy_loss = self.loss(energy_pred, energy_target)
        force_pred = prediction[1]
        if force_pred.nelement() == 0:
            self.alpha = 0

        if self.alpha > 0:
            force_target = target[1]
            force_loss = self.loss(force_pred, force_target)
            loss = 0.5 * (energy_loss + self.alpha * force_loss)
        else:
            loss = 0.5 * energy_loss
        return loss
