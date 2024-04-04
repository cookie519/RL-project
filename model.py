import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(dims, activation=nn.ReLU, output_activation=None):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dimensions (input and output)'
  
    layers = []
    for i in range(n_dims - 2):
        layers.extend([nn.Linear(dims[i], dims[i+1]), activation()])
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
      
    net = nn.Sequential(*layers)
    net = net.to(dtype=torch.float32)
    return net
  

class TwinQ(nn.Module):
    def __init__(self, state_dim, action_dim, num_hidden_layers=2, hidden_dim=256):
        super(TwinQ, self).__init__()
        input_dim = state_dim + action_dim
        dims = [input_dim] + [hidden_dim] * num_hidden_layers + [1]
        self.q1 = mlp(dims)
        self.q2 = mlp(dims)

    def both(self, action, condition=None):
        x = torch.cat([action, condition], dim=-1) if condition is not None else action
        return self.q1(x), self.q2(x)

    def forward(self, action, condition=None):
        return torch.min(*self.both(action, condition))


class ValueFunction(nn.Module):
    def __init__(self, state_dim, num_hidden_layers=2, hidden_dim=256):
        super(ValueFunction, self).__init__()
        dims = [state_dim] + [hidden_dim] * num_hidden_layers + [1]
        self.v = mlp(dims)

    def forward(self, state):
        return self.v(state)


class DiracPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, num_hidden_layers=2, hidden_dim=256):
        super(DiracPolicy, self).__init__()
        dims = [state_dim] + [hidden_dim] * num_hidden_layers + [action_dim]
        self.net = mlp(dims, output_activation=nn.Tanh)

    def forward(self, state):
        return self.net(state)

    def select_actions(self, state):
        return self(state)


class MLPResNetBlock(nn.Module):
    def __init__(self, features, act=nn.ReLU(), dropout_rate=0.0, use_layer_norm=False):
        super(MLPResNetBlock, self).__init__()
        self.features = features
        self.act = act
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm

        layers = [nn.Linear(features, features * 4),
                  act,
                  nn.Linear(features * 4, features)]
        if use_layer_norm:
            layers.insert(1, nn.LayerNorm(features))
        if dropout_rate > 0.0:
            layers.insert(1, nn.Dropout(dropout_rate))
        self.layers = nn.Sequential(*layers)

        self.residual = nn.Linear(features, features)

    def forward(self, x, training=False):
        residual = x
        x = self.layers(x)
        if residual.shape != x.shape:
            residual = self.residual(residual)
        return residual + x


class MLPResNet(nn.Module):
    def __init__(self, num_blocks, input_dim, out_dim, dropout_rate=None, use_layer_norm=False, hidden_dim=256, activations=F.relu):
        super(MLPResNet, self).__init__()
        self.num_blocks = num_blocks
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.hidden_dim = hidden_dim
        self.activations = activations

        self.fc = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList([MLPResNetBlock(hidden_dim, activations, dropout_rate, use_layer_norm)
                                     for _ in range(num_blocks)])

        self.out_fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, training=False):
        x = self.fc(x)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.activations(x)
        x = self.out_fc(x)

        return x


class ScoreNet_IDQL(nn.Module):

class GaussianFourierProjection(nn.Module):


  
