import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(layer_dims, 
               activation_fn=nn.ReLU,
               output_activation_fn=None):
    """
    Create a multi-layer perceptron (MLP) PyTorch model.

    Args:
        layer_dims (list): a list of layer dimensions including input and 
            output dimension. 
        activation_fn (Type[nn.Module]): activation function module after 
            each linear layer.
        output_activation_fn (Type[nn.Module]): activation function for output layer

    Returns: 
        nn.Sequential: the MLP model
    """               
    n = len(layer_dims)
    assert n >= 2, 'MLP requires at least two dimensions (input and output)'

    layers = []               
    for i in range(n - 2):
        layers.extend([nn.Linear(layer_dims[i], layer_dims[i+1]), activation_fn()]) 
    layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
    if output_activation_fn is not None:
        layers.append(output_activation_fn)

    return nn.Sequential(*layers).to(torch.float32)
  

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
    def __init__(self, features, act=F.relu, dropout_rate=None, use_layer_norm=False):
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
        if dropout_rate is not None and dropout_rate > 0.0:
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
    """Gaussian random features for encoding temporal information.

    This module uses fixed random weights to project inputs into a space where
    they can be more easily separated or processed by subsequent layers.
    """
    def __init__(self, embed_dim, scale=30.):
        super(GaussianFourierProjection, self).__init__()
        # Initialize fixed random weights
        self.register_buffer('W', torch.randn(embed_dim // 2) * scale, persistent=False)

    def forward(self, x):
        """Applies the Gaussian Fourier projection to the input tensor."""
        x_proj = x.unsqueeze(dim=-1) * self.W.unsqueeze(dim=0) * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)



  
