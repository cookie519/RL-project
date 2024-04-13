import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(layer_dims, 
         activation=nn.ReLU,
         output_activation=None):
    """
    Create a multi-layer perceptron (MLP) PyTorch model.

    Args:
        layer_dims (list): a list of layer dimensions including input and 
            output dimension. 
        activation (Type[nn.Module]): activation function module after 
            each linear layer.
        output_activation (Type[nn.Module]): activation function for output layer

    Returns: 
        nn.Sequential: the MLP model
    """               
    n = len(layer_dims)
    assert n >= 2, 'MLP requires at least two dimensions (input and output)'

    layers = []               
    for i in range(n - 2):
        layers.extend([nn.Linear(layer_dims[i], layer_dims[i+1]), activation()]) 
    layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
    if output_activation  is not None:
        layers.append(output_activation())

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
    """
    A policy network approximating the policy function π(s) -> a, mapping state inputs to actions.
    Utilizes a tanh activation on the output to ensure the action values are within a normalized range.
    
    Args:
    - state_dim (int): Dimensionality of the state space.
    - action_dim (int): Dimensionality of the action space.
    - num_hidden_layers (int): Number of hidden layers in the network.
    - hidden_dim (int): Sizer of hidden layers in the network.
    """
    def __init__(self, state_dim, action_dim, num_hidden_layers=2, hidden_dim=256):
        super(DiracPolicy, self).__init__()
        dims = [state_dim] + [hidden_dim] * num_hidden_layers + [action_dim]
        self.net = mlp(dims, output_activation=nn.Tanh)

    def forward(self, state):
        return self.net(state)

    def select_actions(self, state):
        return self(state)


class MLPResNetBlock(nn.Module):
    """
    A block of the MLPResNet architecture, implementing a residual connection around two linear layers.
    
    Args:
    - features (int): Number of features in the input and output of the block.
    - act (callable): Activation function to be applied after the first linear layer.
    - dropout_rate (float, optional): Dropout rate applied after the activation function. If None or 0, dropout is not used.
    - use_layer_norm (bool): Flag indicating whether to use Layer Normalization before the activation function.
    """
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
    """
    Defines a ResNet-like architecture with multiple MLPResNetBlocks.
    
    Args:
        num_blocks (int): The number of MLPResNet blocks to include in the model. Each block includes
                          a residual connection around its layers.
        input_dim (int): The dimensionality of the input features.
        out_dim (int): The dimensionality of the output.
        dropout_rate (float, optional): The dropout rate applied within each MLPResNet block. If None or 0,
                                        no dropout is applied. 
        use_layer_norm (bool): Flag indicating whether to apply Layer Normalization within each block.
        hidden_dim (int): The number of units in the hidden layers within each block.
        activations (callable): The activation function to apply after each linear layer except for the final
                                output layer. 

    """
    def __init__(self, num_blocks, input_dim, out_dim, dropout_rate=None, use_layer_norm=False, hidden_dim=256, activations=F.relu):
        super(MLPResNet, self).__init__()
        self.num_blocks = num_blocks
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.hidden_dim = hidden_dim
        self.activations = activations

        self.fc = nn.Linear(input_dim + 128, hidden_dim)

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



class ScoreNet_IDQL(nn.Module):
    """
    Implements a Score Network for Inverse Dynamics Q-Learning (IDQL), integrating an embedding layer for time conditioning.

    Args:
        input_dim (int): The dimensionality of the input state space.
        output_dim (int): The dimensionality of the output action space.
        marginal_prob_std (float): The standard deviation of the marginal probability distribution of the
                                   input states. 
        embed_dim (int, optional): The dimensionality of the embedding for the time variable, used in the
                                   Gaussian Fourier Projection layer. Defaults to 64.
        args: other arguments.
    """
    def __init__(self, input_dim, output_dim, marginal_prob_std, embed_dim=64, args=None):
        super().__init__()
        self.output_dim = output_dim
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim))
        self.device = args.device
        self.marginal_prob_std = marginal_prob_std
        self.args=args

        # Main network with residual blocks
        self.main = MLPResNet(args.actor_blocks, input_dim, output_dim, dropout_rate=0.1, use_layer_norm=True, hidden_dim=256, activations=nn.Mish())

        # Conditional model for embedding the time variable
        self.cond_model = mlp([64, 128, 128], activation=nn.Mish, output_activation=None)

    def forward(self, x, t, condition):
        embed = self.cond_model(self.embed(t))
        all = torch.cat([x, condition, embed], dim=-1)
        h = self.main(all)
        return h
         



  
