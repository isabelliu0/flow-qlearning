"""
Networks module for Flow Q-Learning implmentation.
Contains all neural network architectures needed for FQL.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP (nn.Module):
    """Multi-layer peceptron with optional layer normalization.
    
    NOTE: According to the FQL paper, they use [512, 512, 512, 512]-sized MLP for all neural networks, and also layer normalization for value networks.
    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        activation=nn.GELU(),
        activate_final=False,
        layer_norm=False,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            linear = nn.Linear(prev_dim, hidden_dim)
            # NOTE: a (semi) orthogonal matrix provides good solutions to learning nonlinear dynamics
            nn.init.orthogonal_(linear.weight, gain=1.0)
            nn.init.zeros_(linear.bias)

            layers.append(linear)
            layers.append(activation)
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            prev_dim = hidden_dim
        
        final_layer = nn.Linear(prev_dim, output_dim)
        nn.init.orthogonal_(final_layer.weight, gain=1e-2)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)

        if activate_final:
            layers.append(activation)
            if layer_norm:
                layers.append(nn.LayerNorm(output_dim))
        
        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim
    
    def forward(self, x):
        return self.net(x)


class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.
    
    Takes (state, action, time) and outputs velocity vectors.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=(512, 512, 512, 512),
        layer_norm=False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        input_dim = state_dim + action_dim + 1  # +1 for time
        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim,
            activate_final=False,
            layer_norm=layer_norm,
        )
    
    def forward(self, observations, actions, times):
        inputs = torch.cat([observations, actions, times], dim=-1)
        velocities = self.net(inputs)
        return velocities


class ActorOneStepPolicy(nn.Module):
    """One-step actor policy.
    Maps directly from (state, noise) to actions.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=(512, 512, 512, 512),
        layer_norm=False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        input_dim = state_dim + action_dim  # action_dim here is for the noise dimension
        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim,
            activate_final=False,
            layer_norm=layer_norm,
        )
    
    def forward(self, observations, noises):
        inputs = torch.cat([observations, noises], dim=-1)
        actions = self.net(inputs)
        return torch.tanh(actions)

class Critic(nn.Module):
    """Critic network (Q-function).
    
    NOTE: We use two Q functions to improve stability. Following the FQL paper,
    1. Use mean of Q values for Q loss term in actor objective;
    2. Use mean of target values in critic objective (by default);
    3. Use minimum of Q values for OGBench antmaze-{large, giant} tasks.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=(512, 512, 512, 512),
        num_critics=2,
        layer_norm=True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_critics = num_critics
        
        self.critics = nn.ModuleList()
        for _ in range(num_critics):
            critic = MLP(
                input_dim=state_dim + action_dim,
                hidden_dims=hidden_dims,
                output_dim=1,
                activate_final=False,
                layer_norm=layer_norm
            )
            self.critics.append(critic)
    
    def forward(self, observations, actions):
        inputs = torch.cat([observations, actions], dim=-1)
        q_values = []
        for critic in self.critics:
            q = critic(inputs)
            q_values.append(q.squeeze(-1))
        return torch.stack(q_values)
