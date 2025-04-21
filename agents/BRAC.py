'''
Behavior Regularized Actor-Critic (BRAC) Algorithm
Inspired by Homework 7 TD3 implementation
'''

from torch.distributions import Normal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last_dim, h), nn.ReLU()]
            last_dim = h
        layers += [nn.Linear(last_dim, output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.base = MLP(state_dim, action_dim * 2, hidden_sizes)

    def forward(self, state):
        mean_logstd = self.base(state)
        mean, log_std = torch.chunk(mean_logstd, 2, dim=-1)
        log_std = torch.clamp(log_std, -5, 2)
        std = log_std.exp()
        dist = Normal(mean, std)
        return dist

    def sample(self, state):
        dist = self.forward(state)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

class BRAC:
    def __init__(self, state_dim, action_dim, alpha=0.1, gamma=0.99, tau=0.005, lr=3e-4):
        self.actor = GaussianPolicy(state_dim, action_dim)
        self.critic = MLP(state_dim + action_dim, 1)
        self.target_critic = MLP(state_dim + action_dim, 1)
        self.behavior_policy = GaussianPolicy(state_dim, action_dim)  # Pre-trained with BC loss

        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        self.alpha = alpha  # KL regularization coefficient
        self.gamma = gamma
        self.tau = tau

    def train_behavior_policy(self,dataset, optimizer, batch_size=256, epochs=100):
        states, actions = dataset
        dataset_size = states.size(0)

        for epoch in range(epochs):
            permutation = torch.randperm(dataset_size)
            epoch_loss = 0.0

            for i in range(0, dataset_size, batch_size):
                indices = permutation[i:i + batch_size]
                batch_states = states[indices]
                batch_actions = actions[indices]

                dist = self.behavior_policy(batch_states)
                predicted_mean = dist.mean  # Use predicted mean, not samples

                loss = F.mse_loss(predicted_mean, batch_actions)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")


    def train(self, batch):
        state, action, reward, next_state, done = batch

        # Critic update
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            q_target = self.target_critic(torch.cat([next_state, next_action], dim=-1))
            target = reward + self.gamma * (1 - done) * q_target

        q_val = self.critic(torch.cat([state, action], dim=-1))
        critic_loss = F.mse_loss(q_val, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update with KL regularization to behavior policy
        new_action, log_prob = self.actor.sample(state)
        q_val = self.critic(torch.cat([state, new_action], dim=-1))

        # KL divergence between current policy and behavior policy
        pi_dist = self.actor(state)
        mu_dist = self.behavior_policy(state)
        mu_dist.detach()
        kl_div = torch.distributions.kl.kl_divergence(pi_dist, mu_dist).sum(dim=-1, keepdim=True)

        actor_loss = (-q_val + self.alpha * kl_div).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Update target critic
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)