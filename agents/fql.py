import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agents.networks import ActorVectorField, Critic, ActorOneStepPolicy


def get_config():
    """
    Returns default hyperparameters for Flow Q-Learning agent.
    """
    config = {
        # Learning settings
        'agent_name': 'fql',          
        'lr': 3e-4,                   # Learning rate for optimizer
        'batch_size': 256,            #  # samples per training batch
        'discount': 0.99,             # Discount factor γ for TD targets
        'tau': 0.005,                 # averaging rate for target updates
        'q_agg': 'min',              # specify how to aggregate ensemble Q-values: 'mean' or 'min'
        'alpha': 1.0,                # Weight for distillation loss in actor objective
        'flow_steps': 10,             #  # of  Euler integration steps for BC flow
        'normalize_q_loss': True,    # Whether to re-scale Q-loss term
        # Network architecture settings
        'actor_hidden_dims': (512, 512, 512, 512),
        'value_hidden_dims': (512, 512, 512, 512),
        'layer_norm': True,           # Whether or not to ise layer normalization in value network
        'actor_layer_norm': False,    # Whether or not to use layer normalization in actor networks
        # 'encoder': None,              # Name for visual encoder (e.g. 'impala_small') or None
    }
    return config

class FQLAgent(nn.Module):
    """
    Combines an ensemble critic (Q-function) and a flow-based actor.
    """

    def __init__(self, config, ob_dim, action_dim, device="cpu",seed=0):
        super().__init__()
        # Save config and device
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ob_dim = ob_dim
        self.action_dim = action_dim
        self.device = device
        print(f"Device for FQL is {self.device}")

        # random seed
        torch.manual_seed(seed)

        # Build the value (critic) network with ensemble heads
        self.critic = Critic(
            state_dim = ob_dim,
            action_dim = action_dim,
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm']
        ).to(device)
        # Create a target critic for computing stable TD targets
        self.target_critic = copy.deepcopy(self.critic).to(device)
        # Freeze target network - only updated via soft updates
        for param in self.target_critic.parameters():
            param.requires_grad = False

        # Build the BC flow actor network (predicts velocity field)
        self.actor_bc_flow = ActorVectorField(
            state_dim = ob_dim,
            action_dim = action_dim,
            hidden_dims=config['actor_hidden_dims'],
            layer_norm=config['actor_layer_norm']
        ).to(device)
        # Build the one-step flow actor network (distilled policy)
        self.actor_onestep_flow = ActorOneStepPolicy(
            state_dim = ob_dim,
            action_dim = action_dim,
            hidden_dims=config['actor_hidden_dims'],
            layer_norm=config['actor_layer_norm']
        ).to(device)

        # Register networks as submodules
        self.add_module('critic', self.critic)
        self.add_module('target_critic', self.target_critic)
        self.add_module('actor_bc_flow', self.actor_bc_flow)
        self.add_module('actor_onestep_flow', self.actor_onestep_flow)

        # Combine all parameters into one optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=config['lr'])
        

    def critic_loss(self, batch):
        """Compute the critic (Q-function) mean-squared TD error."""
        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        next_observations = batch['next_observations']
        masks = batch['masks']

        # Ensure rewards and masks have shape [batch_size, 1]
        if len(rewards.shape) == 1:
            rewards = rewards.unsqueeze(-1)
        if len(masks.shape) == 1:
            masks = masks.unsqueeze(-1)

        with torch.no_grad():
            # Sample next actions from current policy, then clip to [-1,1]
            next_actions = self.sample_actions(next_observations, seed=0)
            next_actions = torch.clamp(next_actions, -1.0, 1.0)

            # Compute target Q-values: use target critic network
            with torch.no_grad():
                next_qs = self.target_critic(next_observations, next_actions)
            if self.config['q_agg'] == 'min':
                # Take minimum across ensemble for conservative estimate
                next_q = torch.min(next_qs, dim=0)[0]
            else:
                # Take mean across ensemble
                next_q = torch.mean(next_qs, dim=0)
            if len(next_q.shape) == 1:
                next_q = next_q.unsqueeze(-1)

            # Build TD target r + γ * mask * Q(s', a')
            # Shape [batch_size, 1]
            target_q = rewards + self.config['discount'] * masks * next_q

        # Compute current Q-values from critic under current parameters
        current_qs = self.critic(observations, actions)
 
        # Compute MSE loss for each critic in the ensemble
        critic_loss = 0
        for q in current_qs:
            q = q.unsqueeze(-1)
            critic_loss += F.mse_loss(q, target_q)
        critic_loss /= len(current_qs)

        # Logging diagnostics
        diagnostics = {
            'critic_loss': critic_loss.item(),
            'q_mean': current_qs.mean().item(),
            'q_max': current_qs.max().item(),
            'q_min': current_qs.min().item(),
        }
        return critic_loss, diagnostics

    def actor_loss(self, batch):
        """
        Compute the actor (policy) loss, which combines:
          1. BC flow reconstruction loss,
          2. Distillation loss to one-step policy,
          3. Negative Q-value (to maximize Q).

        Args:
            batch: dict with tensors 'observations' and 'actions'

        Returns:
            loss      : scalar tensor of actor loss
            diagnostics: dict of logging values
        """
        observations = batch['observations']
        batch_size = observations.shape[0]

        # 1. BC flow loss - train the flow model to match noise to action
        # Sample Gaussian noise x0
        x0 = torch.randn(batch_size, self.action_dim, device=self.device)
        x1 = batch['actions']   # Expert action x1
        t = torch.rand(batch_size, 1, device=self.device)
        xt = (1 - t) * x0 + t * x1
        v_true = x1 - x0    # True velocity vector
        v_pred = self.actor_bc_flow(observations, xt, t)
        bc_flow_loss = F.mse_loss(v_pred, v_true)

        # 2) Distillation loss
        # Sample noises for one-step policy
        noises = torch.randn(batch_size, self.action_dim, device=self.device)
        # Compute "ground truth" multi-step flow actions
        with torch.no_grad():
            target_flow_actions = self.compute_flow_actions(observations, noises)
        actor_actions = self.actor_onestep_flow(observations, noises)
        actor_actions_clipped = torch.clamp(actor_actions, -1.0, 1.0)
        distill_loss = F.mse_loss(actor_actions_clipped, target_flow_actions)

        # 3) Q-based loss to maximize value
        actor_actions_clipped = torch.clamp(actor_actions, -1.0, 1.0)
        qs = self.critic(observations, actor_actions_clipped)
        q_val = qs.mean(dim=0)  # Average across ensemble and batch
        q_loss = -q_val.mean()
        # Optionally normalize Q-loss magnitude
        if self.config['normalize_q_loss']:
            factor = 1.0 / (torch.abs(q_val).mean().detach() + 1e-8)
            q_loss = factor * q_loss

        # Total actor loss
        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        # Logging diagnostics
        mse = F.mse_loss(actor_actions, batch['actions'])
        diagnostics = {
            'actor_loss': actor_loss.item(),
            'bc_flow_loss': bc_flow_loss.item(),
            'distill_loss': distill_loss.item(),
            'q_loss': q_loss.item(),
            'q_mean': q_val.mean().item(),
            'mse': mse.item(),
        }
        return actor_loss, diagnostics

    def total_loss(self, batch):
        """
        Compute combined critic + actor loss and merge diagnostics.
        """
        # Critic loss
        c_loss, c_info = self.critic_loss(batch)
        # Actor loss
        a_loss, a_info = self.actor_loss(batch)
        # Sum of both losses
        loss = c_loss + a_loss

        # Merge diagnostics under separate keys
        info = {f'critic/{k}': v for k, v in c_info.items()}
        info.update({f'actor/{k}': v for k, v in a_info.items()})

        # PRINT LOSS FOR LOSS CURVE (COMMENT OUT)
        # with open("losscurve.txt", "a") as f:
        #     f.write(str(loss.item())+"\n")

        return loss, info

    def update(self, batch):
        """
        Perform one gradient update step:
          - Zero gradients
          - Compute total loss
          - Backpropagate
          - Step optimizer
          - Soft-update target critic

        Returns:
            info: dict of logged diagnostics
        """
        # Set module to train mode
        self.train()
        # Zero existing gradients
        self.optimizer.zero_grad()

        # Standardize batch format
        # If batch is a tuple (from ReplayBuffer), convert to dict
        if isinstance(batch, tuple):
            observations, actions, next_observations, rewards, masks = batch
            batch = {
                'observations': observations,
                'actions': actions,
                'next_observations': next_observations,
                'rewards': rewards,
                'masks': masks
            }

        # Compute loss and info
        loss, info = self.total_loss(batch)
        loss.backward()
        self.optimizer.step()

        # update target critic parameters
        self._target_update()
        return info

    def _target_update(self):
        """
        Softly update target critic parameters via Polyak averaging:
          θ_target ← τ·θ + (1−τ)·θ_target
        """
        tau = self.config['tau']
        # Iterate over parameters pairwise
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def sample_actions(self, observations, seed=None):
        """
        Sample actions from one-step flow policy.

        If seed is None: draw fresh Gaussian noise.
        If seed is not None: return the mean action (zero noise).
        """
        self.eval()
        with torch.no_grad():
            batch_size = observations.shape[0]
            if seed is not None:
                torch.manual_seed(seed)
            noises = torch.randn(batch_size, self.action_dim, device=self.device)
            actions = self.actor_onestep_flow(observations, noises)
            actions = torch.clamp(actions, -1.0, 1.0)
        
        return actions


    def compute_flow_actions(self, observations, noises):
        """
        Generate actions by running the BC flow network via the Euler method:

        a_0 = noises
        for i in range(flow_steps):
          t = i / flow_steps
          v = actor_bc_flow(observations, a_i, t)
          a_{i+1} = a_i + v / flow_steps
        return clamp(a_final, -1,1)
        """
        # Optionally encode observations
        obs_enc = observations

        actions = noises.clone()
        K = self.config['flow_steps']
        for i in range(K):
            # Current interpolation time t scalar in [0,1]
            t = torch.full((observations.size(0), 1), float(i) / K, device=self.device)
            # Compute velocity field from BC-flow network
            v = self.actor_bc_flow(obs_enc, actions, t)
            # Euler integration step
            actions = actions + v / K
            actions = torch.clamp(actions, -1.0, 1.0)
        # Ensure actions stay within bounds
        return torch.clamp(actions, -1.0, 1.0)
