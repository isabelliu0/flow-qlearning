import copy
import torch
import torch.nn as nn
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
        'q_agg': 'mean',              # specify how to aggregate ensemble Q-values: 'mean' or 'min'
        'alpha': 10.0,                # Weight for distillation loss in actor objective
        'flow_steps': 10,             #  # of  Euler integration steps for BC flow
        'normalize_q_loss': False,    # Whether to re-scale Q-loss term
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
        self.target_critic = copy.deepcopy(self.critic)

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
        """
        Compute the critic (Q-function) mean-squared TD error.

        Args:
            batch: dict with tensors:
                'observations'       (B x *ob_dims),
                'actions'            (B x action_dim),
                'next_observations'  (B x *ob_dims),
                'rewards'            (B x 1),
                'masks'              (B x 1)  (0 if episode ended, else 1)

        Returns:
            loss      : scalar tensor of MSE loss
            diagnostics: dict of logging values
        """
        # Sample next actions from current policy, then clip to [-1,1]
        next_actions = self.sample_actions(batch['next_observations'])
        next_actions = torch.clamp(next_actions, -1.0, 1.0)

        # Compute target Q-values: use target critic network
        # Shape of next_qs: (B x num_ensembles)
        next_qs = self.target_critic(batch['next_observations'], next_actions)
        if self.config['q_agg'] == 'min':
            # Take minimum across ensemble for conservative estimate
            next_q, _ = next_qs.min(dim=1, keepdim=True)
        else:
            # Take mean across ensemble
            next_q = next_qs.mean(dim=1, keepdim=True)

        # Build TD target r + γ * mask * next_q
        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        # Compute current Q-values from critic under current parameters
        # Shape of q: (B x 1) if critic returns single output per sample, else mean
        q_values = self.critic(batch['observations'], batch['actions'])
        # Mean-squared error loss
        loss = torch.mean((q_values - target_q).pow(2))

        # Logging diagnostics
        diagnostics = {
            'critic_loss': loss.item(),
            'q_mean': q_values.mean().item(),
            'q_max': q_values.max().item(),
            'q_min': q_values.min().item(),
        }
        return loss, diagnostics

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
        B, action_dim = batch['actions'].shape

        # 1) Behavioral cloning (BC) flow loss
        # Sample x0 ~ N(0,I)
        x0 = torch.randn_like(batch['actions'])
        # Expert actions x1
        x1 = batch['actions']
        # Uniform time t in [0,1]
        t = torch.rand(B, 1, device=self.device)
        # Interpolate: x_t = (1-t)x0 + t x1
        xt = (1 - t) * x0 + t * x1
        # True velocity v = x1 - x0
        v_true = x1 - x0
        # Predict velocity from BC flow network
        v_pred = self.actor_bc_flow(batch['observations'], xt, t)
        # MSE loss between predicted and true velocity
        bc_flow_loss = torch.mean((v_pred - v_true).pow(2))

        # 2) Distillation loss
        # Sample noises for one-step policy
        noises = torch.randn_like(batch['actions']).to(self.device)
        # Compute "ground truth" multi-step flow actions
        with torch.no_grad():
            target_flow_actions = self.compute_flow_actions(batch['observations'], noises)
        # One-step policy output
        actor_actions = self.actor_onestep_flow(batch['observations'], noises)
        # MSE distillation loss
        distill_loss = torch.mean((actor_actions - target_flow_actions).pow(2))

        # 3) Q-based loss to maximize value
        # Clip actions to valid range
        actor_actions_clipped = torch.clamp(actor_actions, -1.0, 1.0)
        # Query critic ensemble on these actions
        qs = self.critic(batch['observations'], actor_actions_clipped)
        # Aggregate across ensemble then average over batch
        q_val = qs.mean(dim=1, keepdim=True)
        # Negative because we minimize loss
        q_loss = -q_val.mean()
        # Optionally normalize Q-loss magnitude
        if self.config['normalize_q_loss']:
            factor = 1.0 / (torch.abs(q_val).mean().detach() + 1e-8)
            q_loss = factor * q_loss

        # Total actor loss
        loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        # Logging diagnostics
        mse = torch.mean((actor_actions - batch['actions']).pow(2))
        diagnostics = {
            'actor_loss': loss.item(),
            'bc_flow_loss': bc_flow_loss.item(),
            'distill_loss': distill_loss.item(),
            'q_loss': q_loss.item(),
            'q_mean': q_val.mean().item(),
            'mse': mse.item(),
        }
        return loss, diagnostics

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
        B = observations.shape[0]
        if seed is None:
            # stochastic sampling
            noises = torch.randn(B, self.action_dim, device=self.device)
        else:
            # deterministic: zero noise gives the “mean” action
            noises = torch.zeros(B, self.action_dim, device=self.device)

        actions = self.actor_onestep_flow(observations, noises)
        return torch.clamp(actions, -1.0, 1.0)


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
        # Ensure actions stay within bounds
        return torch.clamp(actions, -1.0, 1.0)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config=None):
        """
        Factory method to instantiate and initialize the agent.

        Args:
            seed           : random seed for reproducibility
            ex_observations: example observations tensor for shape inference
            ex_actions     : example actions tensor for shape inference
            config         : optional config dict. If None, uses get_config().

        Returns:
            agent: an initialized FQLAgent on the correct device
        """
        # Use default config if none provided
        cfg = config or get_config()
        # Infer dimensions from example tensors
        ob_dims = tuple(ex_observations.shape[1:])
        action_dim = ex_actions.shape[-1]
        # Instantiate agent
        agent = cls(cfg, ob_dims, action_dim, seed)
        # Move to correct device
        return agent.to(agent.device)
