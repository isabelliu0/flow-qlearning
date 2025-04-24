from absl import app, flags
import ogbench
import tqdm
import os
from ml_collections import config_flags
from agents import agents
from utils.ReplayBuffer import ReplayBuffer
from utils.Dataset import Dataset
import numpy as np
import torch
from agents.fql import get_config

FLAGS = flags.FLAGS
flags.DEFINE_string('agent', "fql", 'Agent name')
flags.DEFINE_string('env', "cube-double-play-singletask-v0", 'Env name')
flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('batch_size', 256, 'Batch size for training.')
flags.DEFINE_integer('eval_interval', 10000, 'Number of steps between evaluations.')
flags.DEFINE_integer('eval_episodes', 10, 'Number of evaluation episodes.')
flags.DEFINE_integer('save_interval', 1000000, "Number of steps between model saves.")
flags.DEFINE_integer('seed', 42, "Random seed.")
flags.DEFINE_string('save_dir', 'results/', "Directory to save results.")
flags.DEFINE_float('alpha', 10.0, 'BC coefficient for FQL.')

config_flags.DEFINE_config_file(
    'config',
    'config.py',
    'Configuration file for the agent.',
    lock_config=False,
)

# See if CUDA is available
is_avail = torch.cuda.is_available()
if is_avail: 
    print("Cuda is available")
    device = "cuda"
else:
    print("Cuda is not available")
    device = "cpu"


def evaluate(agent, env, num_episodes=10):
    """Evaluate the agent."""
    episode_rewards = []
    episode_lengths = []

    for _ in range(num_episodes):
        obs, _, = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action = agent.sample_actions(obs_tensor)
                action = action.cpu().numpy().squeeze(0)
            
            # Step in the environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
            obs = next_obs
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    return {
        'eval/return_mean': np.mean(episode_rewards),
        'eval/return_std': np.std(episode_rewards),
        'eval/length_mean': np.mean(episode_lengths)
    }

def prepare_batch_for_agent(batch, device):
    """Convert batch data to tensors."""
    return {
        'observations': torch.FloatTensor(batch['observations']).to(device),
        'actions': torch.FloatTensor(batch['actions']).to(device),
        'rewards': torch.FloatTensor(batch['rewards']).to(device),
        'next_observations': torch.FloatTensor(batch['next_observations']).to(device),
        'masks': torch.FloatTensor(batch['masks']).to(device)
    }
 

def main(_):
    best_eval_return = 0
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    exp_name = f"{FLAGS.env}_{FLAGS.agent}_alpha{FLAGS.alpha}_seed{FLAGS.seed}"
    save_dir = os.path.join(FLAGS.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # GETTING DATASET
    env, train_dataset, val_dataset_dict = ogbench.make_env_and_datasets(FLAGS.env)
    eval_env = ogbench.make_env_and_datasets(FLAGS.env, env_only=True)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # DEFINING AGENT
    config = FLAGS.config 
    config = get_config()
    agent = agents[FLAGS.agent](ob_dim=state_dim, action_dim=action_dim,config=config,device=device)

    # TRAINING
    train_dataset = Dataset(train_dataset)
    for step in tqdm.tqdm(range(1, FLAGS.offline_steps + 1)): 
        # Step 1: Sample Data
        batch = train_dataset.sample(FLAGS.batch_size)
        batch = prepare_batch_for_agent(batch,device)
        
        # Step 2: Take a training step
        update_info = agent.update(batch)

        if step % 1000 == 0:
            print(f"Step: {step}, " +
                  f"Critic Loss: {update_info['critic/critic_loss']:.4f}, " +
                  f"Actor Loss: {update_info['actor/actor_loss']:.4f}, " +
                  f"Q Mean: {update_info['critic/q_mean']:.4f}")
        
        # Evaluate agent
        if step % FLAGS.eval_interval == 0:
            eval_metrics = evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes)

            print(f"Evaluation at step {step}: " +
                  f"Mean Return: {eval_metrics['eval/return_mean']:.2f} ± {eval_metrics['eval/return_std']:.2f}")
            
            if eval_metrics['eval/return_mean'] > best_eval_return:
                best_eval_return = eval_metrics['eval/return_mean']
                torch.save(agent.state_dict(), os.path.join(save_dir, "best_model.pt"))
        
        if step % FLAGS.save_interval == 0:
            torch.save(agent.state_dict(), os.path.join(save_dir, f"model_step_{step}.pt"))
    
    # Final evaluation and save
    eval_metrics = evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes)
    print(f"Final Evaluation: " +
          f"Mean Return: {eval_metrics['eval/return_mean']:.2f} ± {eval_metrics['eval/return_std']:.2f}")
    torch.save(agent.state_dict(), os.path.join(save_dir, "final_model.pt"))
        
        
if __name__ == '__main__':
    app.run(main)

# This script uses the absl library to handle command-line arguments.
