from absl import app, flags
import ogbench
import tqdm
import os
import json
from ml_collections import config_flags
from agents import agents
from utils.ReplayBuffer import ReplayBuffer
from utils.Dataset import Dataset
import numpy as np
import torch
from agents.fql import get_config, FQLAgent

FLAGS = flags.FLAGS
flags.DEFINE_string('agent', "fql", 'Agent name')
flags.DEFINE_string('env', "cube-double-play-singletask-v0", 'Env name')
flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('batch_size', 256, 'Batch size for training.')
flags.DEFINE_integer('eval_interval', 1000, 'Number of steps between evaluations.')
flags.DEFINE_integer('eval_episodes', 10, 'Number of evaluation episodes.')
flags.DEFINE_integer('save_interval', 1000000, "Number of steps between model saves.")
flags.DEFINE_integer('seed', 42, "Random seed.")
flags.DEFINE_string('save_dir', 'results/', "Directory to save results.")
flags.DEFINE_float('alpha', 10.0, 'BC coefficient for FQL.')
flags.DEFINE_string('device', "auto", "Device to run on.")

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

def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate(agent, env, num_episodes=10):
    """Evaluate the agent."""
    agent.eval()
    episode_rewards = []
    episode_lengths = []
    max_episode_steps = env.spec.max_episode_steps

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_success = []
        episode_length = 0
        steps = 0

        while not done and steps < max_episode_steps:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action = agent.sample_actions(obs_tensor, seed=0, temperature=0.0)
                action = action.cpu().numpy().squeeze(0)
            
            # Step in the environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
            obs = next_obs
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        if 'success' in info:
            episode_success.append(float(info['success']))
        elif 'episode' in info and 'success' in info['episode']:
            episode_success.append(float(info['episode']['success']))
    
    metrics = {
        'eval/return_mean': np.mean(episode_rewards),
        'eval/return_std': np.std(episode_rewards),
        'eval/length_mean': np.mean(episode_lengths)
    }
    if episode_success:
        metrics['eval/success_rate'] = np.mean(episode_success) * 100.0
    
    return metrics

def prepare_batch(batch, device):
    """Convert batch to tensors on specified device."""
    tensor_batch = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            tensor_batch[k] = torch.FloatTensor(v).to(device)
        else:
            tensor_batch[k] = v
    return tensor_batch

def check_environment(env):
    """
    Check the environment interface and baseline performance.
    
    Args:
        env: The environment to check
        
    Returns:
        baseline_reward: The reward achieved by a random policy
    """
    # Print environment information
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Maximum episode steps: {env.spec.max_episode_steps}")
    
    # Try a random policy to verify baseline performance
    random_rewards = []
    for _ in range(5):  # Run 5 episodes with random actions
        obs, _ = env.reset()
        total_reward = 0
        step = 0
        done = False
        while not done and step < env.spec.max_episode_steps:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            obs = next_obs
            step += 1
        random_rewards.append(total_reward)
        print(f"Random episode: Reward = {total_reward}, Steps = {step}")
        if 'success' in info:
            print(f"  Success: {info['success']}")
        elif 'episode' in info and 'success' in info['episode']:
            print(f"  Success: {info['episode']['success']}")
    
    baseline_reward = sum(random_rewards) / len(random_rewards)
    print(f"Random policy average reward: {baseline_reward}")
    
    return baseline_reward


def main(_):
    if FLAGS.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = FLAGS.device
    print(f"Using device: {device}")

    set_seed(FLAGS.seed)

    exp_name = f"{FLAGS.env}_{FLAGS.agent}_alpha{FLAGS.alpha}_seed{FLAGS.seed}"
    save_dir = os.path.join(FLAGS.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save configuration
    config = {
        'env': FLAGS.env,
        'offline_steps': FLAGS.offline_steps,
        'batch_size': FLAGS.batch_size,
        'eval_interval': FLAGS.eval_interval,
        'eval_episodes': FLAGS.eval_episodes,
        'save_interval': FLAGS.save_interval,
        'seed': FLAGS.seed,
        'alpha': FLAGS.alpha,
        'device': device,
    }
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # GETTING DATASET
    env, train_dataset, val_dataset_dict = ogbench.make_env_and_datasets(FLAGS.env)
    eval_env = ogbench.make_env_and_datasets(FLAGS.env, env_only=True)

    train_dataset = Dataset(train_dataset)

    observation = env.reset()[0]
    ob_dim = observation.shape[0]
    action_dim = env.action_space.shape[0]

    print("=== Checking Environment Interface ===")
    baseline_reward = check_environment(env)
    print(f"Baseline random policy reward: {baseline_reward}")
    print("=====================================")

    agent_config = get_config()
    agent_config['alpha'] = FLAGS.alpha
    agent = FQLAgent(
        ob_dim=ob_dim,
        action_dim=action_dim,
        config=agent_config,
        device=device
    )

    # Create loggers
    train_log_file = open(os.path.join(save_dir, 'train_log.csv'), 'w')
    train_log_file.write('step,critic_loss,actor_loss,q_mean,bc_flow_loss,distill_loss,q_loss\n')
    
    eval_log_file = open(os.path.join(save_dir, 'eval_log.csv'), 'w')
    eval_log_file.write('step,return_mean,return_std,success_rate\n')

    # TRAINING
    best_eval_return = float('-inf')
    for step in tqdm.tqdm(range(1, FLAGS.offline_steps + 1)): 
        # Step 1: Sample Data
        batch = train_dataset.sample(FLAGS.batch_size)
        batch = prepare_batch(batch, device)
        
        # Step 2: Take a training step
        update_info = agent.update(batch)

        if step % 1000 == 0:
            critic_loss = update_info['critic/critic_loss']
            actor_loss = update_info['actor/actor_loss']
            q_mean = update_info['critic/q_mean']
            bc_flow_loss = update_info['actor/bc_flow_loss']
            distill_loss = update_info['actor/distill_loss']
            q_loss = update_info['actor/q_loss']
            
            print(f"Step: {step}, " +
                  f"Critic Loss: {critic_loss:.4f}, " +
                  f"Actor Loss: {actor_loss:.4f}, " +
                  f"Q Mean: {q_mean:.4f}, " +
                  f"BC Flow Loss: {bc_flow_loss:.4f}, " +
                  f"Distill Loss: {distill_loss:.4f}, " +
                  f"Q Loss: {q_loss:.4f}")
                  
            train_log_file.write(f"{step},{critic_loss},{actor_loss},{q_mean},{bc_flow_loss},{distill_loss},{q_loss}\n")
            train_log_file.flush()
        
        # Evaluate agent
        if step % FLAGS.eval_interval == 0:
            eval_metrics = evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes)

            print(f"Evaluation at step {step}: " +
                  f"Mean Return: {eval_metrics['eval/return_mean']:.2f} ± " +
                  f"{eval_metrics['eval/return_std']:.2f}")
            
            if 'eval/success_rate' in eval_metrics:
                print(f"Success Rate: {eval_metrics['eval/success_rate']:.2f}%")
                
            # Log evaluation metrics
            eval_log = f"{step},{eval_metrics['eval/return_mean']},{eval_metrics['eval/return_std']}"
            if 'eval/success_rate' in eval_metrics:
                eval_log += f",{eval_metrics['eval/success_rate']}"
            else:
                eval_log += ",0.0"  # Default value if success rate not available
            eval_log_file.write(eval_log + "\n")
            eval_log_file.flush()
            
            # Save best model
            if eval_metrics['eval/return_mean'] > best_eval_return:
                best_eval_return = eval_metrics['eval/return_mean']
                torch.save(agent.state_dict(), os.path.join(save_dir, "best_model.pt"))
                print(f"New best model saved with return {best_eval_return:.2f}")
        
        # Save checkpoint
        if step % FLAGS.save_interval == 0:
            torch.save(agent.state_dict(), os.path.join(save_dir, f"model_step_{step}.pt"))
    
    # Final evaluation and save
    eval_metrics = evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes)
    print(f"Final Evaluation: " +
          f"Mean Return: {eval_metrics['eval/return_mean']:.2f} ± {eval_metrics['eval/return_std']:.2f}")
    if 'eval/success_rate' in eval_metrics:
        print(f"Success Rate: {eval_metrics['eval/success_rate']:.2f}%")
    
    torch.save(agent.state_dict(), os.path.join(save_dir, "final_model.pt"))
    print(f"Final model saved to {os.path.join(save_dir, 'final_model.pt')}")

    train_log_file.close()
    eval_log_file.close()
        
        
if __name__ == '__main__':
    app.run(main)

# This script uses the absl library to handle command-line arguments.
