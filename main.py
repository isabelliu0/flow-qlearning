from absl import app, flags
import ogbench
import tqdm
from ml_collections import config_flags
from agents import agents
from utils.ReplayBuffer import ReplayBuffer
from utils.Dataset import Dataset
import numpy as np


FLAGS = flags.FLAGS
flags.DEFINE_string('agent', "brac", 'Agent name')
flags.DEFINE_string('env', "cube-double-play-singletask-v0", 'Env name')
flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('batch_size', 256, 'Batch size for training.')

config_flags.DEFINE_config_file(
    'config',
    'config.py',
    'Configuration file for the agent.',
    lock_config=False,
)

def main(_):

    # GETTING DATASET
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(FLAGS.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # DEFINING AGENT
    config = FLAGS.config
    agent = agents[FLAGS.agent](state_dim=state_dim, action_dim=action_dim)

    # TRAINING
    train_dataset = Dataset(train_dataset)
    for i in (range(FLAGS.offline_steps)): 
        # Step 1: Sample Data
        batch = train_dataset.sample(FLAGS.batch_size)

        # Step 2: Train Agent
        agent.train(batch, FLAGS.batch_size)



if __name__ == '__main__':
    app.run(main)

# This script uses the absl library to handle command-line arguments.