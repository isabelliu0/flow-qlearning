from ml_collections import ConfigDict

def get_config():
    config = ConfigDict(
        dict(
            env="cube-double-play-singletask-v0",
            offline_steps=1000000,
            batch_size=64,
            dataset_path="path/to/dataset",
            model_path="path/to/model",
            log_dir="path/to/logs",
            checkpoint_dir="path/to/checkpoints",
            save_interval=10000,
            eval_interval=5000,
            eval_episodes=10,
            eval_batch_size=256,
        )
    )
    
    return config
