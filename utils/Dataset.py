import numpy as np

class Dataset:
    """Dataset class for offline RL."""
    
    def __init__(self, data):
        """Initialize dataset from dictionary of arrays."""
        self.data = data
        # Validate data format
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        
        # Get dataset size
        self.size = len(next(iter(data.values())))
        
        # Verify all arrays have the same first dimension
        for k, v in data.items():
            if len(v) != self.size:
                raise ValueError(f"Array '{k}' has length {len(v)}, but expected {self.size}")
    
    def sample(self, batch_size):
        """Sample a batch of transitions randomly."""
        indices = np.random.randint(0, self.size, size=batch_size)
        batch = {k: np.array(v)[indices] for k, v in self.data.items()}
        return batch