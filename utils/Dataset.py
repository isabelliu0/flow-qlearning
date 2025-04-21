import numpy as np

class Dataset:
    def __init__(self, data):
        self.data = data
        

    def get_data(self):
        return self.data

    def set_data(self, new_data):
        self.data = new_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    def sample(self, batch_size):
        n = len(next(iter(self.data.values())))
        indices = np.random.choice(n, batch_size, replace=False)
        
        batch = {key: np.array(value)[indices] for key, value in self.data.items()}
        return batch