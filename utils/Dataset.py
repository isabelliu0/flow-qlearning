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
        indices = np.random.choice(len(self.data), size=batch_size)
        return [self.data[i] for i in indices]