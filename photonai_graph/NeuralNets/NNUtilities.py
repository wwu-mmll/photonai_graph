try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    pass


class CustomDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, graphs, y):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.graphs = graphs
        self.labels = y

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.graphs[idx], self.labels[idx]


class DGLData(Dataset):
    """Dataset Class for handling dgl graphs with labels."""

    def __init__(self, samples):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.samples[idx]


def zip_data(X, y):
    # returns a zipped list from two input graphs
    data = tuple(zip(X, y))

    return data
