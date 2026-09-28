from scipy.io import loadmat
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from pathlib import Path
import yaml

def load_yaml(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Config not found: {p}')
    with p.open("r",encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_data(data_path):

    dataset = loadmat(data_path)
    faces = dataset["ff"]
    faces = torch.tensor(faces, dtype=torch.float32).T
    faces = faces /255.0 #transfrom to [0,1] pixel vals
    dataset = TensorDataset(faces)
    return dataset


def parse_data(train_size, dataset, batch_size):       

    train_size = int(train_size*len(dataset))
    test_size = len(dataset)- train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=False)

    return train_loader, test_loader


# plots..