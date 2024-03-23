from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import numpy as np
from models.gru import GruNet
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from data.custom_dataset import Wave_Data


device = "cuda" if torch.cuda.is_available() else "cpu"


model = GruNet(input_size=1, hidden_size=16, output_size=1).to(device)


def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train_one_epoch(train_loader, model):
    pass

def test_one_epoch(test_loader, model, loss_fn):
    pass

# def main():
    


#     pass

# if __name__ == "__main__":
#     main()




