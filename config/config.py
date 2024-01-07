import random
import numpy as np
import torch


class Config:
    seed = 9
    csv_path = ''
    device = 'cuda:0' if torch.cuda.is_available() else "cpu"
    lstm_logs = 'lstm_logs/lstm.csv'
    train_csv_path = "data.csv"

    # Train variables
    num_workers = 4
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 100


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
