import random
from argparse import Namespace

import numpy as np
import torch


class Config:
    def __init__(self, args: Namespace):
        self.run = args.run_name
        self.seed = args.seed
        self.csv_path = ''
        if args.device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else "cpu"
        else:
            self.device = args.device

        self.train_csv_path = args.train_data_path
        self.test_csv_path = args.test_data_path
        self.logdir = args.lightning_logdir

        # Train variables
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.learning_rate = args.lr
        self.num_epochs = args.epochs

        # Model variables
        self.input_size = args.input_size
        self.hid_size = args.hid_size
        self.kernel_size = args.kernel_size
        self.num_classes = args.num_classes


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
