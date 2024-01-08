import random
from argparse import Namespace

import numpy as np
import torch


"""
    # Dataset
    parser.add_argument('--data_dir', type=str, default='data/', help='Folder containing original structures')
    parser.add_argument('--train_data_path', type=str, default='data/mitbih_with_syntetic_train.csv',
                        help='Path to the indices used for training')
    parser.add_argument('--val_data_path', type=str, default='data/mitbih_with_syntetic_test.csv', help='')
"""

class Config:
    def __init__(self, args: Namespace):
        self.run = args.run_name
        self.seed = args.seed
        # seed = 9
        self.csv_path = ''
        # csv_path = ''
        if args.device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else "cpu"
        else:
            self.device = args.device
        # device = 'cuda:0' if torch.cuda.is_available() else "cpu"
        # lstm_logs = 'lstm_logs/lstm.csv'

        self.train_csv_path = args.train_data_path
        # train_csv_path = "data/mitbih_with_syntetic_train.csv"
        self.test_csv_path = args.test_data_path
        # test_csv_path = "data/mitbih_with_syntetic_test.csv"
        self.logdir = args.lightning_logdir

        # Train variables
        self.num_workers = args.num_workers
        # num_workers = 4
        self.batch_size = args.batch_size
        # batch_size = 128
        self.learning_rate = args.lr
        # learning_rate = 0.001
        self.num_epochs = args.epochs
        # num_epochs = 2


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
