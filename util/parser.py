import sys
from argparse import ArgumentParser


def parse_train_args(args=sys.argv[1:]):
    parser = ArgumentParser()

    # Seed
    parser.add_argument('--seed', type=int, default=9, help='The seed for the process.')

    # Logging
    parser.add_argument('--lightning_logdir', type=str, default='logs/lightning_logs',
                        help='Path to the log directory.')
    parser.add_argument('--run_name', type=str, default='default', help='The name of the experiment')

    # Training
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default=None)

    # Dataset
    parser.add_argument('--data_dir', type=str, default='data/', help='Folder containing original structures')
    parser.add_argument('--train_data_path', type=str, default='data/mitbih_with_syntetic_train.csv',
                        help='Path to the indices used for training')
    parser.add_argument('--test_data_path', type=str, default='data/mitbih_with_syntetic_test.csv', help='')

    args = parser.parse_args(args)
    return args
