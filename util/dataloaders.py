import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import Config
from dataset.dataset import ECGDataset


def get_dataloader(config: Config, phase: str):
    df = pd.read_csv(config.train_csv_path)
    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=config.seed, stratify=df['label']
    )
    train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)
    df = train_df if phase == 'train' else val_df
    dataset = ECGDataset(df)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        persistent_workers=True
    )
    return dataloader
