import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from config import Config, seed_everything
from util.dataloaders import get_dataloader

if __name__ == '__main__':
    id_to_label = {
        0: "Normal",
        1: "Artial Premature",
        2: "Premature ventricular contraction",
        3: "Fusion of ventricular and normal",
        4: "Fusion of paced and normal"
    }

    config = Config()
    seed_everything(config.seed)
    train_dl = get_dataloader(config, "train")
    item = next(iter(train_dl))
    print(item)
