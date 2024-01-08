from pytorch_lightning import Trainer

from config import Config, seed_everything

from model.cnn_wrapper import CNNWrapper
from util.dataloaders import get_dataloader
from util.parser import parse_train_args

if __name__ == '__main__':
    args = parse_train_args()
    config = Config(args)
    seed_everything(config.seed)
    train_loader = get_dataloader(config, phase='train')
    val_loader = get_dataloader(config, phase='val')

    model = CNNWrapper(config)
    trainer = Trainer(max_epochs=config.num_epochs)
    trainer.fit(model, train_loader, val_loader)

    # ecg = ECGDataset(pd.read_csv(config.train_csv_path))
    # ecg[0]
    """ 
    df = pd.read_csv(config.train_csv_path)
    print(df.shape)
    print(df.head())
    train_loader = get_dataloader(config, phase='train')
    print(len(train_loader))
    """
