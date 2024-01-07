import torch
from pytorch_lightning import LightningModule

from model.cnn import CNN


class CNNWrapper(LightningModule):
    def __init__(
            self,
            input_size=1,
            hid_size=256,
            kernel_size=5,
            num_classes=5,
    ):
        super().__init__()
        self.model = CNN(input_size, hid_size, kernel_size, num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        out = self(data)
        loss = torch.nn.functional.cross_entropy(out, target)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        out = self(data)
        loss = torch.nn.functional.cross_entropy(out, target)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]



