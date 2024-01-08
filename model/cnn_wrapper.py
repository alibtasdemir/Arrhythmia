import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from sklearn.metrics import accuracy_score

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
        self.save_hyperparameters()
        self.model = CNN(input_size, hid_size, kernel_size, num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        logits = self(data)
        loss = torch.nn.functional.cross_entropy(logits, target)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(preds.detach().cpu().numpy(), target.detach().cpu().numpy())
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
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



