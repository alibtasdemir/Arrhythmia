import io

import pandas as pd
import torch
import torchvision.transforms
from PIL import Image
from matplotlib import pyplot as plt
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import confusion_matrix, Accuracy

import seaborn as sns
from model.cnn import CNN


class CNNWrapper(LightningModule):
    def __init__(
            self,
            config,
    ):
        super().__init__()

        self.input_size = config.input_size
        self.hid_size = config.hid_size
        self.kernel_size = config.kernel_size
        self.num_classes = config.num_classes
        self.lr = config.learning_rate

        self.save_hyperparameters()

        self.model = CNN(self.input_size, self.hid_size, self.kernel_size, self.num_classes)
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.valid_acc = Accuracy(task="multiclass", num_classes=self.num_classes)

    def forward(self, x):
        return self.model(x)

    def calculate_metrics(self, logits, targets):
        preds = torch.argmax(logits, dim=1)
        self._log_cf(preds, targets)
        preds = preds.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average="macro")
        precision = precision_score(targets, preds, average="macro", zero_division=1)
        recall = recall_score(targets, preds, average="macro", zero_division=1)

        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }

    def _log_cf(self, outputs, labels):
        tb = self.logger.experiment
        confusion = ConfusionMatrix(num_classes=self.num_classes, task="multiclass").to(outputs.get_device())
        confusion(outputs, labels)
        computed_confusion = confusion.compute().detach().cpu().numpy().astype(int)

        df_cm = pd.DataFrame(
            computed_confusion
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.subplots_adjust(left=0.05, right=.65)
        sns.set(font_scale=1.2)
        sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt="d", ax=ax)
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg', bbox_inches='tight')
        buf.seek(0)
        im = Image.open(buf)
        im = torchvision.transforms.ToTensor()(im)
        tb.add_image("val_confusion_matrix", im, global_step=self.current_epoch)

    def _create_report(self, metrics):
        for key, value in metrics.items():
            self.log(f"val_{key}", value)

    def training_step(self, batch, batch_idx):
        data, target = batch
        logits = self(data)
        loss = torch.nn.functional.cross_entropy(logits, target)
        batch_value = self.train_acc(logits, target)
        self.log('train_acc_step', batch_value, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        data, target = batch
        logits = self(data)
        loss = torch.nn.functional.cross_entropy(logits, target)

        metrics = self.calculate_metrics(logits, target)
        self._create_report(metrics)
        self.valid_acc.update(logits, target)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.log('valid_acc_epoch', self.valid_acc.compute(), prog_bar=True)
        self.valid_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
