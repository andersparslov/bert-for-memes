from transformers import DistilBertForSequenceClassification
import os
from pytorch_lightning import LightningModule
import torch
from torch import optim

from pathlib import Path
PROJECT_PATH = Path(__file__).resolve().parents[2]

class MemeModel(LightningModule):

    def __init__(self, parameters_dict, device_input, num_labels_input=2):
        super().__init__()

        self.save_hyperparameters()

        self.num_labels = self.hparams.num_labels_input

        self.save_every = parameters_dict['save_every']
        self.learning_rate = parameters_dict['lr']
        self.N_train = parameters_dict['N_train']
        self.N_test = parameters_dict['N_test']
        self.print_every = parameters_dict['print_every']
        self.epochs = parameters_dict['epochs']
        self.lr = parameters_dict['lr']
        self.epochs = parameters_dict['epochs']
        self.batch_size = parameters_dict['batch_size']

        # Add model folders if not present
        if "pretrained" not in os.listdir(str(PROJECT_PATH / "models")):
            os.mkdir(str(PROJECT_PATH / "models" / "pretrained"))

        if "finetuned" not in os.listdir(str(PROJECT_PATH / "models")):
            os.mkdir(str(PROJECT_PATH / "models" / "finetuned"))

        if "distilbert-base-uncased" in os.listdir(str(PROJECT_PATH / "models" / "pretrained")):
            model_path = str(PROJECT_PATH / "models" / "pretrained") + "/distilbert-base-uncased"
        else:
            model_path = "distilbert-base-uncased"
        mod_fn = DistilBertForSequenceClassification
        self.mod = mod_fn.from_pretrained(model_path, num_labels=self.num_labels)
        if not "distilbert-base-uncased" in os.listdir(str(PROJECT_PATH / "models" / "pretrained")):
            self.mod.save_pretrained(str(PROJECT_PATH / "models" / "pretrained") + "/distilbert-base-uncased")
        self.steps = 0

    def forward(self, input_ids, mask, labels):
        self.steps += 1
        if self.steps % self.save_every == 0:
            self.save()
        y_pred = self.mod(input_ids, mask, labels=labels)
        return y_pred.logits, y_pred.loss

    def save(self):
        self.mod.save_pretrained(f"models/finetuned/distilbert-base-uncased-{self.steps}")

    def load(self, steps):
        self.mod.from_pretrained(f"models/finetuned/distilbert-base-uncased-{steps}")

    def training_step(self, batch, batch_idx):
        text, labels = batch
        y_pred, loss = self(text['input_ids'].squeeze(),
                            text['attention_mask'].squeeze(),
                            labels)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        text, labels = batch

        y, loss = self(text['input_ids'].squeeze(),
                       text['attention_mask'].squeeze(),
                       labels)

        y = torch.argmax(y, dim=1).cpu()
        correct_count = torch.sum(y == labels.cpu())

        accuracy = correct_count / len(batch)
        self.log("val_accuracy", accuracy)

    def configure_optimizers(self):
        learning_rate = self.lr
        return optim.Adam(self.parameters(), lr=learning_rate)