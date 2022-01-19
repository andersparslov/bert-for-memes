from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
import torch.nn as nn
from variable import PROJECT_PATH
from pytorch_lightning import LightningModule
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger

class MemeModel(LightningModule):

    def __init__(self, parameters_dict, device_input, num_labels_input=2):
        super().__init__()

        self.save_hyperparameters()
        # device = self.hparams.device_input
        #self.dev = self.hparams.device_input
        self.num_labels = self.hparams.num_labels_input
        '''
         'N_test': cfg.hyperparameters.N_test,
                       'print_every': cfg.hyperparameters.print_every,
                       'epochs': cfg.hyperparameters.epochs,
                       'lr': cfg.hyperparameters.lr,
                       'epochs': cfg.hyperparameters.epochs,
                       'batch_size': cfg.hyperparameters.batch_size
        '''
        self.learning_rate = parameters_dict['lr']
        self.N_train = parameters_dict['N_train']
        self.N_test = parameters_dict['N_test']
        self.print_every = parameters_dict['print_every']
        self.epochs = parameters_dict['epochs']
        self.lr = parameters_dict['lr']
        self.epochs = parameters_dict['epochs']
        self.batch_size = parameters_dict['batch_size']

        # Add model folders if not presetn
        if "pretrained" not in os.listdir(str(PROJECT_PATH / "models")):
            os.mkdir(str(PROJECT_PATH / "models" / "pretrained"))
            # os.mkdir("models/pretrained")

        if "finetuned" not in os.listdir(str(PROJECT_PATH / "models")):
            os.mkdir(str(PROJECT_PATH / "models" / "finetuned"))
            # os.mkdir("models/finetuned")

        if "distilbert-base-uncased" in os.listdir(str(PROJECT_PATH / "models" / "pretrained")):
            model_path = str(PROJECT_PATH / "models" / "pretrained") + "/distilbert-base-uncased"
            # model_path = "models/pretrained/distilbert-base-uncased"
        else:
            model_path = "distilbert-base-uncased"
        mod_fn = DistilBertForSequenceClassification
        self.mod = mod_fn.from_pretrained(model_path, num_labels=self.num_labels)
        if not "distilbert-base-uncased" in os.listdir(str(PROJECT_PATH / "models" / "pretrained")):
            self.mod.save_pretrained(str(PROJECT_PATH / "models" / "pretrained") + "/distilbert-base-uncased")
            # self.mod.save_pretrained("models/pretrained/distilbert-base-uncased")
        self.steps = 0
        # self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)

    def forward(self, input_ids, mask, labels):
        self.steps += 1
        # if self.steps % self.save_every == 0:
        #    self.save()
        y = labels
        y_pred = self.mod(input_ids, mask, labels=y)
        return y_pred.logits, y_pred.loss

    def save(self):
        self.mod.save_pretrained(str(PROJECT_PATH) + f"models/finetuned/distilbert-base-uncased-{self.steps}")

    def load(self):
        self.mod.from_pretrained(str(PROJECT_PATH) + f"models/finetuned/distilbert-base-uncased-{self.steps}")

    def training_step(self, batch, batch_idx):
        text, labels = batch
        y_pred, loss = self(text['input_ids'].squeeze(),
                            text['attention_mask'].squeeze(),
                            labels)

        y_pred = torch.argmax(y_pred, dim=1).cpu()
        correct_count = torch.sum(y_pred == labels.cpu())

        accuracy = correct_count / len(batch)
        #self.log("loss", loss)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        text, labels = batch

        y, loss = self(text['input_ids'].squeeze(),
                       text['attention_mask'].squeeze(),
                       labels)

        y = torch.argmax(y, dim=1).cpu()
        correct_count = torch.sum(y == labels.cpu())

        accuracy = correct_count / len(batch)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', accuracy, prog_bar=True)

    def configure_optimizers(self):
        learning_rate = self.lr
        return optim.Adam(self.parameters(), lr=learning_rate)