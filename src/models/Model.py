from transformers import DistilBertForSequenceClassification
import os
import torch.nn as nn

from pathlib import Path
PROJECT_PATH = Path(__file__).resolve().parents[2]

# TODO: remake the model so it does not take a config file but the direct parameters (save_every, etc.) instead
class MemeModel(nn.Module):
    
    def __init__(self, save_every, device, num_labels=2):
        super().__init__()
        self.save_every = save_every
        self.device = device

        # Add model folders if not present
        if "pretrained" not in os.listdir(str(PROJECT_PATH / "models" )):
            os.mkdir(str(PROJECT_PATH / "models" / "pretrained"))

        if "finetuned" not in os.listdir(str(PROJECT_PATH / "models" )):
            os.mkdir(str(PROJECT_PATH / "models" / "finetuned"))

        if "distilbert-base-uncased" in os.listdir(str(PROJECT_PATH / "models"/ "pretrained" )):
            model_path = str(PROJECT_PATH / "models" / "pretrained" ) + "/distilbert-base-uncased"
        else:
            model_path = "distilbert-base-uncased"
        mod_fn = DistilBertForSequenceClassification 
        self.mod = mod_fn.from_pretrained(model_path, num_labels=num_labels).to(device)
        if not "distilbert-base-uncased" in os.listdir(str(PROJECT_PATH / "models"/ "pretrained" )):
            self.mod.save_pretrained(str(PROJECT_PATH / "models" / "pretrained" ) + "/distilbert-base-uncased")
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