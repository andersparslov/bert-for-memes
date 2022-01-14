from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
import torch.nn as nn

# TODO: remake the model so it does not take a config file but the direct parameters (save_every, etc.) instead
class MemeModel(nn.Module):
    
    def __init__(self, config, device, num_labels=2):
        super().__init__()
        #self.save_every = config.save_every
        self.device = device
        if "distilbert-base-uncased" in os.listdir("models/pretrained"):
            model_path = "models/pretrained/distilbert-base-uncased"
        else:
            model_path = "distilbert-base-uncased"
        mod_fn = DistilBertForSequenceClassification 
        self.mod = mod_fn.from_pretrained(model_path, num_labels=num_labels).to(device)
        if not "distilbert-base-uncased" in os.listdir("models/pretrained"):
            self.mod.save_pretrained("models/pretrained/distilbert-base-uncased")
        self.steps = 0
        #self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)    

    def forward(self, input_ids, mask, labels):
        self.steps += 1
        #if self.steps % self.save_every == 0:
        #    self.save()
        y = labels
        y_pred = self.mod(input_ids, mask, labels=y)
        return y_pred.logits, y_pred.loss

    def save(self):
        self.mod.save_pretrained(f"models/finetuned/distilbert-base-uncased-{self.steps}")

    def load(self):
        self.mod.from_pretrained(f"models/finetuned/distilbert-base-uncased-{self.steps}")