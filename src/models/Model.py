from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
import torch.nn as nn


class MemeModel(nn.Module):
    
    def __init__(self, device, num_labels=2):
        super().__init__()
        self.device = device
        if "distilbert-base-uncased" in os.listdir("models"):
            model_path = "models\\distilbert-base-uncased"
        else:
            model_path = "distilbert-base-uncased"
        mod_fn = DistilBertForSequenceClassification 
        self.mod = mod_fn.from_pretrained(model_path, num_labels=num_labels).to(device)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)    

    def forward(self, item):
        y = item['labels'].to(self.device)
        x_doc = item['input_ids'].to(self.device)
        y_pred = self.mod(x_doc, labels=y)
        return y_pred.logits, y_pred.loss