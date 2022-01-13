# Test using 'pytest tests/'
# Or with coverage 'coverage run -m pytest tests/' use 'coverage report' after.
from src.models.Model import MemeModel
from src.data.dataset import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tests import _INPUT_FILE_PATH
import torch
import os


device = torch.device("cpu")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_set = Dataset(_INPUT_FILE_PATH,tokenizer=tokenizer,device=device)
model = MemeModel(None, device, num_labels=4)

assert isinstance(model.mod, DistilBertForSequenceClassification), "Model is not DistilBertForSequenceClassification object"

item_dict, label = train_set[0]
y, loss = model(item_dict['input_ids'], item_dict['attention_mask'], labels=None)

assert y.shape == torch.Size([1, 4]), "Model output has wrong shape"