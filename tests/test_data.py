# Test using 'pytest tests/'
# Or with coverage 'coverage run -m pytest tests/' use 'coverage report' after.
from tests import _INPUT_FILE_PATH, _N_TRAIN
import torch
import numpy as np
from src.models.dataset import Dataset
from transformers import DistilBertTokenizer

# Test training data
device = torch.device("cpu")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_set = Dataset(_INPUT_FILE_PATH,tokenizer=tokenizer,device=device)
item_dict, label = train_set[0]
assert isinstance(item_dict, dict)
assert "input_ids" in item_dict.keys(), "Tokenization error: Input_ids not in dict item"
assert "attention_mask" in item_dict.keys(), "Tokenization error: Masks not in dict item"
assert item_dict['input_ids'].shape == item_dict['attention_mask'].shape, "Input ids and masks not the same shape"
assert len(train_set) == _N_TRAIN, "Incorrect number of training examples"
assert isinstance(label.item(), int), "Label is not integer"
assert all(np.sort(train_set.classes()) == np.array([0, 1, 2, 3])), "Label error: Classes missing or not between 0 and 3."

