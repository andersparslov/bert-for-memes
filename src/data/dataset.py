import torch
import pickle
import numpy as np
import torch
# Code from 

class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_filepath, tokenizer, device):
        with open(data_filepath, "rb") as file_handle:
            data = pickle.load(file_handle)
        self.device = device
        self.data  = data
        self.text = data['texts']
        self.labels = data['labels']
        self.texts = [tokenizer(str(text),
                                padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in self.text]

    def classes(self):
        return np.unique(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        text = self.texts[idx]
        label = self.labels[idx]
        text = {'input_ids':text['input_ids'].to(self.device),
                 'attention_mask':text['attention_mask'].to(self.device)}
        return text, label

