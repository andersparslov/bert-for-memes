import torch
import pickle

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_filepath, tokenizer):
        # Load data
        with open(data_filepath, "rb") as file_handle:
            data = pickle.load(file_handle)

        self.texts = data['texts']
        self.labels = data['labels']

        self.encodings = tokenizer(self.texts, truncation=True, padding=True)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        'Generates one sample of data'
        # Wrap everything into a dictionary
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        remove_key = item.pop("token_type_id", None)
        return item