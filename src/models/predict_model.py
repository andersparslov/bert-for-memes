from Model import MemeModel
from transformers import DistilBertTokenizer
import torch
load_steps = 1000


def predict_model(text):
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    item = tokenizer(text, padding='max_length', 
              max_length=512, truncation=True,
              return_tensors="pt")
    
    device = torch.device("cpu")
    model = MemeModel(1, device=device, num_labels=4)    
    model.load(load_steps)
    with torch.no_grad():
        y_pred, _ = model(item['input_ids'], 
                       item['attention_mask'], 
                       labels=None)
        y_pred = torch.argmax(y_pred, axis=1)
        
    return y_pred


if __name__ == "__main__":
    predict_model()