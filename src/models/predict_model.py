from Model import MemeModel
import torch


def predict_model(item):
    device = torch.device("cuda")
    model = MemeModel()
    # TO-DO : Checkpoint model 
    with torch.no_grad():
        y = item['labels'].to(device)
        x_doc = item['input_ids'].to(device)
        y_pred = model(x_doc.unsqueeze(0), labels=y)
        y_pred = torch.argmax(y_pred.logits, axis=1)
        
    return y_pred


if __name__ == "__main__":
    predict_model()