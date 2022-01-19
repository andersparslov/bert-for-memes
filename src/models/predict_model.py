from Model import MemeModel
from transformers import DistilBertTokenizer
import torch

load_steps = 1000
from variable import PROJECT_PATH
from transformers import DistilBertTokenizer
import hydra
from src.models.load_checkpoint import load_checkpoint
# from detectron2.export.flatten import TracingAdapter
from src.data.dataset import *


@hydra.main(config_path=PROJECT_PATH / "configs", config_name="/config.yaml")

def predict_model(cfg):
    input_filepath_model = str(PROJECT_PATH / "models" / "finetuned")
    input_filepath_data = str(PROJECT_PATH / "data" / "processed")
    output_filepath_model = str(PROJECT_PATH / "models")

    # Loading the model
    model = load_checkpoint(input_filepath_model + '/checkpoint.pth')
    model.eval()

    # Loading a data instance
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    dataset = Dataset(input_filepath_data, tokenizer=tokenizer, device='cpu')

    N_train = cfg.hyperparameters.N_train
    N_test = cfg.hyperparameters.N_test
    batch_size = cfg.hyperparameters.batch_size

    train_set, val_set = torch.utils.data.random_split(dataset, [N_train, N_test])

    valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

    text, labels = next(iter(valloader))

    y_pred, _ = model(text['input_ids'].squeeze(),
                      text['attention_mask'].squeeze(),
                      labels)

    y_pred = torch.argmax(y_pred, axis=1)

    return y_pred


if __name__ == "__main__":
    predict_model()
