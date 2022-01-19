'''
Local Deployment with TorchServe
'''

from variable import PROJECT_PATH
from transformers import DistilBertTokenizer
import collections
import torch
import torchvision
from src.models.Model_red_boilerplate import MemeModel
import hydra
from src.models.load_checkpoint import load_checkpoint
#from detectron2.export.flatten import TracingAdapter
from src.data.dataset import *


@hydra.main(config_path=PROJECT_PATH / "configs", config_name="/config.yaml")
def local_deployment(cfg):
    input_filepath_model = str(PROJECT_PATH / "models" / "finetuned")
    input_filepath_data = str(PROJECT_PATH / "data" / "processed")
    output_filepath_model = str(PROJECT_PATH / "models")

    # Loading the model
    model = load_checkpoint(input_filepath_model + '/checkpoint.pth')
    model.eval()

    # Loading a data instance
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    dataset = Dataset(input_filepath_data,tokenizer=tokenizer,device='cpu')

    N_train = cfg.hyperparameters.N_train
    N_test = cfg.hyperparameters.N_test
    batch_size = cfg.hyperparameters.batch_size

    train_set, val_set = torch.utils.data.random_split(dataset, [N_train, N_test])

    valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

    text, labels = next(iter(valloader))

    dummy_input = [text['input_ids'].squeeze(), text['attention_mask'].squeeze(), labels]

    script_model = torch.jit.trace(model, dummy_input)

    script_model.save(output_filepath_model + "/local_deployment/" + 'deployable_model.pt')

    y_pred_model, _ = model(text['input_ids'].squeeze(), text['attention_mask'].squeeze(), labels)

    y_pred_script_model, _ = script_model(text['input_ids'].squeeze(), text['attention_mask'].squeeze(), labels)

    top5_indices_model = torch.topk(y_pred_model, 4, dim=1)[1]
    top5_indices_script_model = torch.topk(y_pred_model, 4, dim=1)[1]

    assert torch.equal(top5_indices_model, top5_indices_script_model)


if __name__ == "__main__":
    local_deployment()
