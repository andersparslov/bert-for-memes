# -*- coding: utf-8 -*-
import logging
from dotenv import find_dotenv, load_dotenv
from transformers import DistilBertTokenizer
from Model_red_boilerplate import MemeModel
import gc
import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from src.data.dataset import *
from pathlib import Path
import wandb


PROJECT_PATH = Path(__file__).resolve().parents[2]

# Note: Hydra is incompatible with @click
@hydra.main(config_path= PROJECT_PATH / "configs",config_name="/config.yaml")
def main(cfg):

    # for hydra parameters
    input_filepath_data = str(PROJECT_PATH / "data" / "processed")
    output_filepath_model = str(PROJECT_PATH / "models")

    gc.collect()

    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Is CUDA available ? ', torch.cuda.is_available())

    print(cfg.hyperparameters)

    num_labels = cfg.hyperparameters.num_labels
    N_train = cfg.hyperparameters.N_train
    N_test = cfg.hyperparameters.N_test
    wandb_username = cfg.hyperparameters.wandb_username
    #if wandb_username is not None:
    #    wandb.init(project="wandb-lightning", entity=wandb_username)

    batch_size = cfg.hyperparameters.batch_size

    parameters_dict = {'save_every': cfg.hyperparameters.save_every,
                       'N_train': cfg.hyperparameters.N_train,
                       'N_test': cfg.hyperparameters.N_test,
                       'print_every': cfg.hyperparameters.print_every,
                       'epochs': cfg.hyperparameters.epochs,
                       'lr': cfg.hyperparameters.lr,
                       'epochs': cfg.hyperparameters.epochs,
                       'batch_size': cfg.hyperparameters.batch_size
                       }

    # Create model, tokenizer, dataset
    model = MemeModel(parameters_dict=parameters_dict, device_input=device, num_labels_input=num_labels)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    dataset = Dataset(input_filepath_data + '/data.pkl', tokenizer=tokenizer, device=device)
    train_set, val_set = torch.utils.data.random_split(dataset, [N_train, N_test])
    # Define data loader and optimizer
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

    early_stopping_callback = EarlyStopping(
        monitor="val_accuracy", patience=3, verbose=True, mode="min"
    )
    wandb_logger = WandbLogger(project='wandb-lightning', entity=wandb_username, job_type='train')

    # apply Trainer according to whether or not the device is available
    if torch.cuda.is_available():
        trainer = Trainer(progress_bar_refresh_rate=50, gpus=1,
                          logger=wandb_logger, callbacks=[early_stopping_callback])
    else:
        trainer = Trainer(callbacks=[early_stopping_callback], logger=wandb_logger,
                          log_every_n_steps=5)

    trainer.fit(model, trainloader, valloader)

    checkpoint = {
        'parameters_dict': parameters_dict,
        'device': device,
        'num_labels': num_labels,
        'state_dict': model.state_dict()
    }

    torch.save(checkpoint, output_filepath_model + "/finetuned/checkpoint.pth")
    model.save()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
