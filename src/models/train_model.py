# -*- coding: utf-8 -*-
import click
import logging
import hydra
from pathlib import Path
from variable import PROJECT_PATH
from dotenv import find_dotenv, load_dotenv
from torchvision import datasets, transforms
import numpy as np
import torch
import argparse
import sys
import torch
from transformers import DistilBertTokenizer
from Model import MemeModel
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

import sys 
from src.data.dataset import * 

# Note: Hydra is incompatible with @click
@hydra.main(config_path= PROJECT_PATH / "configs",config_name="/config.yaml")

#@click.command()
#@click.argument('input_filepath_data', type=click.Path(exists=True))
#@click.argument('output_filepath_model', type=click.Path(exists=True))
#@click.argument('output_plot_model', type=click.Path())

def main(cfg):

    # for hydra parameters
    input_filepath_data = str(PROJECT_PATH / "data" / "processed")
    output_filepath_model = str(PROJECT_PATH / "models")
    output_plot_model = str(PROJECT_PATH / "reports" / "figures")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Is CUDA? ', torch.cuda.is_available())

    '''
    batch_size = 4
    lr = 1e-4
    epochs = 13
        
    print_every = 2

    num_labels = 4
    N_train = 6000
    N_test = 830
    '''

    print_every = cfg.hyperparameters.print_every
    num_labels = cfg.hyperparameters.num_labels
    N_train = cfg.hyperparameters.N_train
    N_test = cfg.hyperparameters.N_test

    epochs = cfg.hyperparameters.epochs
    lr = cfg.hyperparameters.lr
    batch_size = cfg.hyperparameters.batch_size

    # Create model, tokenizer, dataset
    model = MemeModel(config=None, device=device, num_labels=num_labels)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    dataset = Dataset(input_filepath_data,tokenizer=tokenizer,device=device)
    train_set, val_set = torch.utils.data.random_split(dataset, [N_train, N_test])
    # Define data loader and optimizer
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    steps = 0
    running_loss = 0
    running_losses = []
    steps_list = []

    print_every = 100
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for text, labels in trainloader:
            steps += 1

            optimizer.zero_grad()

            y_pred, loss = model(text['input_ids'].squeeze(),
                                 text['attention_mask'].squeeze(),
                                 labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                # Model in inference mode, dropout is off
                correct_count = 0
                for text, labels in valloader:
                    with torch.no_grad():
                        y, loss = model(text['input_ids'].squeeze(),
                                        text['attention_mask'].squeeze(),
                                        labels)
                        y = torch.argmax(y, dim=1)
                        correct_count += torch.sum(y == labels)
                accuracy = correct_count / len(val_set)

                running_losses.append(running_loss / print_every)
                mean_loss = running_loss / print_every
                print('Epoch: {}/{} - Training loss: {:.2f} Validation accuracy {:.2f}'.format(e,
                                                                                               epochs,
                                                                                               mean_loss,
                                                                                               accuracy))
                steps_list.append(steps)

                running_loss = 0


    #torch.save(model.state_dict(), output_filepath_model + "models/finetuned/checkpoint.pth")
    model.save()

    plt.plot(steps_list, running_losses)
    plt.legend()
    plt.title("Training losses")
    plt.show()
    plt.savefig(output_plot_model + '/training_plot.png')
    plt.close()
    



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()