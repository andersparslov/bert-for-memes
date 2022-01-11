# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from torchvision import datasets, transforms
import numpy as np
import torch
import argparse
import sys
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from Model import MemeModel
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

import sys 
from src.data.dataset import * 

@click.command()
@click.argument('input_filepath_data', type=click.Path(exists=True))
@click.argument('output_filepath_model', type=click.Path(exists=True))
@click.argument('output_plot_model', type=click.Path())

def main(input_filepath_data, output_filepath_model, output_plot_model):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Is CUDA? ', torch.cuda.is_available())

    model = MemeModel(config=None, device=device, num_labels=4)
    

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_set = Dataset(input_filepath_data,tokenizer=tokenizer,device=device)

    
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)

    #criterion = nn.CrossEntropyLoss() #nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    

    epochs = 13
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

            y_pred, loss = model(text['input_ids'].squeeze(), text['attention_mask'].squeeze(), labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                # Model in inference mode, dropout is off

                running_losses.append(running_loss / print_every)
                mean_loss = running_loss / print_every
                print('Epoch: {}/{} - Training loss: {:.2f}'.format(e, epochs, mean_loss))

                steps_list.append(steps)

                running_loss = 0

    torch.save(model.state_dict(), output_filepath_model + "/checkpoint.pth")

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