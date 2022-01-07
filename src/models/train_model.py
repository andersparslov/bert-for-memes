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

@click.command()
@click.argument('input_filepath_data', type=click.Path(exists=True))
@click.argument('output_filepath_model', type=click.Path(exists=True))
@click.argument('output_plot_model', type=click.Path())
def main(input_filepath_data, output_filepath_model, output_plot_model):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Is CUDA? ', torch.cuda.is_available())

    model = MemeModel(device, num_labels=2)

    #train_set_text = torch.load(input_filepath_data + "/text_train.pt")
    #train_set_labels = torch.load(input_filepath_data + "/labels_train.pt")

    #train_set = TensorDataset(train_set_images, train_set_labels)


    # Dummy Data ###################
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, truncation=True, padding=True)
    encoded_input['labels'] = 0
    print(encoded_input)
    train_set = [encoded_input] * 10000
    #####################
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    epochs = 13
    steps = 0
    running_loss = 0
    running_losses = []
    steps_list = []

    print_every = 100
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for item in trainloader:
            steps += 1

            optimizer.zero_grad()

            y_pred, loss = model(item)

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