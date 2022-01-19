import torch
from src.models.Model_red_boilerplate import MemeModel


def load_checkpoint(filepath):

    checkpoint = torch.load(filepath)
    model = MemeModel(parameters_dict=checkpoint['parameters_dict'],
                      device_input=checkpoint['device'],
                      num_labels_input=checkpoint['num_labels'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to('cpu')

    return model
