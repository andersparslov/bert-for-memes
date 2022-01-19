# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import pickle
import json
import torch


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('json_path', type=click.Path())
def main(input_filepath, output_filepath, json_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Load data 
    with open(input_filepath,"rb") as file_handle:
        data_df = pickle.load(file_handle)

    # Cleaning: drop NAs
    data_df = data_df.dropna()

    # Cleaning: map 4 values of 'humour' to numeric
    def categorize_humour(x):
        if x == 'not_funny':
            return 0
        elif x == 'funny':
            return 1
        elif x == 'very_funny':
            return 2
        else: # hilarious
            return 3
    
    texts = data_df.text_corrected.to_numpy(dtype=str)
    labels = data_df.humour.apply(categorize_humour).to_numpy()
    
    # Save processed data
    data_dict = {'texts':texts, 'labels':labels}

    with open(output_filepath,"wb") as file_handle:
        pickle.dump(data_dict, file_handle)

    classification_dict = {
                            0: 'not_funny',
                            1: 'funny',
                            2: 'very_funny',
                            3: 'hilarious'
                          }

    with open(json_path + '/classification_dict.json', 'w') as fp:
        json.dump(classification_dict, fp)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
