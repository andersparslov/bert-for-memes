bert-for-memes
==============================

The goal of this project is to take as input the text of a meme, and predict the degree of humour. The labels are ordinal-categorical, i.e. the degree of humour is encoded as 0 -> not funny, 1 -> funny, 2 ->very funny and 3 -> hilarious. We use the pretrained transformer based model to take the tokenized text as input and convert this to a latent space representation in which the degree of humor is discernible. We use an extra sequence classification layer when finetuning.

The data is sourced from the Memotion dataset (https://www.kaggle.com/williamscott701/memotion-dataset-7k), we finetune an uncased distilbert model  (https://huggingface.co/distilbert-base-uncased) to the data. Distilbert is smaller version of BERT which has around half as many parameters but can achieve nearly the same performance on many downstream tasks. Uncased means that we don't discern between "Uncased" and "uncased".


### Raw data samples
![DataScreenshot](reports/figures/raw_data_screenshot.png?raw=true "Data screenshot")

### Labels distribution
Humour distribution:
![Figure1](reports/figures/humour_distribution.png?raw=true "Humour distribution")

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## How to run the project
### Clone the project:
git clone https://github.com/andersparslov/bert-for-memes
### Download data:
Download the data at https://www.kaggle.com/williamscott701/memotion-dataset-7k and put "labels_pd_pickle" into the "data/raw" folder.
### Prepare the dataset:
```
make data
```
- data will be stored in ./data
### Train the model:
- initialize wandb logging first
```
wandb init
```
- train the model:
```
make train
```
- edit config.yaml to configure hyperparameters
- model will be saved in ./models/finetuned
- training plots will be logged using wandb
- model checkpoints will be saved at "models/finetuned"
### Query deployed model:
python meme_request.py