import os

_N_TRAIN = 6830
_INPUT_FILE_PATH = "data/processed/data.pkl"
_MODELS_FILE_PATH = "models/"


_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(os.path.join(_PROJECT_ROOT, "data"), "processed")  # root of data
