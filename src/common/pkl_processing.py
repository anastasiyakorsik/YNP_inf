import os
import pickle

from src.common.logger import py_logger
from src.workdirs import INPUT_PATH, OUTPUT_PATH


def read_pkl_or_default(file_name: str, default_value=None):
    try:
        return read_pkl(file_name)
    except Exception as e:
        py_logger.exception(f'Failed to read {file_name}. Return value is None', exc_info=True)
        return default_value

def load_pkl_data(file_path):
    return read_pkl_or_default(file_path) if os.path.exists(file_path) else []

def read_pkl(filename: str):
    if filename is None:
        py_logger.exception(f'File name is None', exc_info=True)
        raise Exception()
    if not os.path.isfile(filename):
        py_logger.exception(f'File does not exist: {filename}', exc_info=True)
        raise Exception()
    try:
        with open(filename, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        py_logger.exception(f'Exception occurred in pkl_processing.read_pkl(): {e}', exc_info=True)
        raise Exception()

def create_pkl(filename: str, data):
    """
    Dumps data in pkl file
    """
    try:
        with open(filename, "wb") as file:
            pickle.dump(data, file)
    except Exception as e:
        py_logger.exception(f'Exception occurred in pkl_processing.create_pkl(): {e}\nFor file: {filename}', exc_info=True)


def get_chains_and_markups_pkl_file_names(file_name):
    return file_name + '_chains_vectors.pkl', file_name + '_markups_vectors.pkl'

