import os
import pickle

from common.logger import py_logger
from workdirs import INPUT_PATH, OUTPUT_PATH


def read_pkl_or_default(file_name: str, default_value=[]):
    try:
        return read_pkl(file_name)
    except Exception as e:
        py_logger.exception(f'Failed to read {file_name}. Return value is []', exc_info=True)
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
            pkl_data = pickle.load(file)
            return pkl_data if pkl_data is not None else []
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

def get_input_pkl_vector_elements_length(pkl_vector_name, pkl_vector_value):
    py_logger.info("Check input pkl vectors lengths")
    pkl_vector_value.tolist()
    if len(pkl_vector_value) > 0:
        py_logger.info(f"Length of pkl vector ({pkl_vector_name}) is {len(pkl_vector_value)}")
        for i in range(len(pkl_vector_value)):
            py_logger.warning(f"Length of {i} element of pkl vector ({pkl_vector_name}) is {len(pkl_vector_value[i])}; Should be 1000.")
    else:
        py_logger.info(f"Length of pkl vector ({pkl_vector_name}) is 0")

def get_output_pkl_vector_elements_length(pkl_vector_name, pkl_vector_value):
    py_logger.info("Check output pkl vectors lengths")
    if len(pkl_vector_value) > 0:
        py_logger.info(f"Length of pkl vector ({pkl_vector_name}) is {len(pkl_vector_value)}")
        for i in range(len(pkl_vector_value)):
            if not (len(pkl_vector_value[i])==1000 or len(pkl_vector_value[i])==1017 or len(pkl_vector_value[i])==17):
                py_logger.warning(f"Length of {i} element of pkl vector ({pkl_vector_name}) is {len(pkl_vector_value[i])}; Should be 1000 or 1017 or 17.")
    else:
        py_logger.info(f"Length of pkl vector ({pkl_vector_name}) is 0")

