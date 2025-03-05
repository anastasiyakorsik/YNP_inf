import os
import pickle

from common.logger import py_logger
from workdirs import INPUT_PATH, OUTPUT_PATH

def read_pkl(filename: str):
    """
    Get data from pkl file
    """
    try:
        with open(filename, "rb") as file:
            data = pickle.load(file)
        
        return data
    except Exception as e:
        py_logger.exception(f'Exception occurred in pkl_processing.read_pkl(): {e}', exc_info=True)
        return None

def create_pkl(filename: str, data):
    """
    Dumps data in pkl file
    """
    try:
        with open(filename, "wb") as file:
            pickle.dump(data, file)
    except Exception as e:
        py_logger.exception(f'Exception occurred in pkl_processing.create_pkl(): {e}', exc_info=True)

def define_pkl_name(file_name: str, add: bool = False):
    """
     
    """
    chains_vectors = file_name + '_chains_vectors.pkl'
    markups_vectors = file_name + '_markups_vectors.pkl'

    chains_vectors_in = os.path.join(INPUT_PATH, chains_vectors)
    markups_vectors_in = os.path.join(INPUT_PATH, markups_vectors)
    
    if add:
        chains_vectors = 'add_' + chains_vectors
        markups_vectors = 'add_' + markups_vectors 

    chains_vectors_out = os.path.join(OUTPUT_PATH, chains_vectors)
    markups_vectors_out = os.path.join(OUTPUT_PATH, markups_vectors)

    return chains_vectors, markups_vectors, chains_vectors_in, markups_vectors_in, chains_vectors_out, markups_vectors_out
