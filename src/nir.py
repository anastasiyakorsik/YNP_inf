import argparse
import json
import os
import logging

import torch

from super_gradients.training import models

from inference.inference import inference_mode
# from train.train_yolonas_pose import train_mode
from common.container_status import ContainerStatus as CS
from common.generate_callback_data import generate_error_data

from workdirs import WEIGHTS_PATH, OUTPUT_PATH, INPUT_PATH, INPUT_DATA_PATH

logger_name = "ynp_diff_sizes"
py_logger = logging.getLogger(logger_name)
py_logger.setLevel(logging.DEBUG)

py_handler = logging.FileHandler(f"{logger_name}.log", mode='w')
py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")

py_handler.setFormatter(py_formatter)
py_logger.addHandler(py_handler)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process input data and work format.')
    parser.add_argument('--input_data', type=str, default='', help='Input data in JSON format')
    parser.add_argument('--host_web', type=str, required = True, help='Callback url')
    parser.add_argument('--work_format_training', action='store_true', default=False, help='Program mode')
    return parser.parse_args()

def main():
    