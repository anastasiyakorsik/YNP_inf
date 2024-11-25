import sys
import argparse
import json
import os
from typing import Dict, Any, List
import uuid
import logging

import torch

from super_gradients.training import models

from inference.inference import inference_mode
# from train.train_yolonas_pose import train_mode
from common.container_status import ContainerStatus as CS
from common.generate_callback_data import generate_error_data

from workdirs import WEIGHTS_PATH, OUTPUT_PATH, INPUT_PATH, INPUT_DATA_PATH

logger_name = "yolo-nas-pose"
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
    try:
        args = parse_arguments()
        # json_data = args.input_data
        host_web = args.host_web
        training_mode = args.work_format_training

        # get all JSON files for each video from input_data directory
        json_files = os.listdir(INPUT_DATA_PATH)

        # model size accroding to model weights file
        model_size = "l"

        # init callback
        cs = None
        # cs = CS(host_web)
        # cs.post_start()

        # Getting model from weigths
        py_logger.info("Getting model:")
        model = models.get(f"yolo_nas_pose_{model_size}", num_classes = 17, checkpoint_path=WEIGHTS_PATH)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        if training_mode:
            #TODO: merge files in one or other way
            # train_mode(model, data)
            print('Trining mode is under development')
        else:
            # for each video go inference and add its json with new data
            py_logger.info("Start inference")
            print("Start inference")
            inference_mode(model, cs, json_files)
        
        # cs.post_end()
            
    except Exception as err:
        py_logger.exception(f"Exception occurred in main(): {err}",exc_info=True)


if __name__ == "__main__":
    main()
