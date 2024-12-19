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
from common.generate_callback_data import generate_error_data, generate_progress_data

from workdirs import OUTPUT_PATH, INPUT_PATH, INPUT_DATA_PATH, WEIGHTS_DIR

WEIGHTS_FILE = "yolo_nas_pose_l_coco_pose.pth"
MODEL_SIZE = "l"
EPOCHS_NUM = 5

logger_name = "yolo-nas-pose"
py_logger = logging.getLogger(logger_name)
py_logger.setLevel(logging.DEBUG)

py_handler = logging.FileHandler(f"{logger_name}.log", mode='w')
py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")

py_handler.setFormatter(py_formatter)
py_logger.addHandler(py_handler)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process input data and work format.')
    parser.add_argument('--input_data', type=str, default='', help='Necessary input data in JSON format')
    parser.add_argument('--host_web', type=str, default='', help='Callback url')
    parser.add_argument('--work_format_training', action='store_true', default=False, help='Program mode')
    return parser.parse_args()

def main():
    try:
        args = parse_arguments()
        # sample: input_data = {"weights_file": "yolo_nas_pose_l_coco_pose.pth", "model_size": "l", "epochs_num": 5}
        json_data = args.input_data
        host_web = args.host_web
        training_mode = args.work_format_training

        if input_data:
            try:
                WEIGHTS_FILE = input_data["weights_file"]
                MODEL_SIZE = input_data["model_size"]
                EPOCHS_NUM = input_data["epochs_num"]
            except Exception as ex:
                py_logger.error(f"Failed to load necessary input params from given input_data: {input_data}. Default params will be used.")

        # get all JSON files for each video from input_data directory
        json_files = os.listdir(INPUT_DATA_PATH)

        # Callback init
        cs = None

        if host_web:
            cs = CS(host_web)

        if cs is not None:
            cs.post_start({"msg": "Start processing by YOLO-NAS Pose model"})
            cs.post_progress(generate_progress_data("start", 0))

        WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, WEIGHTS_FILE)

        # Getting model from weigths
        py_logger.info("Getting model:")
        model = models.get(f"yolo_nas_pose_{MODEL_SIZE}", num_classes = 17, checkpoint_path=WEIGHTS_PATH)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        reload_model = False

        if training_mode:
            # train_mode(model, data)
            print('Training mode is under development')
            reload_model = True
        
        if reload_model:
            py_logger.info("Reload model from updated weights:")
            model = models.get(f"yolo_nas_pose_{MODEL_SIZE}", num_classes = 17, checkpoint_path=WEIGHTS_PATH)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)

        # for each video go inference and create output json with new data
        py_logger.info("Start inference")
        output_files = inference_mode(model, json_files, cs)
        
        if cs is not None:
            cs.post_end({"out_files": output_files})
            
    except Exception as err:
        py_logger.exception(f"Exception occurred in main(): {err}",exc_info=True)


if __name__ == "__main__":
    main()
