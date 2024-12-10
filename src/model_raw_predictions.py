import sys
import argparse
import json
import os
from typing import Dict, Any, List
import uuid
import logging
from tqdm import tqdm

import torch

from super_gradients.training import models

INPUT_FILE_PATH = "../input/video.mp4"
WEIGHTS_PATH = "../weights/yolo_nas_pose_l_coco_pose.pth"

def main():
    try:
        # path to model weights
        model_size = "l"

        # Getting model from weigths
        print("Getting model:")
        model = models.get(f"yolo_nas_pose_{model_size}", num_classes = 17, checkpoint_path=WEIGHTS_PATH)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        model_prediction = model.predict(INPUT_FILE_PATH, conf=0.6)
        raw_prediction = [res for res in tqdm(model_prediction._images_prediction_gen, total=model_prediction.n_frames, desc="Processing Video")]

        for prediction in raw_prediction:
            print(prediction)
            break
            
    except Exception as err:
        print(f"Exception occurred in main(): {err}")
        #TODO: Add on_error


if __name__ == "__main__":
    main()