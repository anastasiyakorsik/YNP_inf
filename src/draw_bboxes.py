import sys
import argparse
import json
import os
from typing import Dict, Any, List
import uuid
import logging

import torch

from super_gradients.training import models

import cv2
import numpy as np

import time
from tqdm import tqdm
from typing import Type

from inference.prediction_processor import get_prediction_per_frame, compare_bboxes, calculate_chain_vector
from inference.video_processor import get_frame_times, check_video_extension

from common.data_classes import OutputData, File, Chain, Markup, MarkupPath, save_class_in_json
from common.json_processing import load_json, save_json
from common.generate_callback_data import generate_error_data, generate_progress_data
from common.logger import py_logger

from workdirs import INPUT_PATH, OUTPUT_PATH, INPUT_DATA_PATH, WEIGHTS_PATH

OUT_FRAMES = "../out_mark_frames"

def collect_bboxes(input_data):
    """
    Collects dict where key - video frame and value - list of marked bboxes
    """
    try:
        input_bboxes = {}

        for chain in input_data.get('file_chains', []):
            for markup in chain.get('chain_markups', []):

                markup_frame = markup['markup_frame']
                markup_path = markup['markup_path']

                if markup_frame in input_bboxes:
                    input_bboxes[markup_frame].append(markup_path)
                else:
                    input_bboxes[markup_frame] = [markup_path]
                    
        return input_bboxes
    except Exception as e:
        py_logger.exception(f'Exception occurred in collect_bboxes(): {e}', exc_info=True)

def draw_bboxes(frame, frame_index, existing_bboxes, predicted_bboxes, overlap_color=(0, 255, 255)):
    """
    Draws multiple bounding boxes on a video frame and highlights overlap areas.
    
    Parameters:
        frame (numpy.ndarray): текущий кадр видео
        existing_bboxes (list): список существующих боксов [{"x", "y", "width", "height"}]
        predicted_bboxes (list): список предсказанных боксов [{"x", "y", "width", "height"}]
        overlap_color (tuple): цвет перекрытия в формате BGR (по умолчанию жёлтый)
    """
    try:
        overlay = frame.copy()
        for existing_bbox in existing_bboxes:
            # Рисуем существующий бокс (зеленый)
            x1, y1, w1, h1 = map(int, [existing_bbox["x"], existing_bbox["y"], existing_bbox["width"], existing_bbox["height"]])
            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)

            for predicted_bbox in predicted_bboxes:
                # Рисуем предсказанный бокс (красный)
                x2, y2, w2, h2 = map(int, [predicted_bbox["x"], predicted_bbox["y"], predicted_bbox["width"], predicted_bbox["height"]])
                cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)

                # Вычисляем область пересечения
                xi1 = max(x1, x2)
                yi1 = max(y1, y2)
                xi2 = min(x1 + w1, x2 + w2)
                yi2 = min(y1 + h1, y2 + h2)
                inter_width = max(xi2 - xi1, 0)
                inter_height = max(yi2 - yi1, 0)

                if inter_width > 0 and inter_height > 0:
                    # Рисуем область пересечения (жёлтая)
                    xi1, yi1, xi2, yi2 = map(int, [xi1, yi1, xi2, yi2])
                    cv2.rectangle(overlay, (xi1, yi1), (xi2, yi2), overlap_color, -1)  # Заполненный прямоугольник

                    intersection_area = inter_width * inter_height
                    bbox1_area = w1 * h1
                    bbox2_area = w2 * h2
                    union_area = bbox1_area + bbox2_area - intersection_area
                    iou = 0
                    if union_area > 0:
                        iou = intersection_area / union_area
                    
                    text = f"IoU: {iou:.2f}"
                    text_x, text_y = xi1, yi1 - 5
                    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    
        alpha = 0.4  # Прозрачность
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        if not os.path.exists(OUT_FRAMES):
            os.makedirs(OUT_FRAMES)
        output_path = os.path.join(OUT_FRAMES, f"frame_{frame_index:05d}.png")
        cv2.imwrite(output_path, frame)
        print(f"Saved: {output_path}")

        return frame
    except Exception as e:
        py_logger.exception(f'Exception occurred in draw_bboxes(): {e}', exc_info=True)

def process_bboxes(model, json_files, conf: float = 0.6):
    """
    """
    try:
        for json_file in json_files:

            full_json_path = os.path.join(INPUT_DATA_PATH, json_file)
            py_logger.info(f"Processing current input JSON file: {full_json_path}")
            data = load_json(full_json_path)

            input_data = data['files'][0]
            file_name = os.path.basename(input_data['file_name'])
            full_file_name = os.path.join(INPUT_PATH, file_name)
            py_logger.info(f"Processing current video file: {full_file_name}")

            if not os.path.exists(full_file_name):
                py_logger.error(f"Failed to create predictions for {file_name}. File does not exist.")
                #cs.post_error(generate_error_data(f"Failed to create predictions for {file_name}. File does not exist.", file_name))
                continue  # Skip if file does not exist
                
            if not check_video_extension(file_name):
                py_logger.error(f"Failed to create predictions for {file_name}. File extension doens't match appropriate ones.")
                #cs.post_error(generate_error_data(f"Failed to create predictions for {file_name}. File extension doens't match appropriate ones.", file_name))
                continue  # Skip if file extension doens't match appropriate ones

            # Calculate frame times
            # frame_times = get_frame_times(full_file_name)

            # Collect all bboxes from input json
            input_bboxes = collect_bboxes(input_data)
            # print(f"Collected_bboxes: {input_bboxes}")
            

            # Get model predictions
            preds, inference_time = get_prediction_per_frame(model, full_file_name, conf=conf)

            output_path = os.path.join(INPUT_PATH, "bboxes_" + file_name)

            cap = cv2.VideoCapture(full_file_name)
            if not cap.isOpened():
                print("Error: Could not open video.")
                return

            # Настройка для сохранения видео
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_id = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                try:
                    predicted_frame_bboxes = preds[frame_id]['processed_prediction']['bboxes']
                    input_frame_bboxes = input_bboxes[frame_id]
                    if predicted_frame_bboxes and input_frame_bboxes:
                        # Рисуем боксы на текущем кадре
                        frame = draw_bboxes(frame, input_frame_bboxes, predicted_frame_bboxes)
                except Exception as e:
                    break
                    

                # Записываем обработанный кадр
                out.write(frame)

                frame_id += 1  # Переходим к следующему кадру

            cap.release()
            out.release()
            cv2.destroyAllWindows()


    except Exception as e:
        py_logger.exception(f'Exception occurred in process_bboxes(): {e}', exc_info=True)

def main():
    try:
        # get all JSON files for each video from input_data directory
        json_files = os.listdir(INPUT_DATA_PATH)

        # model size accroding to model weights file
        model_size = "l"

        # Getting model from weigths
        py_logger.info("Getting model:")
        model = models.get(f"yolo_nas_pose_{model_size}", num_classes = 17, checkpoint_path=WEIGHTS_PATH)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        process_bboxes(model, json_files)
    except Exception as e:
        py_logger.exception(f'Exception occurred in main(): {e}', exc_info=True)

if __name__ == "__main__":
    main()

