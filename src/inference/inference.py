import time
import uuid
import os
from tqdm import tqdm
from typing import Type
import statistics

import cv2

from inference.prediction_processor import get_prediction_per_frame, compare_bboxes, calculate_chain_vector, calc_mean
from inference.video_processor import get_frame_times, check_video_extension
from inference.bbox_processing import filter_bboxes_with_content, draw_bboxes_and_save, OUT_FRAMES

from common.data_classes import OutputData, File, Chain, Markup, MarkupPath, save_class_in_json
from common.json_processing import load_json, save_json
from common.generate_callback_data import generate_error_data, generate_progress_data
from common.logger import py_logger

from workdirs import INPUT_PATH, OUTPUT_PATH, INPUT_DATA_PATH

def create_add_json(input_data: dict, predictions: list, matched_bboxes: list, frame_times: list, progress: float, cs = None):
    """
    Creates additional JSON file with unmatched bboxes and skeletones in new chains

    Arguments:
        input_data (dict): Input JSON data for one video
        predictions (list): Model predictions for video
        matched_bboxes (list): Predicted bboxes that matched input ones
        frame_times (list): Calculated times of video frames
        progress (float): Progress percent 
        cs (obj): ContainerStatus object
        
    Returns:
        Name of additional output file if is not empty, else - None
    """
    try:
        file_name = os.path.basename(input_data['file_name'])

        # Output JSON file name for recording new bboxes
        out_json_add = file_name + '_add.json'
        out_json_add_path = os.path.join(OUTPUT_PATH, out_json_add)

        output_add_data = OutputData()
        output_add_data.files = []

        output_add_file_data = File(file_id = input_data.get('file_id', ''), file_name = file_name)
        output_add_file_data.file_chains = []

        for pred in predictions:
            frame = pred['markup_frame']
            predicted_bboxes = pred['processed_prediction']['bboxes']
            predicted_nodes = pred['processed_prediction']['nodes']
            predicted_edges = pred['processed_prediction']['edges']
            predicted_scores = pred['processed_prediction']['scores']

            for i in range (len(predicted_bboxes)):
                detected_bbox = predicted_bboxes[i]

                if not detected_bbox in matched_bboxes:

                    out_add_chain = Chain()
                    out_add_chain.chain_name = str(int(uuid.uuid4().int >> 64) % 1000000)
                    out_add_chain.chain_markups = []

                    out_markup = Markup()
                    out_markup.markup_time = frame_times[frame]
                    out_markup.markup_frame = frame

                    detected_person_nodes = predicted_nodes[i]

                    out_markup_path = MarkupPath()
                    out_markup_path.x, out_markup_path.y, out_markup_path.width, out_markup_path.height = detected_bbox['x'], detected_bbox['y'], detected_bbox['width'], detected_bbox['height']
                    out_markup_path.nodes = detected_person_nodes
                    out_markup_path.edges = predicted_edges
                                    
                    out_markup.markup_path = out_markup_path
                    out_markup.markup_id = str(int(uuid.uuid4().int >> 64) % 1000000)
                    out_markup.markup_vector = predicted_scores[i]
                    out_markup.markup_confidence = statistics.fmean(predicted_scores[i])

                    out_add_chain.chain_markups.append(out_markup)
                    out_add_chain.chain_vector = predicted_scores[i]
                    out_add_chain.chain_confidence = statistics.fmean(predicted_scores[i])

                    output_add_file_data.file_chains.append(out_add_chain)

        if output_add_file_data:
            output_add_data.files.append(output_add_file_data)

            save_class_in_json(output_add_data, out_json_add_path)

            py_logger.info(f"Inference results for {file_name} saved and recorded into {out_json_add_path}")
            if cs is not None:
                cs.post_progress(generate_progress_data("processed", progress, out_json_add_path))
            
            return out_json_add_path
        else:
            py_logger.info("No unmatched bboxes recorded")
            return None

    except Exception as e:
        py_logger.exception(f'Exception occurred in inference.create_add_json(): {e}', exc_info=True)
        if cs is not None:
            cs.post_error(generate_error_data(f"Failed to inference. Exception occurred in inference.create_add_json(): {e}"))



def create_json_with_predictions(input_data: dict, frame_times: list, model, progress: float, cs = None, conf: float = 0.6):
    """
    Adds to input JSON model predictions and saves result in output JOSN for each videofile
    
    Arguments:
        input_data (dict): Input JSON data for one video
        frame_times (list): Calculated times of video frames
        model: Prediction model
        progress (float): Progress percent 
        cs (obj): ContainerStatus object
        conf (float): [OPTIONAL] Model confidence
        
    Returns:
        Names of main and additional output files
    """
    try:

        file_name = os.path.basename(input_data['file_name'])
        full_file_name = os.path.join(INPUT_PATH, file_name)
            
        # Output JSON file name
        out_json = file_name + '.json'
        out_json_path = os.path.join(OUTPUT_PATH, out_json)

        # Dict for output collected data - collect new json data containing model predictions
        output_data = OutputData()
        output_data.files = []

        output_file_data = File(file_id = input_data.get('file_id', ''), file_name = file_name)
        output_file_data.file_chains = []

        # Inference for each frame of video
        if cs is not None:
            cs.post_progress(generate_progress_data("predicting", progress))
        preds, inference_time = get_prediction_per_frame(model, full_file_name, conf=conf)

        # Processing inference result
        if cs is not None:
            cs.post_progress(generate_progress_data("processing", progress))

        # open video to record marked frames
        cap = cv2.VideoCapture(full_file_name)
        if not cap.isOpened():
            py_logger.error(f"Error: Could not open video {full_file_name}.")
            return None

        # create dir for detected frames
        if not os.path.exists(OUT_FRAMES):
            os.makedirs(OUT_FRAMES)

        # List of matched predicted bboxes
        # then will be used to define unique ones
        matched_pred_bboxes = []

        for chain in input_data.get('file_chains', []):
            	
            matched_chain_bboxes = 0

            # create new chain with input and added info to save if bboxes matched
            out_chain = Chain()
            out_chain.chain_name = chain.get('chain_name', '')
            out_chain.chain_markups = []

            input_chain_vector = chain.get('chain_vector', [])
            # array of markup_vectors per chain for calculating chain_vector
            predicted_markup_vectors = []
            # array of markup_confidence per chain for calculating chain_confidence
            predicted_markup_confidence = []

            for markup in chain.get('chain_markups', []):
                out_markup = Markup()

                markup_frame = markup.get('markup_frame', '')

                complete_step = False

                for pred in preds:

                    frame = pred['markup_frame']

                    if markup_frame == frame:
                        predicted_bboxes = pred['processed_prediction']['bboxes']
                        predicted_nodes = pred['processed_prediction']['nodes']
                        predicted_edges = pred['processed_prediction']['edges']
                        predicted_scores = pred['processed_prediction']['scores']

                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                        ret, cur_frame_obj = cap.read()
                        if not ret:
                            py_logger.error(f"Error: Could not read frame {frame} from video {full_file_name}")
                            continue

                        out_markup.markup_time = frame_times[frame]
                        out_markup.markup_frame = frame

                        input_markup_path = markup['markup_path']

                        # YOLO NAS POSE creates detections per frame
                        # hence, predicted_bboxes is an array of all bboxes per frame
                        for i in range (len(predicted_bboxes)):
                            detected_bbox = predicted_bboxes[i]
                            
                            out_frame_name = f"{file_name}_chain_{chain['chain_name']}_frame_{frame}.png"
                            
                            # draw_bboxes_and_save(cur_frame_obj, frame, out_frame_name, detected_bbox, input_markup_path)

                            if filter_bboxes_with_content(input_markup_path, detected_bbox, cur_frame_obj):

                                matched_pred_bboxes.append(detected_bbox)

                                detected_person_nodes = predicted_nodes[i]

                                out_markup_path = MarkupPath()
                                out_markup_path.x, out_markup_path.y, out_markup_path.width, out_markup_path.height = input_markup_path['x'], input_markup_path['y'], input_markup_path['width'], input_markup_path['height']
                                out_markup_path.nodes = detected_person_nodes
                                out_markup_path.edges = predicted_edges
                                
                                out_markup.markup_path = out_markup_path

                                predicted_markup_vectors.append(predicted_scores[i])
                                
                                #print("BBoxes matched")
                                matched_chain_bboxes += 1

                                out_markup.markup_parent_id = markup.get('markup_id', '')
                                out_markup.markup_vector = markup['markup_vector'] + predicted_scores[i]
                                
                                
                                mean_conf_score = statistics.fmean(predicted_scores[i])
                                out_markup.markup_confidence = mean_conf_score
                                predicted_markup_confidence.append(mean_conf_score)

                                out_chain.chain_markups.append(out_markup)
                                complete_step = True
                                break

                        if complete_step:
                            break

                
                out_chain.chain_vector = calculate_chain_vector(input_chain_vector, predicted_markup_vectors)
                # print(predicted_markup_confidence)
                if predicted_markup_confidence:
                    out_chain.chain_confidence = statistics.fmean(predicted_markup_confidence)
                    
            py_logger.info(f"Have {matched_chain_bboxes} matched bboxes for chain {chain['chain_name']}")
                    
            if out_chain.chain_markups:
                output_file_data.file_chains.append(out_chain)

        # Save new JSON
        output_data.files.append(output_file_data)

        save_class_in_json(output_data, out_json_path)

        py_logger.info(f"Inference results for {file_name} saved and recorded into {out_json_path}")
        # print(f"Inference results for {file_name} saved and recorded into {out_json_path}")
        if cs is not None:
            cs.post_progress(generate_progress_data("processed", progress, out_json_path))

        out_json_add_path = create_add_json(input_data, preds, matched_pred_bboxes, frame_times, progress, cs)
        
        return out_json_path, out_json_add_path

    except Exception as e:
        py_logger.exception(f'Exception occurred in inference.create_json_with_predictions(): {e}', exc_info=True)
        if cs is not None:
            cs.post_error(generate_error_data(f"Failed to inference. Exception occurred in inference.create_json_with_predictions(): {e}"))
       
def inference_mode(model, json_files: list, cs = None):
    """
    Inference program mode. Processing video and
    adding new data about skeleton to each input JSON and save

    Arguments:
        model: YOLO-NAS Pose downloaded model
        json_files (list): List of all json files in input directory
        cs (obj): [OPTIONAL] ContainerStatus object
        
    Returns: List of output files names
    """
    try:
        # for each video go inference and create output JSON
        output_files = []
        file_num = 0
        for json_file in json_files:

            full_json_path = os.path.join(INPUT_DATA_PATH, json_file)
            py_logger.info(f"Processing current input JSON file: {full_json_path}")
            data = load_json(full_json_path)
            
            if data:
                input_data = data['files'][0]
                file_name = os.path.basename(input_data['file_name'])
                full_file_name = os.path.join(INPUT_PATH, file_name)
                py_logger.info(f"Processing current video file: {full_file_name}")

                if not os.path.exists(full_file_name):
                    py_logger.error(f"Failed to create predictions for {file_name}. File does not exist.")
                    if cs is not None:
                        cs.post_error(generate_error_data(f"Failed to create predictions for {file_name}. File does not exist.", file_name))
                    continue  # Skip if file does not exist
                
                if not check_video_extension(file_name):
                    py_logger.error(f"Failed to create predictions for {file_name}. File extension doens't match appropriate ones.")
                    if cs is not None:
                        cs.post_error(generate_error_data(f"Failed to create predictions for {file_name}. File extension doens't match appropriate ones.", file_name))
                    continue  # Skip if file extension doens't match appropriate ones

                # Make and process model predictions for video
                file_num += 1
                progress = round((file_num) / len(json_files) * 100, 2)

                frame_times = get_frame_times(full_file_name)
                out_file, out_add_file = create_json_with_predictions(input_data, frame_times, model, progress, cs)
            
                output_files.append(out_file)
                if out_add_file is not None:
                    output_files.append(out_add_file)
            else:
                py_logger.error(f"Failed to create predictions for {full_json_path}. File can not be read.")
                continue
            
        if (len(output_files) == 0):
            py_logger.warning("No files have been processed.")
            
        return output_files

    except Exception as e:
        py_logger.exception(f'Exception occurred in inference.inference_mode(): {e}', exc_info=True)
        if cs is not None:
            cs.post_error(generate_error_data(f"Failed to inference. Exception occurred in inference.inference_mode(): {e}"))
