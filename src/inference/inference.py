import time
import uuid
import os
from tqdm import tqdm
from typing import Type
import statistics
import pickle

import cv2

from inference.prediction_processor import get_prediction_per_frame, compare_bboxes, calculate_chain_vector, calc_mean
from inference.video_processor import get_frame_times, check_video_extension
from inference.bbox_processing import filter_bboxes_with_content, draw_bboxes_and_save, OUT_FRAMES

from common.data_classes import OutputData, File, Chain, Markup, MarkupPath, save_class_in_json
from common.json_processing import load_json, save_json
from common.pkl_processing import read_pkl, create_pkl, define_pkl_name
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
        Dict containing out_add_file name, chains and markups counts for file
    """
    chains_count = 0
    markups_count = 0
    out_add_result = {}
    try:
        file_name = os.path.basename(input_data['file_name'])

        # Create arrays for big chains_vectors and markups_vectors for writing in .pkl files
        chains_vectors = []
        markups_vectors = []

        # Create .pkl file output name for chains and markups
        pkl_chains_vectors, pkl_markups_vectors, pkl_chains_vectors_in, pkl_markups_vectors_in, pkl_chains_vectors_out, pkl_markups_vectors_out = define_pkl_name(file_name, True)

        # Output JSON file name for recording new bboxes
        out_json_add = 'OUT_' + file_name + '_add.json'
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
                    markups_vectors.append(predicted_scores[i])
                    out_markup.markup_vector = markups_count
                    out_markup.markup_confidence = statistics.fmean(predicted_scores[i])
                    markups_count += 1

                    out_add_chain.chain_markups.append(out_markup)
                    chains_vectors.append(predicted_scores[i])
                    out_add_chain.chain_vector = chains_count
                    out_add_chain.chain_confidence = statistics.fmean(predicted_scores[i])
                    chains_count += 1

                    output_add_file_data.file_chains.append(out_add_chain)

        if output_add_file_data:

            # Save chains and markups vectors in .pkl files
            create_pkl(pkl_chains_vectors_out, chains_vectors)
            create_pkl(pkl_markups_vectors_out, markups_vectors)

            py_logger.info(f"Additional chains vectors for {file_name} saved and recorded into {pkl_chains_vectors_out}")
            py_logger.info(f"Additional markups vectors for {file_name} saved and recorded into {pkl_markups_vectors_out}")
            if cs is not None:
                cs.post_progress(generate_progress_data("3 of 4", progress, pkl_chains_vectors))
                cs.post_progress(generate_progress_data("3 of 4", progress, pkl_markups_vectors))

            output_add_data.files.append(output_add_file_data)
            save_class_in_json(output_add_data, out_json_add_path)

            py_logger.info(f"Inference results for {file_name} saved and recorded into {out_json_add_path}")
            if cs is not None:
                cs.post_progress(generate_progress_data("4 of 4", progress, out_json_add, chains_count, markups_count))

            out_add_result["out_file"] = out_json_add
            out_add_result["chains_count"] = chains_count
            out_add_result["markups_count"] = markups_count
            
            return out_add_result
        else:
            py_logger.info(f"No unmatched bboxes recorded for {file_name}")
            return out_add_result

    except Exception as e:
        py_logger.exception(f'Exception occurred in inference.create_add_json(): {e}', exc_info=True)
        if cs is not None:
            cs.post_error(generate_error_data(f"Failed to inference. Exception occurred in inference.create_add_json(): {e}"))
        return out_add_result



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
    out_result_data = []
    try:
        # Create arrays for big chains_vectors and markups_vectors for writing in .pkl files
        chains_vectors = []
        markups_vectors = []

        chains_count = 0
        markups_count = 0
        out_main_result = {}

        file_name = os.path.basename(input_data['file_name'])
        full_file_name = os.path.join(INPUT_PATH, file_name)

        # Create .pkl file name for chains and markups
        pkl_chains_vectors, pkl_markups_vectors, pkl_chains_vectors_in, pkl_markups_vectors_in, pkl_chains_vectors_out, pkl_markups_vectors_out = define_pkl_name(file_name)

        # Get data from chains and markups .pkl files
        input_chains_vectors = read_pkl(pkl_chains_vectors_in)
        input_markups_vectors = read_pkl(pkl_markups_vectors_in)
            
        # Output JSON file name
        out_json = 'OUT_' + file_name + '.json'
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
            chains_count += 1
            out_chain.chain_name = chain.get('chain_name', '')
            out_chain.chain_markups = []

            input_chain_vector = chain.get('chain_vector', [])
            # array of markup_vectors per chain for calculating chain_vector
            predicted_markup_vectors = []
            # array of markup_confidence per chain for calculating chain_confidence
            predicted_markup_confidence = []

            for markup in chain.get('chain_markups', []):
                out_markup = Markup()
                markups_count += 1

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

                                input_markup_vector = markup.get('markup_vector', [])
                                if input_markup_vector is not None:
                                    if (type(input_markup_vector) == int):
                                        in_markup_vector_val = input_markups_vectors[input_markup_vector]
                                        out_markup.markup_vector = input_markup_vector
                                        markups_vectors.append(in_markup_vector_val + predicted_scores[i])
                                    else:
                                        markups_vectors.append(input_markup_vector + predicted_scores[i])
                                        out_markup.markup_vector = len(markups_vectors) - 1
                                else:
                                    markups_vectors.append(predicted_scores[i])
                                    out_markup.markup_vector = len(markups_vectors) - 1
                                
                                mean_conf_score = statistics.fmean(predicted_scores[i])
                                out_markup.markup_confidence = mean_conf_score
                                predicted_markup_confidence.append(mean_conf_score)

                                out_chain.chain_markups.append(out_markup)
                                complete_step = True
                                break

                        if complete_step:
                            break

                if input_chain_vector is not None:
                    if (type(input_chain_vector) == int):
                        in_chain_vector_val = input_chains_vectors[input_chain_vector]
                        out_chain.chain_vector = input_chain_vector
                        chains_vectors.append(calculate_chain_vector(in_chain_vector_val, predicted_markup_vectors))
                    else:
                        chains_vectors.append(calculate_chain_vector(input_chain_vector, predicted_markup_vectors))
                        out_chain.chain_vector = len(chains_vectors) - 1
                else:
                    chains_vectors.append(predicted_markup_vectors)
                    out_chain.chain_vector = len(chains_vectors) - 1


                # print(predicted_markup_confidence)
                if predicted_markup_confidence:
                    out_chain.chain_confidence = statistics.fmean(predicted_markup_confidence)
                    
            py_logger.info(f"Have {matched_chain_bboxes} matched bboxes for chain {chain['chain_name']}")
                    
            if out_chain.chain_markups:
                output_file_data.file_chains.append(out_chain)

        # Save .pkl files
        # Save chains and markups vectors in .pkl files
        create_pkl(pkl_chains_vectors_out, chains_vectors)    
        create_pkl(pkl_markups_vectors_out, markups_vectors)

        py_logger.info(f"Chains vectors for {file_name} saved and recorded into {pkl_chains_vectors_out}")
        py_logger.info(f"Markups vectors for {file_name} saved and recorded into {pkl_markups_vectors_out}")
        if cs is not None:
            cs.post_progress(generate_progress_data("3 of 4", progress, pkl_chains_vectors))
            cs.post_progress(generate_progress_data("3 of 4", progress, pkl_markups_vectors))

        # Save new JSON
        output_data.files.append(output_file_data)

        save_class_in_json(output_data, out_json_path)

        py_logger.info(f"Inference results for {file_name} saved and recorded into {out_json_path}")

        out_main_result["out_file"] = out_json
        out_main_result["chains_count"] = chains_count
        out_main_result["markups_count"] = markups_count
        # print(f"Inference results for {file_name} saved and recorded into {out_json_path}")
        if cs is not None:
            cs.post_progress(generate_progress_data("4 of 4", progress, out_json, chains_count, markups_count))

        out_result_data.append(out_main_result)
        out_result_data.append(create_add_json(input_data, preds, matched_pred_bboxes, frame_times, progress, cs))
        
        return out_result_data

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
        
    Returns: Dict for post_end callback
    """
    inference_result = {}
    try:
        # for each video go inference and create output JSON
        output_files = []
        out_chains_count = 0
        out_markups_count = 0
        file_num = 0
        for json_file in json_files:

            full_json_path = os.path.join(INPUT_DATA_PATH, json_file)

            #TODO: add check if file was already processed
            json_out_files = os.listdir(OUTPUT_PATH)
            if json_out_files:
                json_input_name = json_file.replace("IN_", "OUT_")
                if json_input_name in json_out_files:
                    py_logger.info(f"Current input JSON file is already processed: {json_file}")
                    continue

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
                if frame_times:
                    out_file, out_add_file = create_json_with_predictions(input_data, frame_times, model, progress, cs)
            
                    output_files.append(out_file["out_file"])
                    out_chains_count = out_chains_count + out_file["chains_count"]
                    out_markups_count = out_markups_count + out_file["markups_count"]
                    if out_add_file:
                        output_files.append(out_add_file["out_file"])
                else:
                    py_logger.error(f"Failed to get times for each video frame for {full_file_name}. Probably FFPROBE error. Skip current file, go to the next one.")
                    continue
            else:
                py_logger.error(f"Failed to create predictions for {full_json_path}. File can not be read. Skip current file, go to the next one.")
                continue
            
        if (len(output_files) == 0):
            py_logger.warning("No files have been processed.")
            
        inference_result["out_files"] = output_files
        inference_result["chains_count"] = out_chains_count
        inference_result["markups_count"] = out_markups_count
        return inference_result

    except Exception as e:
        py_logger.exception(f'Exception occurred in inference.inference_mode(): {e}', exc_info=True)
        if cs is not None:
            cs.post_error(generate_error_data(f"Failed to inference. Exception occurred in inference.inference_mode(): {e}"))
        return inference_result
