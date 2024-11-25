import time
import uuid
import os

from tqdm import tqdm
from typing import Type

from prediction_processor import get_prediction_per_frame, compare_bboxes, calculate_chain_vector

from common.json_processing import load_json, save_json
from common.generate_callback_data import generate_error_data, generate_callback_data
from common.logger import py_logger

from workdirs import INPUT_PATH, OUTPUT_PATH, INPUT_DATA_PATH

INPUT_PATH = os.path.join("../", INPUT_PATH)
OUTPUT_PATH = os.path.join("../", OUTPUT_PATH)
INPUT_DATA_PATH = os.path.join("../", INPUT_DATA_PATH)

def check_video_extension(video_path):
    """
    Checks video extension for compliance with the specified ones
    """
    valid_extensions = {"avi", "mp4", "m4v", "mov", "mpg", "mpeg", "wmv"}
    ext = os.path.splitext(video_path)[1][1:].lower()
    return ext in valid_extensions

def create_json_with_predictions(data_file: dict, model, cs, progress, conf: float = 0.6):
    """
    Adds to input JSON model predictions and saves result in output JOSN for each videofile
    
    Arguments:
        data: Input JSON data for one video
        model: Prediction model
        conf: Model confidence
    """
    try:
        file_name = data_file['file_name']
        file_id = data_file['file_id']
        full_file_name = os.path.join(INPUT_PATH, data_file['file_name'])
            
        # Output JSON file name
        out_json = file_name + '.json'
        out_json_path = os.path.join(OUTPUT_PATH, out_json)

        # Output JSON file name for recording new bboxes
        out_json_additional = file_name + '_add.json'
        output_add_file_data = {}
        output_add_file_data['file_name'] = file_name
        output_add_file_data['file_id'] = file_id
        output_add_file_data['file_data'] = []
        out_add_chains = []
        out_json_additional_path = os.path.join(OUTPUT_PATH, out_json_additional)

        # Inference for each frame of video
        cs.post_progress(generate_progress_data(progress, "predicting"))
        preds, inference_time = get_prediction_per_frame(model, full_file_name, conf=conf)

        # Processing inference result
        cs.post_progress(generate_progress_data(progress, "processing"))
        for pred in preds:
            frame = pred['markup_frame']
            predicted_bboxes = pred['processed_prediction']['bboxes']
            predicted_nodes = predicted_vector['processed_prediction']['nodes']
            predicted_edges = predicted_vector['processed_prediction']['edges']
            predicted_scores = predicted_vector['processed_prediction']['scores']

            matched = False

            # Compare input bboxes (markup_path) with predicted by models

            # if MATCHED - put skeleton data in markup_vector and chain_vector, 
            # also calculate markup_frame and markup_time; 
            # chain_name, markup_parent_id, markup_path - from input

            # array of markup_vectors per chain for calculating chain_vector
            for chain in data_file.get('file_chains', []):

                input_chain_vector = chain['chain_vector']

                # array of markup_vectors per chain for calculating chain_vector
                predicted_markup_vectors = []

                for markup in chain.get('chain_markups', []):
                    markup_frame = markup['markup_frame']
                    markup_path = markup['markup_path']

                    # YOLO NAS POSE creates detections per frame
                    # hence, predicted_bboxes is an array of all bboxes per frame
                    for i in range (len(predicted_bboxes)):
                        detected_bbox = predicted_bboxes[i]
                        detected_person_nodes = predicted_nodes[i]
                        if markup_frame == frame:
                            if compare_bboxes(markup_path, detected_bbox):
                                markup_path = {}
                                markup_path['nodes'] = detected_person_nodes
                                markup_path['edges'] = predicted_edges
                                markup['markup_path'] = markup_path
                                markup['markup_vector'] = predicted_scores[i]
                                predicted_markup_vectors.append(predicted_scores[i])
                                matched = True
                                break
                
                chain['chain_vector'] = calculate_chain_vector(input_chain_vector, predicted_markup_vectors)

                if matched:
                    break

            '''
            # if NOT MATCHED - create new chain with all new inner params (except markup_parent_id)
            # record in additional output JSON
            if not matched:
                new_chain_name = int(uuid.uuid4().int >> 64) % 1000000
                new_chain = {
                    'chain_name': new_chain_name,
                    'chain_vector': predicted_vector,
                    'chain_markups': [
                        {
                            'markup_parent_id': '',
                            'markup_frame': frame,
                            'markup_time': frame, #TODO: count time for frame
                            'markup_path': predicted_path,
                            'markup_vector': predicted_vector
                        }
                    ]
                }
                out_add_chains.append(new_chain)
            '''

        # Save new JSONs
        output_add_file_data['file_data'] = out_add_chains

        save_json(data_file, out_json_path)
        # save_json(output_add_file_data, out_json_additional_path)
        py_logger.info(f"Inference results for {file_name} saved and recorded into {out_json_path}")
        print(f"Inference results for {file_name} saved and recorded into {out_json_path}")
        cs.post_progress(generate_progress_data(progress, "processed", out_json_path))

    except Exception as e:
        py_logger.exception(f'Exception occurred in inference.create_json_with_predictions() while inferencing {data_file['file_name']}: {e}', exc_info=True)
        cs.post_error(generate_error_data(f"Failed to inference. Exception occurred in inference.inference_mode(): {err}"))
       
def inference_mode(model, cs, json_files):
    """
    Inference program mode. Processing video and
    adding new data about skeleton to each input JSON and save

    Arguments:
        model: YOLO-NAS Pose downloaded model
        cs: ContainerStatus object
        data: Input json data of video
    """
    try:
        # for each video go inference and create output JSON
        file_num = 0
        for json_file in json_files:

            full_json_path = os.path.join(INPUT_DATA_PATH, json_file)
            # print(full_json_path)
            data = load_json(full_json_path)

            #TODO: Add on_progress
            data_file = data['files'][0]
            file_name = data_file['file_name']
            full_file_name = os.path.join(INPUT_PATH, data_file['file_name'])

            if not os.path.exists(full_file_name) or not check_video_extension(file_name):
                cs.post_error(generate_error_data(f"Failed to create predictions for {file_name}"))
                continue  # Skip if file does not exist or extension doens't match appropriate ones

            # Make and process model predictions for video
            file_num += 1
            progress = round((file_num + 1) / len(json_files) * 100, 2)
            create_json_with_predictions(data_file, model, cs, progress)

    except Exception as e:
        py_logger.exception(f'Exception occurred in inference.inference_mode(): {e}', exc_info=True)
        cs.post_error(generate_error_data(f"Failed to inference. Exception occurred in inference.inference_mode(): {err}"))
