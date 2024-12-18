import time
import uuid
import os
from tqdm import tqdm
from typing import Type

from inference.prediction_processor import get_prediction_per_frame, compare_bboxes, calculate_chain_vector
from inference.video_processor import get_frame_times, check_video_extension

from common.data_classes import OutputData, File, Chain, Markup, MarkupPath, save_class_in_json
from common.json_processing import load_json, save_json
from common.generate_callback_data import generate_error_data, generate_progress_data
from common.logger import py_logger

from workdirs import INPUT_PATH, OUTPUT_PATH, INPUT_DATA_PATH



def create_json_with_predictions(input_data: dict, frame_times, model, cs, progress, conf: float = 0.6):
    """
    Adds to input JSON model predictions and saves result in output JOSN for each videofile
    
    Arguments:
        input_data: Input JSON data for one video
        frame_times: Calculated times of video frames
        model: Prediction model
        cs: ContainerStatus object
        progress: Progress percent 
        conf: [OPTIONAL] Model confidence
        
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

        output_file_data = File(file_id = input_data['file_id'], file_name = file_name)
        output_file_data.file_chains = []
        
        # Output JSON file name for recording new bboxes
        out_json_add = file_name + '_add.json'
        out_json_add_path = os.path.join(OUTPUT_PATH, out_json_add)

        output_add_data = OutputData()
        output_add_data.files = []

        output_add_file_data = File(file_id = input_data['file_id'], file_name = file_name)
        output_add_file_data.file_chains = []

        # Inference for each frame of video
        cs.post_progress(generate_progress_data("predicting", progress))
        preds, inference_time = get_prediction_per_frame(model, full_file_name, conf=conf)

        # Processing inference result
        cs.post_progress(generate_progress_data("processing", progress))

        # Compare input bboxes (markup_path) with predicted by models

        # if MATCHED - put skeleton data in markup_vector and chain_vector, 
        # also calculate markup_frame and markup_time; 
        # chain_name, markup_parent_id, markup_path - from input

        # array of markup_vectors per chain for calculating chain_vector
        for chain in input_data.get('file_chains', []):
            	
            matched_bboxes = 0
            # print("Processing chain "+ chain['chain_name'])

            # create new chain with input and added info to save if bboxes matched
            out_chain = Chain()
            # out_chain.chain_id = chain['chain_id'] 
            out_chain.chain_name = chain['chain_name']
            # out_chain.chain_dataset_id = chain['chain_dataset_id']
            out_chain.chain_markups = []
                
            # create new chain with new info to save if bboxes didn't match
            out_add_chain = Chain()
            # out_add_chain.chain_id = chain['chain_id']
            out_add_chain.chain_name = str(int(uuid.uuid4().int >> 64) % 1000000)
            # out_add_chain.chain_dataset_id = chain['chain_dataset_id']
            out_add_chain.chain_markups = []


            input_chain_vector = chain['chain_vector']

            # array of markup_vectors per chain for calculating chain_vector
            predicted_markup_vectors = []

            for markup in chain.get('chain_markups', []):
                out_markup = Markup()

                complete_step = False

                for pred in preds:
                    frame = pred['markup_frame']
                    predicted_bboxes = pred['processed_prediction']['bboxes']
                    predicted_nodes = pred['processed_prediction']['nodes']
                    predicted_edges = pred['processed_prediction']['edges']
                    predicted_scores = pred['processed_prediction']['scores']

                    # out_markup.markup_id = 0
                    out_markup.markup_time = frame_times[frame]
                    out_markup.markup_frame = frame

                    markup_frame = markup['markup_frame']
                    input_markup_path = markup['markup_path']

                    # YOLO NAS POSE creates detections per frame
                    # hence, predicted_bboxes is an array of all bboxes per frame
                    for i in range (len(predicted_bboxes)):
                        detected_bbox = predicted_bboxes[i]
                        detected_person_nodes = predicted_nodes[i]

                        out_markup_path = MarkupPath()
                        out_markup_path.nodes = detected_person_nodes
                        out_markup_path.edges = predicted_edges
                        
                        out_markup.markup_path = out_markup_path

                        predicted_markup_vectors.append(predicted_scores[i])

                        if compare_bboxes(input_markup_path, detected_bbox):
                            if markup_frame == frame or markup_frame == "NODATA":
                            
                                #print("BBoxes matched")
                                matched_bboxes += 1

                                out_markup.markup_parent_id = markup['markup_id']
                                out_markup.markup_vector = markup['markup_vector'] + predicted_scores[i]

                                out_chain.chain_markups.append(out_markup)
                                complete_step = True
                                break
                        else:
                            #print("BBoxes not matched")
                            # if NOT MATCHED - create new chain with all new inner params (except markup_parent_id)
                            # record in additional output JSON
                            out_markup.markup_id = str(int(uuid.uuid4().int >> 64) % 1000000)
                            out_markup.markup_vector = predicted_scores[i]

                            out_add_chain.chain_markups.append(out_markup)
                            complete_step = True
                            break

                    if complete_step:
                        break

                
                out_chain.chain_vector = calculate_chain_vector(input_chain_vector, predicted_markup_vectors)
                out_add_chain.chain_vector = calculate_chain_vector([], predicted_markup_vectors)
                    
            py_logger.info(f"Have {matched_bboxes} matched bboxes for chain {chain['chain_name']}")
            
            if out_chain.chain_markups:
                output_file_data.file_chains.append(out_chain)
            if out_add_chain.chain_markups:
                output_add_file_data.file_chains.append(out_add_chain)

        # Save new JSONs
        output_data.files.append(output_file_data)
        output_add_data.files.append(output_add_file_data)

        save_class_in_json(output_data, out_json_path)
        save_class_in_json(output_add_data, out_json_add_path)

        py_logger.info(f"Inference results for {file_name} saved and recorded into {out_json_path} and {out_json_add_path}")
        # print(f"Inference results for {file_name} saved and recorded into {out_json_path}")
        cs.post_progress(generate_progress_data("processed", progress, out_json_path))
        cs.post_progress(generate_progress_data("processed", progress, out_json_add_path))
        
        return out_json_path, out_json_add_path

    except Exception as e:
        py_logger.exception(f'Exception occurred in inference.create_json_with_predictions(): {e}', exc_info=True)
        cs.post_error(generate_error_data(f"Failed to inference. Exception occurred in inference.inference_mode(): {e}"))
       
def inference_mode(model, cs, json_files):
    """
    Inference program mode. Processing video and
    adding new data about skeleton to each input JSON and save

    Arguments:
        model: YOLO-NAS Pose downloaded model
        cs: ContainerStatus object
        data: Input json data of video
        
    Returns: List of output files names
    """
    try:
        # for each video go inference and create output JSON
        output_files = []
        file_num = 0
        for json_file in json_files:

            full_json_path = os.path.join(INPUT_DATA_PATH, json_file)
            py_logger.info(f"Processing current input JSON file: {full_json_path}")
            # print(full_json_path)
            data = load_json(full_json_path)

            input_data = data['files'][0]
            file_name = os.path.basename(input_data['file_name'])
            full_file_name = os.path.join(INPUT_PATH, file_name)
            py_logger.info(f"Processing current video file: {full_file_name}")
            # print(full_file_name)

            if not os.path.exists(full_file_name):
                py_logger.error(f"Failed to create predictions for {file_name}. File does not exist.")
                cs.post_error(generate_error_data(f"Failed to create predictions for {file_name}. File does not exist.", file_name))
                continue  # Skip if file does not exist
                
            if not check_video_extension(file_name):
                py_logger.error(f"Failed to create predictions for {file_name}. File extension doens't match appropriate ones.")
                cs.post_error(generate_error_data(f"Failed to create predictions for {file_name}. File extension doens't match appropriate ones.", file_name))
                continue  # Skip if file extension doens't match appropriate ones

            # Make and process model predictions for video
            file_num += 1
            progress = round((file_num) / len(json_files) * 100, 2)
            print(progress)

            frame_times = get_frame_times(full_file_name)
            out_file, out_add_file = create_json_with_predictions(input_data, frame_times, model, cs, progress)
            
            output_files.append(out_file)
            output_files.append(out_add_file)
            
            
        if (len(output_files) == 0):
            py_logger.warning("No files have been processed.")
            
        return output_files

    except Exception as e:
        py_logger.exception(f'Exception occurred in inference.inference_mode(): {e}', exc_info=True)
        cs.post_error(generate_error_data(f"Failed to inference. Exception occurred in inference.inference_mode(): {e}"))
