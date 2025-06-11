import uuid
import os
import statistics
import numpy as np

import cv2

from src.inference.prediction_processor import get_prediction_per_frame, calculate_chain_vector
from src.inference.bbox_processing import get_bbox_from_keypoints
from src.inference.video_processor import get_frame_times, check_video_extension
from src.inference.bbox_processing import filter_bboxes_with_content, OUT_FRAMES

from src.common.data_classes import OutputData, File, Chain, Markup, MarkupPath, save_class_in_json
from src.common.json_processing import load_json, save_json
from src.common.pkl_processing import read_pkl_or_default, create_pkl, get_chains_and_markups_pkl_file_names, get_input_pkl_vector_elements_length, get_output_pkl_vector_elements_length
from src.common.generate_callback_data import generate_error_data, generate_progress_data
from src.common.logger import py_logger

from src.workdirs import INPUT_PATH, OUTPUT_PATH, INPUT_DATA_PATH

'''
def create_add_json(input_data: dict, predictions: list, matched_bboxes: list, frame_times: list, progress: float, cs = None):
    """
    Creates additional JSON file with unmatched bboxes and skeletons in new chains

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
'''

def define_vectors_value(entity: str, input_vector_index_or_value, predictions):
    try:
        if isinstance(input_vector_index_or_value, int):
            output_vector_value = predictions if entity == 'markup' else calculate_chain_vector(
                [], predictions)
            return output_vector_value
        else:
            output_vector_value = [] + predictions if entity == 'markup' else calculate_chain_vector(
                [], predictions)
            return output_vector_value
    except Exception as ex:
        py_logger.exception(f"Exception occurred in inference.define_vectors_value_and_index: {ex}", exc_info=True)
        return None, None


def define_vectors_value_and_index(entity: str, input_vector_index_or_value, input_pkl_vectors: list, output_vectors_len: int,
                                   predictions):
    '''
    py_logger.info(f"input_vector_index_or_value == {input_vector_index_or_value}")
    py_logger.info(f"isinstance(input_vector_index_or_value, int) == {isinstance(input_vector_index_or_value, int)}")
    py_logger.info(f"len(input_pkl_vectors) > 0 == {len(input_pkl_vectors) > 0}")
    '''
    try:
        if input_vector_index_or_value is not None:
            if isinstance(input_vector_index_or_value, int):
                if len(input_pkl_vectors) > 0:
                    output_vector_index = input_vector_index_or_value
                    input_pkl_vector_element = input_pkl_vectors[output_vector_index]
                    #if entity == 'chain':
                        # input_vector_index_or_value = input_vector_index_or_value + 1
                        # py_logger.info(f"Element {input_vector_index_or_value} of input chain_vector has len {len(input_pkl_vectors[input_vector_index_or_value])} and value: {input_pkl_vectors[input_vector_index_or_value]}")
                    output_vector_value = input_pkl_vector_element + predictions if entity == 'markup' else calculate_chain_vector(input_pkl_vector_element, predictions)
                    output_vector_parent_id = None
                    return output_vector_index, output_vector_value, output_vector_parent_id
                else:
                    output_vector_index = output_vectors_len
                    output_vector_value = predictions if entity == 'markup' else calculate_chain_vector(
                        input_pkl_vectors, predictions)
                    output_vector_parent_id = input_vector_index_or_value
                    return output_vector_index, output_vector_value, output_vector_parent_id
            else:
                output_vector_index = output_vectors_len
                # output_vector_value = input_vector_index_or_value + predictions if entity == 'markup' else calculate_chain_vector(
                #    input_vector_index_or_value, predictions)
                output_vector_value = [] + predictions if entity == 'markup' else calculate_chain_vector(
                    [], predictions)
                output_vector_parent_id = None
                return output_vector_index, output_vector_value, output_vector_parent_id
        else:
            output_vector_index = output_vectors_len
            output_vector_value = predictions
            output_vector_parent_id = None
            return output_vector_index, output_vector_value, output_vector_parent_id
    except Exception as ex:
        py_logger.exception(f"Exception occurred in inference.define_vectors_value_and_index: {ex}", exc_info=True)
        return None, None, None


def create_out_markup_if_bboxes_matched(predicted_person_nodes_per_frame, predicted_person_scores_per_frame,
                                        predicted_person_edges_per_frame, input_markup, output_pkl_markup_vectors_len,
                                        frame, frame_times):
    try:
        input_markup_path = input_markup['markup_path']

        out_markup_path = MarkupPath()
        out_markup_path.x, out_markup_path.y, out_markup_path.width, out_markup_path.height = \
            input_markup_path['x'], input_markup_path['y'], input_markup_path['width'], \
                input_markup_path['height']
        out_markup_path.nodes = predicted_person_nodes_per_frame
        out_markup_path.edges = predicted_person_edges_per_frame

        out_markup = Markup()
        out_markup.markup_parent_id = input_markup.get('markup_id', '')
        out_markup.markup_time = frame_times[frame]
        out_markup.markup_frame = frame
        out_markup.markup_path = out_markup_path

        mean_conf_score = statistics.fmean(predicted_person_scores_per_frame)
        out_markup.markup_confidence = mean_conf_score

        output_markup_vector_value = define_vectors_value(
            'markup', input_markup.get('markup_vector', []), predicted_person_scores_per_frame)

        out_markup.markup_vector = output_pkl_markup_vectors_len

        return mean_conf_score, out_markup, output_markup_vector_value
    except Exception as ex:
        py_logger.exception(f"Exception occurred in inference.create_out_markup_if_bboxes_matched: {ex}", exc_info=True)
        return None, None, None


# refactor
def collect_file_chains(input_data: dict, model_predictions_per_video: list, video_cap, full_video_file_name: str,
                        input_pkl_markups_vectors, input_pkl_chains_vectors, frame_times):
    chains_count = 0
    markups_count = 0
    file_chains = []
    matched_predicted_bboxes = []
    output_pkl_markups_vectors = []
    output_pkl_chains_vectors = []

    try:
        for input_chain in input_data.get('file_chains', []):

            matched_chain_bboxes = 0
            chain_markups = []

            # array of markup_vectors per chain for calculating chain_vector
            predicted_markup_vectors = []
            # array of markup_confidence per chain for calculating chain_confidence
            predicted_markup_confidence = []

            for input_markup in input_chain.get('chain_markups', []):
                markups_count += 1
                markup_frame = input_markup.get('markup_frame', None)
                if markup_frame is None:
                    markup_time = round(input_markup["markup_time"], 2)
                    markup_frame = min(range(len(frame_times)), key=lambda i: abs(frame_times[i] - markup_time))

                current_frame_predictions = model_predictions_per_video[int(markup_frame)]

                predicted_bboxes = current_frame_predictions['processed_prediction']['bboxes']
                predicted_nodes = current_frame_predictions['processed_prediction']['nodes']
                predicted_edges = current_frame_predictions['processed_prediction']['edges']
                predicted_scores = current_frame_predictions['processed_prediction']['scores']

                video_cap.set(cv2.CAP_PROP_POS_FRAMES, int(markup_frame))
                ret, cur_frame_obj = video_cap.read()
                if not ret:
                    py_logger.error(f"Error: Could not read frame {int(markup_frame)} from video {full_video_file_name}")
                    continue

                x = input_markup['markup_path'].get("x", None)
                y = input_markup['markup_path'].get("y", None)
                width = input_markup['markup_path'].get("width", None)
                height = input_markup['markup_path'].get("height", None)

                if x is None or y is None or width is None or height is None:
                    x, y, width, height = get_bbox_from_keypoints(input_markup['markup_path']["nodes"])

                (input_markup['markup_path']["x"], input_markup['markup_path']["y"],
                 input_markup['markup_path']["width"], input_markup['markup_path']["height"]) = x, y, width, height

                input_markup_path = input_markup['markup_path']

                # YOLO NAS POSE creates detections per frame
                # hence, predicted_bboxes is an array of all bboxes per frame
                i = next(
                    (
                        i for i in range(len(predicted_bboxes))
                        if
                        len(predicted_scores[i]) == 17 and filter_bboxes_with_content(input_markup_path,
                                                                                      predicted_bboxes[i],
                                                                                      cur_frame_obj)
                    ),
                    None
                )

                if i is not None:
                    '''
                    mean_conf_score, out_markup, output_markup_vector_value = create_out_markup_if_bboxes_matched(
                        predicted_nodes[i], predicted_scores[i], predicted_edges, input_markup,
                        input_pkl_markups_vectors, len(output_pkl_markups_vectors), int(markup_frame), frame_times)
                    '''

                    mean_conf_score, out_markup, output_markup_vector_value = create_out_markup_if_bboxes_matched(
                        predicted_nodes[i], predicted_scores[i], predicted_edges, input_markup,
                        len(output_pkl_markups_vectors), int(markup_frame), frame_times)

                    matched_chain_bboxes += 1
                    matched_predicted_bboxes.append(predicted_bboxes[i])
                    chain_markups.append(out_markup)
                    output_pkl_markups_vectors.append(output_markup_vector_value)
                    # py_logger.info(f"Current output_pkl_markups_vectors len: {len(output_pkl_markups_vectors)}")
                    predicted_markup_vectors.append(predicted_scores[i])
                    predicted_markup_confidence.append(mean_conf_score)
                else:
                    py_logger.warning(
                        "Model nodes prediction scores for current markup are not for 17 keypoints or none bboxes matched.")

            if len(predicted_markup_vectors) > 0:
                input_chain_index = input_chain.get(f'chain_vector', [])

                # py_logger.info(f"Calculating chain_pkl_vector - predicted_markup_vectors len: {len(predicted_markup_vectors)}; input_pkl_chains_vectors len: {len(input_pkl_chains_vectors)}")
                # py_logger.info(f"Chain vector index: {input_chain_index}")
                # py_logger.info(f"Element {input_chain_index} of input_pkl_chains_vectors is: {input_pkl_chains_vectors[input_chain_index]}")

                '''
                output_chain_vector_index, output_chain_vector_value, _ = define_vectors_value_and_index(
                    'chain',  input_chain_index, input_pkl_chains_vectors, len(output_pkl_chains_vectors),
                    predicted_markup_vectors)
                '''

                output_chain_vector_value = define_vectors_value('chain', input_chain_index,
                                                                 predicted_markup_vectors)

                out_chain = Chain()
                chains_count += 1
                out_chain.chain_name = input_chain.get('chain_name', '')
                out_chain.chain_vector = len(output_pkl_chains_vectors)
                out_chain.chain_markups = chain_markups

                output_pkl_chains_vectors.append(output_chain_vector_value)
                # py_logger.info(f"Current output_pkl_chains_vectors len: {len(output_pkl_chains_vectors)}")
                # get_output_pkl_vector_elements_length('current_chain', output_pkl_chains_vectors)

                # print(predicted_markup_confidence)
                if predicted_markup_confidence:
                    out_chain.chain_confidence = statistics.fmean(predicted_markup_confidence)

                if out_chain.chain_markups:
                    file_chains.append(out_chain)

            py_logger.info(f"Have {matched_chain_bboxes} matched bboxes for chain {input_chain['chain_name']}")
    except Exception as ex:
        py_logger.exception(f"Exception occurred in inference.collect_file_chains: {ex}", exc_info=True)

    return chains_count, markups_count, file_chains, matched_predicted_bboxes, output_pkl_markups_vectors, output_pkl_chains_vectors


def get_input_pkl_vectors_names(video_file_name):
    # Create .pkl file name for chains and markups
    pkl_chains_vectors_file_name, pkl_markups_vectors_file_name = get_chains_and_markups_pkl_file_names(video_file_name)

    input_pkl_chains_vectors, input_pkl_markups_vectors = read_pkl_or_default(
        str(os.path.join(INPUT_DATA_PATH, pkl_chains_vectors_file_name))), read_pkl_or_default(
        str(os.path.join(INPUT_DATA_PATH, pkl_markups_vectors_file_name)))

    # get_input_pkl_vector_elements_length('chains', input_pkl_chains_vectors)
    # get_input_pkl_vector_elements_length('markups', input_pkl_markups_vectors)

    #input_pkl_chains_vector_ind_0 = input_pkl_chains_vectors.tolist()[0]
    #py_logger.info(f"input_pkl_chains_vector_ind_0: {input_pkl_chains_vector_ind_0}")

    return pkl_chains_vectors_file_name, pkl_markups_vectors_file_name, input_pkl_chains_vectors, input_pkl_markups_vectors

def create_json_with_predictions(input_data: dict, frame_times: list, model, progress: float, container_status=None,
                                 conf: float = 0.6):
    """
    Adds to input JSON model predictions and saves result in output JOSN for each videofile

    Arguments:
        input_data (dict): Input JSON data for one video
        frame_times (list): Calculated times of video frames
        model: Prediction model
        progress (float): Progress percent
        container_status (obj): ContainerStatus object
        conf (float): [OPTIONAL] Model confidence

    Returns:
        Names of main and additional output files
    """
    out_result_data = []
    try:
        video_file_name = os.path.basename(input_data['file_name'])
        full_video_file_name = os.path.join(INPUT_PATH, video_file_name)

        # pkl_chains_vectors_file_name, pkl_markups_vectors_file_name, input_pkl_chains_vectors, input_pkl_markups_vectors = get_input_pkl_vectors_names(video_file_name)
        pkl_chains_vectors_file_name, pkl_markups_vectors_file_name = get_chains_and_markups_pkl_file_names(
            video_file_name)

        # Inference for each frame of video
        if container_status is not None:
            container_status.post_progress(generate_progress_data(f"Создание предсказаний модели", progress))
        model_predictions, inference_time = get_prediction_per_frame(model, str(full_video_file_name), conf=conf)

        # Processing inference result
        if container_status is not None:
            container_status.post_progress(generate_progress_data("Обработка полученных предсказаний и входных данных", progress))

        # open video to record marked frames
        cap = cv2.VideoCapture(str(full_video_file_name))
        if not cap.isOpened():
            py_logger.error(f"Error: Could not open video {full_video_file_name}.")
            return {}

        # create dir for detected frames
        if not os.path.exists(OUT_FRAMES):
            os.makedirs(OUT_FRAMES)

        '''
        if not isinstance(input_pkl_markups_vectors, list):
            input_pkl_markups_vectors = input_pkl_markups_vectors.tolist()
        if not isinstance(input_pkl_chains_vectors, list):
            input_pkl_chains_vectors = input_pkl_chains_vectors.tolist()
            
        chains_count, markups_count, file_chains, matched_predicted_bboxes, output_pkl_markups_vectors, output_pkl_chains_vectors = collect_file_chains(
            input_data, model_predictions, cap, str(full_video_file_name), input_pkl_markups_vectors,
            input_pkl_chains_vectors, frame_times)
        '''

        chains_count, markups_count, file_chains, matched_predicted_bboxes, output_pkl_markups_vectors, output_pkl_chains_vectors = collect_file_chains(
            input_data, model_predictions, cap, str(full_video_file_name), [],
            [], frame_times)

        # get_output_pkl_vector_elements_length('chains', output_pkl_chains_vectors)
        # get_output_pkl_vector_elements_length('markups', output_pkl_markups_vectors)

        # Save chains and markups vectors in .pkl files
        create_pkl(str(os.path.join(OUTPUT_PATH, pkl_chains_vectors_file_name)), output_pkl_chains_vectors)
        create_pkl(str(os.path.join(OUTPUT_PATH, pkl_markups_vectors_file_name)), output_pkl_markups_vectors)

        py_logger.info(
            f"Chains vectors for {video_file_name} saved and recorded into {os.path.join(OUTPUT_PATH, pkl_chains_vectors_file_name)}")
        py_logger.info(
            f"Markups vectors for {video_file_name} saved and recorded into {os.path.join(OUTPUT_PATH, pkl_markups_vectors_file_name)}")
        '''
        if container_status is not None:
            container_status.post_progress(generate_progress_data("3 of 4", progress, pkl_chains_vectors))
            container_status.post_progress(generate_progress_data("3 of 4", progress, pkl_markups_vectors))
        '''

        output_file_data = File(file_id=input_data.get('file_id', ''), file_name=video_file_name, file_chains=file_chains)

        output_data = OutputData()
        output_data.files.append(output_file_data)

        # Output JSON file name
        out_json = 'OUT_' + video_file_name + '.json'
        out_json_path = os.path.join(OUTPUT_PATH, out_json)

        save_class_in_json(output_data, str(out_json_path))

        py_logger.info(f"Inference results for {video_file_name} saved and recorded into {out_json_path}")

        out_main_result = {"out_file": out_json, "chains_count": chains_count, "markups_count": markups_count}

        # print(f"Inference results for {file_name} saved and recorded into {out_json_path}")
        if container_status is not None:
            container_status.post_progress(
                generate_progress_data("Запись и сохранение результатов", progress, out_json, chains_count, markups_count))

        out_result_data.append(out_main_result)
        # out_result_data.append(
        #   create_add_json(input_data, preds, matched_pred_bboxes, frame_times, progress, container_status))

        return out_main_result

    except Exception as e:
        py_logger.exception(f'Exception occurred in inference.create_json_with_predictions(): {e}', exc_info=True)
        if container_status is not None:
            container_status.post_error(generate_error_data(
                f"Failed to inference. Exception occurred in inference.create_json_with_predictions(): {e}", f"Processing file {input_data['file_name']}"))


def inference_mode(model, json_files: list, container_status = None):
    """
    Inference program mode. Processing video and
    adding new data about skeleton to each input JSON and save

    Arguments:
        model: YOLO-NAS Pose downloaded model
        json_files (list): List of all json files in input directory
        container_status (obj): [OPTIONAL] ContainerStatus object
        
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

            '''
            #add check if file was already processed
            json_out_files = os.listdir(OUTPUT_PATH)
            if json_out_files:
                json_input_name = json_file.replace("IN_", "OUT_")
                if json_input_name in json_out_files:
                    py_logger.info(f"Current input JSON file is already processed: {json_file}")
                    continue
            '''
            py_logger.info(f"Processing current input JSON file: {full_json_path}")
            data = load_json(full_json_path)
            
            if data:
                input_data = data['files'][0]
                file_name = os.path.basename(input_data['file_name'])
                full_file_name = os.path.join(INPUT_PATH, file_name)
                py_logger.info(f"Processing current video file: {full_file_name}")

                if not os.path.exists(full_file_name):
                    py_logger.error(f"Failed to create predictions for {file_name}. File does not exist.")
                    if container_status is not None:
                        container_status.post_error(generate_error_data(f"Failed to create predictions for {file_name}. File does not exist.", f"Processing file {file_name}"))
                    continue  # Skip if file does not exist
                
                if not check_video_extension(file_name):
                    py_logger.error(f"Failed to create predictions for {file_name}. File extension doesn't match appropriate ones.")
                    if container_status is not None:
                        container_status.post_error(generate_error_data(f"Failed to create predictions for {file_name}. File extension doesn't match appropriate ones.", f"Processing file {file_name}"))
                    continue  # Skip if file extension doesn't match appropriate ones

                # Make and process model predictions for video
                file_num += 1
                progress = round((file_num) / len(json_files) * 100, 2)

                if container_status is not None and file_num == 1:
                    container_status.post_progress(
                        generate_progress_data(f"Создание предсказаний модели", 0))

                    # Processing inference result
                    container_status.post_progress(
                        generate_progress_data("Обработка полученных предсказаний и входных данных", 0))

                    container_status.post_progress(
                        generate_progress_data("Запись и сохранение результатов", 0))

                frame_times = get_frame_times(full_file_name)
                if frame_times:
                    out_file = create_json_with_predictions(input_data, frame_times, model, progress, container_status)
            
                    output_files.append(out_file["out_file"])
                    out_chains_count = out_chains_count + out_file["chains_count"]
                    out_markups_count = out_markups_count + out_file["markups_count"]
                    '''
                    if out_add_file:
                        output_files.append(out_add_file["out_file"])
                    '''
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
        if container_status is not None:
            container_status.post_error(generate_error_data(f"Failed to inference. Exception occurred in inference.inference_mode(): {e}"))
        return inference_result
