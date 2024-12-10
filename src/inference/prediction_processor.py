import json
import time
import collections
import numpy as np
from tqdm import tqdm

from common.data_classes import Node
from common.logger import py_logger

def calc_mean(*args):
    """
    Calcuates mean value of given vector
    """
    if not args:
        return None
    if len(args) == 1 and isinstance(args[0], collections.Container):
        args = args[0]
    total = sum(args)
    ave = 1.0 * total / len(args)
    return ave

def calculate_chain_vector(input_chain_vector, markup_vectors):
    """
    Calculates chain_vector as the combination of input chain_vector and the average of all markup_vectors

    Arguments:
        input_chain_vector: Chain_vector from input JSON
        markup_vectors: Array of arrays - all markup vectors of chain

    Returns:
        chain_vector: Resulting chain_vector field in output json
    """
    chain_vector = []
    for markup_vector in markup_vectors:
        average_markup_vector = calc_mean(*markup_vector)
        chain_vector.append(average_markup_vector)
    return input_chain_vector + chain_vector

def compare_bboxes(existing_bbox: dict, predicted_bbox: dict, threshold: float = 0.3) -> bool:
    """
    Compares two bboxes (form input JSON (existing_bbox) and predicted_bbox)
    Using Intersection over Union (IoU) for comparison
    """
    try:
        x1, y1, w1, h1 = existing_bbox["x"], existing_bbox["y"], existing_bbox["width"], existing_bbox["height"]
        x2, y2, w2, h2 = predicted_bbox["x"], predicted_bbox["y"], predicted_bbox["width"], predicted_bbox["height"]

        # intersection coordinates
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter_width = max(xi2 - xi1, 0)
        inter_height = max(yi2 - yi1, 0)
        intersection = inter_width * inter_height

        # association area
        union = w1 * h1 + w2 * h2 - intersection

        iou = intersection / union if union != 0 else 0

        return iou >= threshold

    except Exception as err:
        py_logger.exception(f"Exception occured in prediction_processor.compare_bboxes() {err=}, {type(err)=}", exc_info=True)

def predicted_bboxes_processing(predicted_bboxes):
    """
    Processes raw model prediction of bboxes for one frame in video

    Arguments:
        predicted_bboxes: List of all detected bboxes in frame

    Returns:
        processed_bboxes: List of bboxes to compare with input ones
    """
    try:
        processed_bboxes = []
        # cycle for each detected bbox (person) in frame
        for bbox in predicted_bboxes:
            box_coords = {}

            # box_coords["score"] = float(scores[i])
            box_coords["x"] = float(bbox[0])
            box_coords["y"] = float(bbox[1])
            box_coords["width"] = float(bbox[2] - bbox[0])
            box_coords["height"] = float(bbox[3] - bbox[1])

            processed_bboxes.append(box_coords)
        return processed_bboxes

    except Exception as err:
        py_logger.exception(f"Exception occured in prediction_processor.predicted_bboxes_processing {err=}, {type(err)=}", exc_info=True)

def skeletons_processing(poses, edge_links):
    """
    Process raw model prediction of skeletons for one frame in vodeo

    Arguments:
        poses: List of all detected skeletons in frame
        edge_links: Constant list of links between nodes

    Returns:
        frame_nodes: List of nodes of all detected skeletons in frame (one element - one skeleton)
        edges: Processed edge_links
        frame_scores: Array of arrays of each skeleton scores
    """
    try:
        # array of nodes scores for markup_vector
        frame_scores = []
        # array of skeletons nodes
        frame_nodes = []
        # cycle for each detected skeleton (person) in frame
        for i in range(len(poses)):
            
            skeleton_nodes = {}
            skeleton_scores = []
            person = poses[i]
            # cycle for each node in skeleton
            for j in range(len(person)):

                pose = person[j]
                node_coords = Node(x = float(pose[0]), y = float(pose[1]), score = float(pose[2]))
                
                skeleton_scores.append(float(pose[2]))
                skeleton_nodes[f"node_{j}"] = node_coords
            
            frame_scores.append(skeleton_scores)
            frame_nodes.append(skeleton_nodes)


        edges = {}
        for i in range(len(edge_links)):
            edge_nodes = {}

            link = edge_links[i]

            edge_nodes["from"] = int(link[0])
            edge_nodes["to"] = int(link[1])

            edges[f"edge_{i}"] = edge_nodes 

        return frame_scores, frame_nodes, edges
    except Exception as err:
        py_logger.exception(f"Exception occured in prediction_processor.skeleton_processing() {err=}, {type(err)=}", exc_info=True)

def prediction_processing(raw_frame_pred):
    """
    Processing raw model prediction for one frame in video
    Arguments:
        raw_frame_pred: Model prediction for one frame in video
    Returns:
        Processed frame predictions dict
    """
    try:
        processed_prediction = {}
        poses = raw_frame_pred.prediction.poses
        # scores = raw_frame_pred.prediction.scores
        bboxes = raw_frame_pred.prediction.bboxes_xyxy
        edge_links = raw_frame_pred.prediction.edge_links

        processed_prediction['bboxes'] = predicted_bboxes_processing(bboxes)
        processed_prediction['scores'], processed_prediction['nodes'], processed_prediction['edges'] = skeletons_processing(poses, edge_links)

        return processed_prediction

    except Exception as err:
        py_logger.exception(f"Exception occured in prediction_processor.prediction_processing() {err=}, {type(err)=}", exc_info=True)
    
def get_prediction_per_frame(model, file_path: str, conf = 0.6):
    """
    Creates model prediction for video frame
    and processes prediction
    """
    try:
        preds = []
        start_time = time.time()

        model_prediction = model.predict(file_path, conf=conf)
        raw_prediction = [res for res in tqdm(model_prediction._images_prediction_gen, total=model_prediction.n_frames, desc="Processing Video")]

        for i in range(len(raw_prediction)):
            prediction = {}
            frame_id = i
            prediction["markup_frame"] = frame_id
            prediction['processed_prediction'] = prediction_processing(raw_prediction[i])
            preds.append(prediction)

        end_time = time.time()

        inference_time = end_time - start_time

        return preds, inference_time

    except Exception as err:
        py_logger.exception(f"Exception occured in prediction_processor.get_prediction_per_frame() {err=}, {type(err)=}", exc_info=True)
