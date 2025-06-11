import os
import cv2
import numpy as np

from common.logger import py_logger

OUT_FRAMES = "../out_mark_frames"

def draw_bboxes_and_save(frame, frame_index, out_frame_name, existing_bbox, predicted_bbox, overlap_color=(0, 255, 255)):
    """
    Draws multiple bounding boxes on a video frame and highlights overlap areas.
    
    Parameters:
        frame (numpy.ndarray): Current video frame
        frame_index (int): Frame number
        out_frame_name (str): Output marked frame file name
        existing_bbox (dict): Bbox from input data {"x", "y", "width", "height"}
        predicted_bbox (dict): Bbox predicted by model {"x", "y", "width", "height"}
        overlap_color (tuple): [OPTIONAL] Intersection color
    """
    try:
        overlay = frame.copy()

        # Draw existing bbox (green)
        x1, y1, w1, h1 = map(int, [existing_bbox["x"], existing_bbox["y"], existing_bbox["width"], existing_bbox["height"]])
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)

        # Draw predicted bbox (red)
        x2, y2, w2, h2 = map(int, [predicted_bbox["x"], predicted_bbox["y"], predicted_bbox["width"], predicted_bbox["height"]])
        cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)

        # Calculate intersection area
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter_width = max(xi2 - xi1, 0)
        inter_height = max(yi2 - yi1, 0)

        if inter_width > 0 and inter_height > 0:
            # Draw intersection area
            xi1, yi1, xi2, yi2 = map(int, [xi1, yi1, xi2, yi2])
            cv2.rectangle(overlay, (xi1, yi1), (xi2, yi2), overlap_color, -1)

            # Calculate union area
            intersection_area = inter_width * inter_height
            bbox1_area = w1 * h1
            bbox2_area = w2 * h2
            union_area = bbox1_area + bbox2_area - intersection_area

            # Calculate Intersection over Union param
            iou = 0
            if union_area > 0:
                iou = intersection_area / union_area

            # Draw IoU value next to intersection area        
            text = f"IoU: {iou:.2f}"
            text_x, text_y = xi1, yi1 - 5
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        alpha = 0.4  # Transparency
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Save marked video frame
        if not os.path.exists(OUT_FRAMES):
            os.makedirs(OUT_FRAMES)
        output_path = os.path.join(OUT_FRAMES, out_frame_name)
        cv2.imwrite(output_path, frame)
        # print(f"Save marked video frame in file: {output_path}")

        return frame
    except Exception as e:
        py_logger.exception(f'Exception occurred in draw_bboxes(): {e}', exc_info=True)

def get_bbox_from_keypoints(keypoints_dict):
    """
    Вычисляет ограничивающий прямоугольник (bbox) по ключевым точкам.

    keypoints_dict: словарь вида {"node_0": {"x": float, "y": float}, ...}

    Возвращает: (x_min, y_min, width, height)
    """
    xs = [point["x"] for point in keypoints_dict.values()]
    ys = [point["y"] for point in keypoints_dict.values()]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    width = x_max - x_min
    height = y_max - y_min

    return (x_min, y_min, width, height)

def compute_iou(bbox1, bbox2):
    """
    Calculates Intersection over Union of two bboxes
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def compute_center_distance(bbox1, bbox2):
    """
    Calculates the Euclidean distance between the centers of bboxes
    """
    cx1, cy1 = bbox1["x"] + bbox1["width"] / 2, bbox1["y"] + bbox1["height"] / 2
    cx2, cy2 = bbox2["x"] + bbox2["width"] / 2, bbox2["y"] + bbox2["height"] / 2
    return ((cx1 - cx2)**2 + (cy1 - cy2)**2) ** 0.5

def compute_size_difference(bbox1, bbox2):
    """
    Calculates size difference of two bboxes
    """
    return abs(bbox1["width"] - bbox2["width"]) / bbox1["width"]

def crop_bbox_content(frame, bbox):
    """
    Crop the region of interest (ROI) defined by the bounding box
    """
    x, y, w, h = map(int, [bbox["x"], bbox["y"], bbox["width"], bbox["height"]])
    return frame[y:y+h, x:x+w]

def compare_histograms(img1, img2):
    """
    Compare the color histograms of two images.
    Returns a similarity score (0 to 1, where 1 is identical).
    """
    img1_hist = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    img1_hist = cv2.normalize(img1_hist, img1_hist).flatten()

    img2_hist = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    img2_hist = cv2.normalize(img2_hist, img2_hist).flatten()

    similarity = cv2.compareHist(img1_hist, img2_hist, cv2.HISTCMP_CORREL)
    return similarity

def filter_bboxes_with_content(existing_bbox, predicted_bbox, frame, iou_thresh=0.5, dist_thresh=150, size_thresh=0.2, hist_thresh=0.8):
    """
    Filters predicted bounding boxes to match the same object as the existing bounding box,
    including content analysis (color histogram comparison).
    
    Parameters:
        existing_bbox (dict): Bbox from input data {"x", "y", "width", "height"}
        predicted_bbox (dict): Bbox predicted by model {"x", "y", "width", "height"}
        frame (numpy.ndarray): Current video frame
        iou_thresh (float): [OPTIONAL] Minimum IoU to consider a match
        dist_thresh (float): [OPTIONAL] Maximum distance between centers to consider a match
        size_thresh (float): [OPTIONAL] Maximum relative size difference to consider a match
        hist_thresh (float): [OPTIONAL] Minimum similarity of color histograms to consider a match (0 to 1)
        
    Returns:
        Boolean value describing if bboxes matched or not
    """

    match = False

    # Crop content of the existing bounding box
    existing_content = crop_bbox_content(frame, existing_bbox)
    # Crop content of the predicted bounding box
    predicted_content = crop_bbox_content(frame, predicted_bbox)
    # Compare content similarity using histograms
    hist_similarity = compare_histograms(existing_content, predicted_content)

    iou = compute_iou(
        [existing_bbox["x"], existing_bbox["y"], existing_bbox["width"], existing_bbox["height"]],
        [predicted_bbox["x"], predicted_bbox["y"], predicted_bbox["width"], predicted_bbox["height"]]
    )
    dist = compute_center_distance(existing_bbox, predicted_bbox)
    # size_diff = compute_size_difference(existing_bbox, predicted_bbox)

    if iou >= iou_thresh and dist <= dist_thresh and hist_similarity >= hist_thresh:
        match = True

    return match
