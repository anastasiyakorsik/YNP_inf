import json
import os
import uuid
from datetime import datetime
from PIL import Image
from common.logger import py_logger
from common.json_processing import load_json, save_json

def info_coco():
    """
    Create 'info' field in COCO format.
    """
    try:
        return {
            "description": "Converted YOLO-NAS-Pose annotations",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().strftime("%Y/%m/%d")
        }
    except Exception as err:
        py_logger.exception(f"Exception occured in training.convert.info_coco() {err=}, {type(err)=}", exc_info=True)
    

def categories_coco():
    """
    Create 'categories' field in COCO format.
    """
    try:
        return [{
            "supercategory": "person",
            "id": 1,
            "name": "person",
            "keypoints": [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"
            ],
            "skeleton": [
                [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], 
                [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], 
                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], 
                [2, 4], [3, 5], [4, 6], [5, 7]
            ]
        }]
    except Exception as err:
        py_logger.exception(f"Exception occured in training.convert.categories_coco() {err=}, {type(err)=}", exc_info=True)
    


def generate_unique_id():
    """
    Generate a unique integer ID.
    """
    try:
        return uuid.uuid4().int & (1 << 64) - 1
    except Exception as err:
        py_logger.exception(f"Exception occured in training.convert.generate_unique_id() {err=}, {type(err)=}", exc_info=True)

def convert_bbox(markup_path):
    """
    Convert bounding box information to COCO format.
    """
    try:    
        return [
            markup_path["x"],
            markup_path["y"],
            markup_path["width"],
            markup_path["height"]
        ]
    except Exception as err:
        py_logger.exception(f"Exception occured in training.convert.convert_bbox() {err=}, {type(err)=}", exc_info=True)



def convert_keypoints(markup_path):
    """
    Convert keypoint annotations to COCO format.
    """
    try:
        keypoints = []
        num_keypoints = 0
        for key, value in markup_path["nodes"].items():
            #key = list(node.keys())[0]
            #point = node[key]
            x, y, score = value["x"], value["y"], value["score"]
            visibility = 2 if score > 0.5 else 1  # Visible or labeled but not visible
            keypoints.extend([x, y, visibility])
            num_keypoints += 1
        return keypoints, num_keypoints
    except Exception as err:
        py_logger.exception(f"Exception occured in training.convert.convert_keypoints() {err=}, {type(err)=}", exc_info=True)


def extract_image_dimensions(image_path):
    """
    Get the dimensions (width, height) of an image.
    """
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as err:
        py_logger.exception(f"Exception occured in training.convert.extract_image_dimensions() {err=}, {type(err)=}", exc_info=True)


def convert_to_coco(input_json, frames_folder, extracted_frames, full_tmp_frames_path, coco_ann_file):
    """
    Convert YOLO-NAS-Pose annotations to COCO format.
    """
    try:
        coco_data = {
            "info": info_coco(),
            "images": [],
            "annotations": [],
            "categories": categories_coco()
        }

        for frame_path in extracted_frames:
            # Extract image metadata
            frame_file = os.path.basename(frame_path)
            video_name, frame_number = frame_file.rsplit('_', 1)
            frame_id = int(frame_number.split('.')[0])
            image_id = generate_unique_id()
            frame_path = os.path.join(frames_folder, frame_file)
            width, height = extract_image_dimensions(frame_path)

            # Add image info
            coco_data["images"].append({
                "id": image_id,
                "file_name": frame_file,
                "width": width,
                "height": height
            })

            # Process annotations for this frame
            for video_data in input_json["files"]:
                video_name_from_json = os.path.splitext(os.path.basename(video_data["file_name"]))[0]
                if video_name_from_json == video_name:
                    for chain in video_data["file_chains"]:
                        for markup in chain["chain_markups"]:
                            if markup["markup_frame"] == frame_id:
                                annotation_id = generate_unique_id()
                                bbox = convert_bbox(markup["markup_path"]) # пересмотреть
                                keypoints, num_keypoints = convert_keypoints(markup["markup_path"]) # переписать  

                                coco_data["annotations"].append({
                                    "id": annotation_id,
                                    "image_id": image_id,
                                    "category_id": 1,  # Person category
                                    "bbox": bbox,
                                    "keypoints": keypoints,
                                    "num_keypoints": num_keypoints,
                                    "iscrowd": 0
                                })
        coco_data_path = os.path.join(os.path.dirname(full_tmp_frames_path), coco_ann_file)
        if coco_data:
            save_json(coco_data, coco_data_path)
        else:
            py_logger.error(f"coco_data IS EMPTY")
        py_logger.info(f"INFO - COCO annotations saved to {coco_data_path}")
        return coco_data_path, coco_data
    except Exception as err:
        py_logger.exception(f"Exception occured in training.convert.convert_to_coco() {err=}, {type(err)=}", exc_info=True)


def main(input_json_path, output_frames_path, output_coco_path):
    """
    Main function to convert YOLO-NAS-Pose annotations to COCO format.
    """
    with open(input_json_path, 'r') as f:
        input_json = json.load(f)

    # Assuming `output_frames_path` contains extracted frame image paths
    extracted_frames = sorted([
        os.path.join(output_frames_path, frame) for frame in os.listdir(output_frames_path)
    ])

    coco_data = convert_to_coco(input_json, extracted_frames)

    # Save COCO annotations
    with open(output_coco_path, 'w') as f:
        json.dump(coco_data, f, indent=4)

    print(f"COCO annotations saved to {output_coco_path}")


if __name__ == "__main__":
    input_json_path = "yolo_nas_pose.json"
    output_frames_path = "output_frames"  # Folder containing extracted frames
    output_coco_path = "coco_annotations.json"
    main(input_json_path, output_frames_path, output_coco_path)
