import os
import time
import json
import shutil
import random

from super_gradients.training import Trainer
from super_gradients.training.datasets.pose_estimation_datasets import YoloNASPoseCollateFN
from super_gradients.training.datasets.pose_estimation_datasets import COCOPoseEstimationDataset
from torch.utils.data import DataLoader

from training.train_params import define_train_params, EDGE_LINKS, EDGE_COLORS, KEYPOINT_COLORS
from training.keypoint_transforms import define_set_transformations
from training.convert import convert_to_coco

import cv2
from workdirs import INPUT_PATH, OUTPUT_PATH, INPUT_DATA_PATH

from common.logger import py_logger
from common.json_processing import load_json, save_json
from common.generate_callback_data import generate_error_data, generate_progress_data
from sklearn.model_selection import train_test_split
from inference.video_processor import get_frame_times, check_video_extension

def check_input_file_for_skeleton_markup_containing(input_json_data):
    try:
        if input_json_data:
            input_data = input_json_data['files'][0]
            skeleton_nodes = input_data['file_chains'][0]['chain_markups'][0]['markup_path'].get('nodes', [])
            if len(skeleton_nodes) > 0:
                check_file_result = True
            else:
                check_file_result = False
        else:
            check_file_result = False
        return check_file_result
    except Exception as ex:
        return False

def extract_frames_from_videos(video_paths: list, output_folder: str, cs = None):
    """
    Extract frames from videos and save them to the output folder.

    Parameters:
        video_paths (list): List of video file paths.
        output_folder (str): Folder where frames will be saved.
        cs (obj): ContainerStatus object
    """
    try: 
        py_logger.info(f"Extracting has started")
        extract_frames_folder = os.path.join(output_folder, "extracted_frames") 
        os.makedirs(extract_frames_folder, exist_ok=True)

        frame_count = 0
        for video_path in video_paths:

            if not os.path.exists(video_path):
                py_logger.error(f"Video file not found: {video_path}")
                continue
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                py_logger.error(f"Failed to open video: {video_path}")
                continue

            video_name = os.path.basename(video_path)

            success, frame = cap.read()
            frame_idx = 0

            while success:
                frame_path = os.path.join(extract_frames_folder, f"{video_name}_{frame_idx}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_count += 1
                success, frame = cap.read()
                frame_idx += 1

            py_logger.info(f"{frame_idx} frames from {video_name} has been extracted. Total amount of extracted frames: {frame_count}")
            cap.release()

        py_logger.info(f"Extracting completed. Total amount of extracted frames: {frame_count}")

        return extract_frames_folder
    
    except Exception as err:
        py_logger.exception(f"Exception occured in training.train.extract_frames_from_videos() {err=}, {type(err)=}", exc_info=True)

def train_val_split(images_path: str, full_tmp_training_path: str, coco_data: dict, split_ratio: float = 0.2):
    """
    Split the extracted frames into training and validation sets.

    Parameters:
        images_dir (str): Folder containing extracted frames. -- frames_path
        split_ratio (float): Fraction of data to use for validation.
        seed (int): Random seed for reproducibility.
    Returns:
        tuple: Paths to training and validation folders, and corresponding train/val frame times.
    """

    images = coco_data['images']
    random.shuffle(images)

    split_index = int(len(images) * split_ratio)
    train_images = images[split_index:]
    val_images = images[:split_index]

    train_annotations = {"info": coco_data['info'], "images": train_images, "annotations": [], "categories": coco_data['categories']}
    val_annotations = {"info": coco_data['info'], "images": val_images, "annotations": [], "categories": coco_data['categories']}

    # Create separate annotation files
    train_img_ids = {img['id'] for img in train_images}
    for ann in coco_data['annotations']:
        if ann['image_id'] in train_img_ids:
            train_annotations['annotations'].append(ann)
        else:
            val_annotations['annotations'].append(ann)
    
    # Create train and validation folders
    train_folder = os.path.join(full_tmp_training_path, "train")
    val_folder = os.path.join(full_tmp_training_path, "val")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # save train and validation annotations
    save_json(train_annotations, os.path.join(full_tmp_training_path, "train_coco_ann.json"))
    save_json(val_annotations, os.path.join(full_tmp_training_path, "val_coco_ann.json"))

    for img in train_images:
        shutil.copyfile(os.path.join(images_path, img['file_name']), os.path.join(train_folder, img['file_name']))
    for img in val_images:
        shutil.copyfile(os.path.join(images_path, img['file_name']), os.path.join(val_folder, img['file_name']))

    py_logger.info(f"INFO - Split data: {len(train_images)} train frames, {len(val_images)} val frames")

    return os.path.relpath(train_folder, full_tmp_training_path), os.path.relpath(val_folder, full_tmp_training_path), "train_coco_ann.json", "val_coco_ann.json"

def split_pose_dataset(coco_data: dict, train_path: str, val_fraction: float):
    """
    Split the pose dataset into training and validation sets.

    :param coco_data: all JSON annotations already in COCO pose format.
    : 
    :param annotation_file: Path to the JSON file containing all annotations.
    :param train_path: Path where the all training files located.
    :param val_annotation_file: Path where the annotations for the validation set will be saved.
    :param val_fraction: Fraction of the total dataset to be used for validation.
    """
    # Open and load the entire annotations JSON file
    annotation = coco_data

    train_annotation_path = os.path.join(train_path, "train_anns.json")
    py_logger.info(f"Train anns file path is: {train_annotation_path}")
    val_annotation_path = os.path.join(train_path, "val_anns.json")
    py_logger.info(f"Val anns file path is: {val_annotation_path}")

    # Extract image IDs from the annotations
    image_ids = [ann["id"] for ann in annotation["images"]]



    # Split the dataset into training and validation sets
    train_ids, val_ids = train_test_split(image_ids, test_size=val_fraction)

    # Prepare annotations dictionary for training set
    
    train_annotations = {
        "info": annotation["info"],
        "categories": annotation["categories"],
        "images": [image for image in annotation["images"] if image["id"] in train_ids],
        "annotations": [ann for ann in annotation["annotations"] if ann["image_id"] in train_ids],
    }

    # Prepare annotations dictionary for validation set
    val_annotations = {
        "info": annotation["info"],
        "categories": annotation["categories"],
        "images": [image for image in annotation["images"] if image["id"] in val_ids],
        "annotations": [ann for ann in annotation["annotations"] if ann["image_id"] in val_ids],
    }

    # Save the annotations for the training set to a JSON file
    with open(train_annotation_path, "w") as f:
        json.dump(train_annotations, f)
        py_logger.info(f"Train annotations saved to {train_annotation_path}")
        py_logger.info(f"Train images: {len(train_ids)}")
        py_logger.info(f"Train annotations len: {len(train_annotations['annotations'])}")
        # py_logger.info(f"Train annotations data: {train_annotations}")

    # Save the annotations for the validation set to a JSON file
    with open(val_annotation_path, "w") as f:
        json.dump(val_annotations, f)
        py_logger.info(f"Val annotations saved to {val_annotation_path}")
        py_logger.info(f"Val images: {len(val_ids)}")
        py_logger.info(f"Val annotations len: {len(val_annotations['annotations'])}")
        # py_logger.info(f"Val annotations data: {val_annotations}")

    return train_annotation_path, val_annotation_path


def training(model, config):
    """
    Train the YOLO-NAS Pose model and save model weights.
    """
    try:
        ckpt_path = os.path.join(config["train_dir"], "checkpoints") 
        os.makedirs(ckpt_path, exist_ok=True)

        py_logger.info(f"INFO - Training chekpoint path created")

        train_transforms, val_transforms = define_set_transformations()
        NUM_EPOCHS = config.get("NUM_EPOCHS", 10)
        WEIGHTS_PATH = config["weights_path"]

        py_logger.info(f"INFO - Train data transformations defined")

        train_dataset = COCOPoseEstimationDataset(
            data_dir=config["train_dir"],
            images_dir=config["train_imgs_dir"],
            json_file=config["train_anns"],
            transforms=train_transforms,
            edge_links=EDGE_LINKS,
            edge_colors=EDGE_COLORS,
            keypoint_colors=KEYPOINT_COLORS,
            include_empty_samples=False
        )

        val_dataset = COCOPoseEstimationDataset(
            data_dir=config["val_dir"],
            images_dir=config["val_imgs_dir"],
            json_file=config["val_anns"],
            transforms=val_transforms,
            edge_links=EDGE_LINKS,
            edge_colors=EDGE_COLORS,
            keypoint_colors=KEYPOINT_COLORS,
            include_empty_samples=False
        )

        py_logger.info(f"INFO - Train datasets created")

        train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=16, drop_last=True, pin_memory=False, collate_fn=YoloNASPoseCollateFN()
        )
        val_dataloader = DataLoader(
            val_dataset, shuffle=True, batch_size=16, drop_last=False, pin_memory=False, collate_fn=YoloNASPoseCollateFN()
        )

        py_logger.info(f"INFO - Train dataloaders created")

        trainer = Trainer(experiment_name="yolo_nas_pose_run", ckpt_root_dir=ckpt_path)

        py_logger.info(f"INFO - Train class created")

        train_params = define_train_params(NUM_EPOCHS)

        # py_logger.info("INFO - Starting training...")
        start_time = time.time()
        py_logger.info(f"INFO - Train process starting")
        trainer.train(model, training_params=train_params, train_loader=train_dataloader, valid_loader=val_dataloader)
        py_logger.info(f"INFO - Training completed in {time.time() - start_time:.2f}s")

        best_weights_path = os.path.join(os.path.join(ckpt_path, "yolo_nas_pose_run"), "ckpt_best.pth")

        shutil.copy(best_weights_path, WEIGHTS_PATH)
        py_logger.info(f"INFO - Weights are updated")

    except Exception as e:
        py_logger.error(f"ERROR - Training failed: {e}")




def train_mode(model, json_files: list, WEIGHTS_PATH, cs = None):
    """
    Main training function.

    Arguments:
        model: YOLO-NAS Pose downloaded model.
        json_files (list): List of all JSON files in the input directory.
    """
    try:
        full_tmp_training_path = os.path.join(OUTPUT_PATH, "training")
        os.makedirs(full_tmp_training_path, exist_ok=True)
        py_logger.info(f"Training output path is: {full_tmp_training_path}")

        all_videos_names = []
        all_json_data = {"files": []}

        # Load and combine data from all JSON files
        for json_file in json_files:

            full_json_path = os.path.join(INPUT_DATA_PATH, json_file)
            py_logger.info(f"Processing current input JSON file: {full_json_path}")
            data = load_json(full_json_path)

            input_data = data['files'][0]
            file_name = os.path.basename(input_data['file_name'])
            full_file_name = os.path.join(INPUT_PATH, file_name)
            py_logger.info(f"Processing current video file: {full_file_name}")

            if not os.path.exists(full_file_name):
                py_logger.error(f"Failed to train on {file_name}. File does not exist.")
                if cs is not None:
                    cs.post_error(generate_error_data(f"Failed to train on {file_name}. File does not exist.", file_name))
                continue  # Skip if file does not exist

            if not check_input_file_for_skeleton_markup_containing(data):
                py_logger.error(f"Failed to train on {file_name}. File does not contain skeleton markup.")
                if cs is not None:
                    cs.post_error(
                        generate_error_data(f"Failed to train on {file_name}. File does not contain skeleton markup.", file_name))
                continue  # Skip if file does not exist
                
            if not check_video_extension(file_name):
                py_logger.error(f"Failed to train on {file_name}. File extension doens't match appropriate ones.")
                if cs is not None:
                    cs.post_error(generate_error_data(f"Failed to train on {file_name}. File extension doens't match appropriate ones.", file_name))
                continue  # Skip if file extension doens't match appropriate ones    

            all_json_data["files"].append(input_data)

            all_videos_names.append(full_file_name)

        # Extract frames from all videos
        all_json_data_file = os.path.join(full_tmp_training_path, "all_json_data.json")
        save_json(all_json_data, all_json_data_file)
        py_logger.info(f"All combined anns file path is: {all_json_data_file}")

        frames_folder = extract_frames_from_videos(all_videos_names, full_tmp_training_path)
        py_logger.info(f"Extracted frames folder is: {frames_folder}")

        # Convert data to COCO format        
        coco_ann_file = "coco_ann.json"
        
        extracted_frames = []
        for file in os.listdir(frames_folder):
            extracted_frames.append(os.path.join(frames_folder, file))

        coco_data_path, coco_data = convert_to_coco(all_json_data, extracted_frames, full_tmp_training_path, coco_ann_file)
        py_logger.info(f"All combined converted anns file path is: {coco_data_path}")

        # Split frames into train and validation sets
        # train_folder, val_folder, train_coco_ann, val_coco_ann = train_val_split(frames_folder, full_tmp_training_path, coco_data)

        # split_pose_dataset(coco_data_path, "/kaggle/working/train_anns.json", "/kaggle/working/val_anns.json", 0.2)
        train_coco_ann, val_coco_ann = split_pose_dataset(coco_data, full_tmp_training_path, 0.2)


        # Configure training parameters
        train_config = {
            "weights_path": WEIGHTS_PATH,
            "train_dir": full_tmp_training_path,
            "train_imgs_dir": "extracted_frames",
            "train_anns": "train_anns.json",
            "val_dir": full_tmp_training_path,
            "val_imgs_dir": "extracted_frames",
            "val_anns": "val_anns.json",
            "NUM_EPOCHS": 5,  # Default; can be modified as needed
        }

        # Start training
        # py_logger.info(f"Train config: {train_config}")
        training(model, train_config)

    except Exception as err:
        py_logger.exception(f"Exception occured in training.train.train_mode() {err=}, {type(err)=}", exc_info=True)
