import os
import time
import json
import shutil
import random

from super_gradients.training import Trainer
from super_gradients.training.datasets.pose_estimation_datasets import YoloNASPoseCollateFN
from torch.utils.data import DataLoader

from training.train_params import define_train_params, EDGE_LINKS, EDGE_COLORS, KEYPOINT_COLORS
from training.keypoint_transforms import define_set_transformations
from training.pose_estimation_dataset import PoseEstimationDataset
from training.convert import convert_to_coco

import cv2
from workdirs import INPUT_PATH, OUTPUT_PATH, INPUT_DATA_PATH

from common.logger import py_logger
from common.data_classes import OutputData, File, Chain, Markup, MarkupPath, save_class_in_json
from common.json_processing import load_json, save_json
from common.generate_callback_data import generate_error_data, generate_progress_data

from inference.video_processor import get_frame_times, check_video_extension


def extract_frames_from_videos(input_data: dict, video_paths: list, output_folder: str, cs = None):
    """
    Extract frames from videos and save them to the output folder.

    Parameters:
        input_data (dict): Input JSON data for one video
        video_paths (list): List of video file paths.
        output_folder (str): Folder where frames will be saved.
        cs (obj): ContainerStatus object
    """
    try: 
        py_logger.info(f"Extracting has started")
        extract_frames_folder = os.path.join(output_folder, "extracted_frames") 
        if not os.path.exists(extract_frames_folder):
            os.makedirs(extract_frames_folder)

        for video_id, video_ann in enumerate(input_data['files']):
            video_path = video_paths[video_id]
            chains = video_ann["file_chains"]

            if not os.path.exists(video_path):
                py_logger.error(f"Video file not found: {video_path}")
                continue
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                py_logger.error(f"Failed to open video: {video_path}")
                continue

            for chain in chains:
                for markup in chain["chain_markups"]:
                    frame_number = markup["markup_frame"]

                    # Set the video frame position
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

                    # Read the frame
                    ret, frame = cap.read()
                    if not ret:
                        py_logger.error(f"Failed to read frame {frame_number} in {video_path}")
                        continue

                    # Save the frame to the output directory
                    video_name = os.path.splitext(os.path.basename(video_path))[0]
                    output_path = os.path.join(extract_frames_folder, f"{video_name}_{frame_number}.jpg")
                    cv2.imwrite(output_path, frame)
                    py_logger.info(f"Saved: {output_path}")

            py_logger.info(f"{video_path} extracted")

        py_logger.info(f"Extracting completed")

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
    # random.seed(images)
    random.shuffle(images)

    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]

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
    
    return train_folder, val_folder, os.path.join(full_tmp_training_path, "train_coco_ann.json"), os.path.join(full_tmp_training_path, "val_coco_ann.json")




    # frame_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
    # frame_files.sort() 
    
    # if len(frame_files) != len(flattened_frame_times):
    #     raise ValueError("Mismatch between the number of frames and frame times")
    
    # combined = list(zip(frame_files, flattened_frame_times))

    # # Shuffle combined data to maintain alignment
    # random.seed(seed)
    # random.shuffle(combined)

    # # Split into train and validation
    # val_size = int(len(combined) * val_split)
    # val_split_data = combined[:val_size]
    # train_split_data = combined[val_size:]

    # # Unpack train and validation splits
    # train_frames, train_frame_times = zip(*train_split_data)
    # val_frames, val_frame_times = zip(*val_split_data)

    # # Create train and validation folders
    # train_folder = os.path.join(data_folder, "train")
    # val_folder = os.path.join(data_folder, "val")
    # os.makedirs(train_folder, exist_ok=True)
    # os.makedirs(val_folder, exist_ok=True)

    # # Move files to their respective folders
    # for frame in train_frames:
    #     shutil.move(os.path.join(data_folder, frame), os.path.join(train_folder, frame))
    # for frame in val_frames:
    #     shutil.move(os.path.join(data_folder, frame), os.path.join(val_folder, frame))

    # py_logger.info(f"INFO - Split data: {len(train_frames)} train frames, {len(val_frames)} val frames")
    # return train_folder, val_folder, list(train_frame_times), list(val_frame_times)


def training(model, config):
    """
    Train the YOLO-NAS Pose model and save model weights.
    """
    try:
        ckpt_path = os.path.join(config["train_dir"], "checkpoints") 
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        train_transforms, val_transforms = define_set_transformations()
        NUM_EPOCHS = config.get("NUM_EPOCHS", 10)
        WEIGHTS_PATH = config["weights_path"]

        train_dataset = PoseEstimationDataset(
            data_dir=config["train_dir"],
            images_dir=config["train_imgs_dir"],
            json_file=config["train_anns"],
            transforms=train_transforms,
            edge_links=EDGE_LINKS,
            edge_colors=EDGE_COLORS,
            keypoint_colors=KEYPOINT_COLORS,
        )

        val_dataset = PoseEstimationDataset(
            data_dir=config["val_dir"],
            images_dir=config["val_imgs_dir"],
            json_file=config["val_anns"],
            transforms=val_transforms,
            edge_links=EDGE_LINKS,
            edge_colors=EDGE_COLORS,
            keypoint_colors=KEYPOINT_COLORS,
        )

        train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=16, drop_last=True, pin_memory=False, collate_fn=YoloNASPoseCollateFN()
        )
        val_dataloader = DataLoader(
            val_dataset, shuffle=True, batch_size=16, drop_last=False, pin_memory=False, collate_fn=YoloNASPoseCollateFN()
        )

        trainer = Trainer(experiment_name="yolo_nas_pose_run", ckpt_root_dir=ckpt_path)
        train_params = define_train_params(NUM_EPOCHS)

        # py_logger.info("INFO - Starting training...")
        start_time = time.time()
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
        if not os.path.exists(full_tmp_training_path):
            os.makedirs(full_tmp_training_path)

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
                
            if not check_video_extension(file_name):
                py_logger.error(f"Failed to train on {file_name}. File extension doens't match appropriate ones.")
                if cs is not None:
                    cs.post_error(generate_error_data(f"Failed to train on {file_name}. File extension doens't match appropriate ones.", file_name))
                continue  # Skip if file extension doens't match appropriate ones    

            all_json_data["files"].append(input_data)

            all_videos_names.append(full_file_name)

        # Extract frames from all videos
        save_json(all_json_data, os.path.join(full_tmp_training_path,"all_json_data.json"))
        frames_folder = extract_frames_from_videos(all_json_data, all_videos_names, full_tmp_training_path)

        # Convert data to COCO format        
        coco_ann_file = "coco_ann.json"
        coco_data_path, coco_data = convert_to_coco(all_json_data, frames_folder, full_tmp_training_path, coco_ann_file)

        # Split frames into train and validation sets
        train_folder, val_folder, train_coco_ann, val_coco_ann = train_val_split(frames_folder, full_tmp_training_path, coco_data, val_split=0.2)


        # Configure training parameters
        train_config = {
            "weights_path": WEIGHTS_PATH,
            "train_dir": full_tmp_training_path,
            "train_imgs_dir": train_folder,
            "train_anns": train_coco_ann,
            "val_dir": full_tmp_training_path,
            "val_imgs_dir": val_folder,
            "val_anns": val_coco_ann,
            "NUM_EPOCHS": 5,  # Default; can be modified as needed
        }

        # Start training
        training(model, train_config)

    except Exception as e:
        print(f"ERROR - train_mode failed: {e}")
