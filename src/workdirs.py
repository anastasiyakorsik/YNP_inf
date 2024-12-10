import os

MAIN_DIR = ".."

WEIGHTS_DIR = os.path.join(MAIN_DIR, "weights")
WEIGHTS_FILE = "yolo_nas_pose_l_coco_pose.pth"
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, WEIGHTS_FILE)

OUTPUT_PATH = os.path.join(MAIN_DIR, "output")
INPUT_PATH = os.path.join(MAIN_DIR, "input")
INPUT_DATA_PATH = os.path.join(MAIN_DIR, "markups")