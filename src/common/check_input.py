import os

from src.workdirs import INPUT_DATA_PATH
from src.common.json_processing import load_json
from src.common.logger import py_logger

def check_input_files_for_training_mode(input_json_files):
    if input_json_files:
        py_logger.info(f"Training program mode")
        check_files_result = False
        for json_file in input_json_files:
            full_json_path = os.path.join(INPUT_DATA_PATH, json_file)
            py_logger.info(f"Check current input JSON file: {full_json_path} for skeletons markup in")
            data = load_json(str(full_json_path))
            check_files_result = False
            if data:
                input_data = data['files'][0]
                skeleton_nodes = input_data['file_chains'][0]['chain_markups'][0]['markup_path'].get('nodes', [])
                if len(skeleton_nodes) > 0:
                    check_files_result = True
                else:
                    check_files_result = False
            else:
                check_files_result = False
        return check_files_result
    else:
        return False