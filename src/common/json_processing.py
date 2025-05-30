import json
import chardet
import os

from common.logger import py_logger

def collect_json_files(input_files: list):
    input_json_files = []
    for input_file in input_files:
        if os.path.splitext(input_file)[1] == ".json":
            input_json_files.append(input_file)

    return input_json_files


def load_json(input_arg: str) -> dict:
    """
    Loads JSON file
    """
    data = {}
    try:  
        if os.path.isfile(input_arg):
            with open(input_arg, 'rb') as fp:
                raw_data = fp.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']

            with open(input_arg, 'r', encoding=encoding) as f:
                data = json.load(f)
        else:
            data = json.loads(input_arg)
        return data
    except Exception as e:
        py_logger.exception(f'Exception occurred in json_processing.load_json(): {e}', exc_info=True)
        return data

def save_json(data: dict, output_path: str):
    """
    Saves JSON to output file
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        py_logger.exception(f'Exception occurred in json_processing.save_json(): {e}', exc_info=True)
     


