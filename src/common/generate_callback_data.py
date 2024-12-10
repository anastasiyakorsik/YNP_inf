from typing import Dict 

def generate_error_data(msg: str, details: str = None):
    data = {"msg": msg}
    if details is not None:
        data["details"] = details
    return data 

def generate_progress_data(stage: str, progress: float, output_file: str = None):
    data = {
        "stage": stage,
        "progress": progress
    }
    if output_file is not None:
        data["statistics"] = {"out_file": output_file}
    return data
