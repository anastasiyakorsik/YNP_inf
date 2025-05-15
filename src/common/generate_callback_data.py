from typing import Dict 

def generate_error_data(msg: str, details: str = None):
    data = {"msg": msg}
    if details is not None:
        data["details"] = details
    return data 

def generate_progress_data(stage: str, progress: float, output_file: str = None, chains_count = None, markups_count = None, train_msg: str = None, test_error: str = None):
    data = {
        "stage": stage,
        "progress": progress
    }
    if output_file is not None:
        data["statistics"] = {"out_file": output_file, "chains_count": chains_count, "markups_count": markups_count, "train_msg": train_msg, "test_error": test_error}
    return data


