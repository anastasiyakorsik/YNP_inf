# from dataclasses import dataclass, field, asdict
from typing import Dict, List
import json

class Node:
    def __init__(self, x: float, y: float, score: float):
        self.x = x
        self.y = y
        self.score = score

class MarkupPath:
    nodes: dict[str, Node]
    edges: dict[str, dict[str, int]]

class Markup:
    markup_id: str
    markup_parent_id: str
    markup_time: float
    markup_frame: int
    markup_vector = list[float]
    markup_path: MarkupPath

class Chain:
    chain_id: str
    chain_name: str
    chain_vector: list[float]
    chain_dataset_id: int
    chain_markups: list[Markup]

class File:
    def __init__(self, file_id: int, file_name: str, file_subset: str = None):
        self.file_id = file_id
        self.file_name = file_name
        self.file_subset = file_subset
        
    file_chains: list[Chain] 

class OutputData:
    files: list[File]


def recursive_asdict(obj):
    if isinstance(obj, list):
        return [recursive_asdict(o) for o in obj]
    elif isinstance(obj, dict):
        return {key: recursive_asdict(value) for key, value in obj.items()}
    elif hasattr(obj, "__dict__"):
        return {key: recursive_asdict(value) for key, value in obj.__dict__.items()}
    else:
        return obj
            
def save_class_in_json(save_data, save_file_name):    
    with open(save_file_name, 'w') as f:
        json.dump(recursive_asdict(save_data), f, indent=4)
