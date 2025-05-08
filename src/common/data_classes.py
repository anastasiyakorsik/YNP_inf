from typing import Dict, List
import json

class Node:
    """
    Describes skeleton keypoints

    Attributes:
        x (float), y (float): Keypoint coordinates
        score (float): Confidence score of keypoint
    """
    def __init__(self, x: float, y: float, score: float):
        self.x = x
        self.y = y
        self.score = score

class MarkupPath:
    """
    Describes bbox and skeleton of a person

    Attributes:
        x (float), y (float), width (float), height (float): Bbox coordinates and size params
        nodes (dict): Structure describing skeleton keypoints {<node_id>: {x: <node_x>, y: <node_y>}, ...}
        edges (dict): Structure describing links between skeleton keypoints {<edge_id>: {from: <src_node_id>, to: <dest_node_id>}, ...}
    """
    x: float
    y: float
    width: float
    height: float
    nodes: dict[str, Node]
    edges: dict[str, dict[str, int]]

class Markup:
    """
    Describes markup for person in current frame

    Attributes:
        markup_id (str): Markup ID
        markup_parent_id (str): Markup Parent ID
        markup_time (float): Time of current video frame 
        markup_frame (int): Video frame number
        markup_vector (list): List of input scores and keypoints scores
        markup_path (MarkupPath): Structure containing information about person bbox and skeleton in current frame
        markup_confidence (float): Mean confidence score for all skeleton nodes
    """
    markup_id: str
    markup_parent_id: str
    markup_time: float
    markup_frame: int
    markup_vector = list[float]
    markup_path: MarkupPath
    markup_confidence: float

class Chain:
    """
    Describes all markups for person (equals chain) in video

    Attributes:
        chain_name (str): Chain name
        chain_vector (list): List of input scores and mean scores of all keypoints in chain
        chain_confidence (float): Chain mean confidence score
        chain_markups (list): Structure containing information about person bboxes and skeletons in video
    """
    chain_name: str
    chain_vector: list[float]
    chain_confidence: float
    chain_markups: list[Markup]

class File:
    """
    Describes file markups

    Attributes:
        file_id (str): File ID
        file_name (str): File name
        file_subset (str): [OPTIONAL] File subset
        file_chains (list): Structure containing information about all people's bboxes and skeletons in video
    """
    def __init__(self, file_id: str, file_name: str, file_chains: list[Chain], file_subset: str = None):
        self.file_id = file_id
        self.file_name = file_name
        self.file_subset = file_subset
        self.file_chains = file_chains


class OutputData:
    files: list[File]

    def __init__(self):
        self.files = []

def recursive_asdict(obj):
    if isinstance(obj, list):
        return [recursive_asdict(o) for o in obj]
    elif isinstance(obj, dict):
        return {key: recursive_asdict(value) for key, value in obj.items()}
    elif hasattr(obj, "__dict__"):
        return {key: recursive_asdict(value) for key, value in obj.__dict__.items()}
    else:
        return obj
            
def save_class_in_json(save_data, save_file_name: str):    
    with open(save_file_name, 'w', encoding='utf-8') as f:
        json.dump(recursive_asdict(save_data), f, ensure_ascii=False, indent=4)
