from pathlib import Path
import json

def dict_to_json(data_dict: dict, file_path: Path):
    """
    Converts a dictionary to a JSON file and saves it to disk.

    Args:
        data_dict (dict): The dictionary to convert to JSON.
        file_path (Path): The path to the output JSON file.
    """
    with open(file_path, 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)
        
        
def json_to_dict(file_path : Path):
    """
    Loads a JSON file from disk and converts it to a dictionary.

    Args:
        file_path (Path): The path to the input JSON file.
    """
    with open(file_path,'r') as json_file:
        return json.load(json_file)