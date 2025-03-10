import json

    
def parse_json(json_path:str)->dict:
    """
    Function to parse a json file and return a dictionary
    Args:
    json_path : str : Path to the json file
    Returns:
    dict : Dictionary containing the json data
    """
    with open(json_path, "r") as file:
        data = json.load(file)

    return data