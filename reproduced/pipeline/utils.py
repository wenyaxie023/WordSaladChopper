import json

def load_json(file_path: str):
    """Load a JSON file and return its content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json(file_path: str, data):
    """Save data to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)