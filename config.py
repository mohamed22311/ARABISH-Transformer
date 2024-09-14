
from pathlib import Path

def get_config():
    """
    Function that returns a configruation of model training

    Returns: Dict
    """
    return {
        'dataset_name': 'yhavinga/ccmatrix',
        'dataset_dir': './data',
        'batch_size': 8,
        'num_epochs': 20,
        'lr': 10**-4,
        'seq_len': 300,
        'd_model': 512,
        'd_ff': 2048,
        'dropout': 0.1,
        'number_of_layers': 6,
        'number_of_heads': 8,
        'lang_src': 'en',
        'lang_trg': 'ar',
        'model_folder': 'weights',
        'model_basename': 'tmodel_',
        'preload': 'latest',
        'tokenizer_file': 'tokenizer_{0}.json',
        'experiment_name': 'runs/tmodel'
    }

def get_weights_file_path(config, epoch: str):
    """
    Function returns the weights file path 
    """
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pth"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    """
    Function returns the latest weights file path 
    """
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
