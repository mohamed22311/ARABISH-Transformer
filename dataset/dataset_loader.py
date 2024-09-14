
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tokenizers import Tokenizer
from datasets import load_dataset, load_from_disk
from dataset.make_or_load_tokenizer import make_or_load_tokenizer
from dataset.BilingualDataset import BilingualDataset
from dataset.max_seq_len import calculate_max_seq_len
from typing import List, Dict, Tuple
from pathlib import Path

def dataset_loader(conf: Dict) -> Tuple:
    """
    Function the loads the raw dataset, split it into train and validation, create tokenizers and tokenize it,
    encopmase the data into PyTorch Dataset and turn it into dataloaders ready for training.

    Args:
        conf: Dict
            configration of the datasets and tokenizers. Example:
            ```bashconf= 
            {
                'lang_src' : 'en',
                'lang_trg` : 'ar',
                'tokenizer_name: 'tokenizer',
                'seq_len' : 200,
                'batch_size': 8,
                'dataset_name': opus
            }```

    Examples:
        tr_dataloader, val_dataloader, src_tokenizer, trg_tokenizer = load_dataset("dataset", config)
    
    Returns:
        out: Tuple(DataLoader, DataLoader, Tokenizer, Tokenizer)
            training dataloader, validation dataloader, source tokenizer, target tokenizer
    """
    dataset_raw = None
    dataset_dir = Path(conf['dataset_dir'])
    if dataset_dir.exists() and dataset_dir.is_dir():
        # Load the dataset from the existing folder
        print("Folder exists. Loading the dataset from the disk...")
        dataset_raw = load_from_disk(str(dataset_dir))
    else:
        print("Folder does not exist. Creating folder and downloading the dataset...")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_raw = load_dataset(conf['dataset_name'], f"{conf['lang_src']}-{conf['lang_trg']}")
        dataset_raw.save_to_disk(str(dataset_dir))


    tokenizer_src = make_or_load_tokenizer(tokenizer_name=conf['tokenizer_file'], 
                           lang=conf['lang_src'],
                           dataset=dataset_raw)
    tokenizer_trg = make_or_load_tokenizer(tokenizer_name=conf['tokenizer_file'], 
                           lang=conf['lang_trg'],
                           dataset=dataset_raw)
    

    train_ds_size = int(0.9 * len(dataset_raw['train']))
    val_ds_size = len(dataset_raw['train']) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(dataset=dataset_raw['train'], lengths=[train_ds_size, val_ds_size])
    
    if conf['seq_len'] is None:
        conf['seq_len'] = calculate_max_seq_len(dataset=dataset_raw,
                                                src_tokenizer=tokenizer_src,
                                                trg_tokenizer=tokenizer_trg,
                                                lang_src=conf['lang_src'],
                                                lang_trg=conf['lang_trg'],
                                                offset=20)
    
    train_dataset = BilingualDataset(datasaet=train_ds_raw,
                                     src_tokenizer=tokenizer_src,
                                     trg_tokenizer=tokenizer_trg,
                                     seq_len=conf['seq_len'])
    val_dataset = BilingualDataset(datasaet=val_ds_raw,
                                     src_tokenizer=tokenizer_src,
                                     trg_tokenizer=tokenizer_trg,
                                     seq_len=conf['seq_len'])
    
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=conf['batch_size'],
                                  shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                  batch_size=1,
                                  shuffle=False)
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg
    
    
