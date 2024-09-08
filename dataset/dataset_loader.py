
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tokenizers import Tokenizer
from datasets import load_dataset
from make_or_load_tokenizer import make_or_load_tokenizer
from BilingualDataset import BilingualDataset
from max_seq_len import calculate_max_seq_len
def dataset_loader(dataset_name: str,
                 conf: Dict) -> Tuple(DataLoader, DataLoader, Tokenizer, Tokenizer):
    """
    Function the loads the raw dataset, split it into train and validation, create tokenizers and tokenize it,
    encopmase the data into PyTorch Dataset and turn it into dataloaders ready for training.

    Args:
        dataset_name: str 
            the name of the Hugging Face dataset should be downloaded and loaded.
        conf: Dict
            configration of the datasets and tokenizers. Example:
            ```bashconf= 
            {
                'src_lang' : 'en',
                'trg_lang` : 'ar',
                'tokenizer_name: 'tokenizer',
                'seq_len' : 200,
                'batch_size': 8
            }```

    Examples:
        tr_dataloader, val_dataloader, src_tokenizer, trg_tokenizer = load_dataset("dataset", config)
    
    Returns:
        out: Tuple(DataLoader, DataLoader, Tokenizer, Tokenizer)
            training dataloader, validation dataloader, source tokenizer, target tokenizer
    """
    dataset_raw = load_dataset(dataset_name, f"{conf['src_lang']}-{conf['trg_lang']}")

    tokenizer_src = make_or_load_tokenizer(tokenizer_name=conf['tokenizer_name'], 
                           lang=conf['src_lang'],
                           dataset=dataset_raw)
    tokenizer_trg = make_or_load_tokenizer(tokenizer_name=conf['tokenizer_name'], 
                           lang=conf['trg_lang'],
                           dataset=dataset_raw)
    

    train_ds_size = int(0.9 * len(dataset_raw))
    val_ds_size = len(dataset_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(dataset=dataset_raw, lengths=[train_ds_size, val_ds_size])
    
    conf['seq_len'] = calculate_max_seq_len(dataset=dataset_raw,
                                            src_tokenizer=tokenizer_src,
                                            trg_tokenizer=tokenizer_trg,
                                            src_lang=conf['src_lang'],
                                            trg_lang=conf['trg_lang'],
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
    
    