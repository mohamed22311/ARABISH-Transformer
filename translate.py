from pathlib import Path
from config import get_config, latest_weights_file_path
from model.build_transformer import build_transformer
from tokenizers import Tokenizer
from datasets import load_dataset
from dataset.BilingualDataset import BilingualDataset
import torch
import sys

def translate(sentence: str):
    """
    Function used to make predictions on custom inputs. 
    The function takes a sentence in English and output Arabic translation of it.

    Args:
        sentence: str
            English sentence to be translated
    
    Example: 
        arabic = translate('I want to go to school')

    Returns: 
        out: str
            Arabic translation
    """
    # Define the device, tokenizers, and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    config = get_config()

    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_trg = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_trg']))))
    
    model = build_transformer(src_vocab_size=tokenizer_src.get_vocab_size(),
                              trg_vocab_size=tokenizer_trg.get_vocab_size(),
                              src_seq_len=config["seq_len"],
                              trg_seq_len=config['seq_len'],
                              d_model=config['d_model']).to(device)

    # Load the pretrained weights
    model_filename = latest_weights_file_path(config)
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])

    # if the sentence is a number use it as an index to the test set
    label = ""
    if type(sentence) == int or sentence.isdigit():
        id = int(sentence)
        ds = load_dataset(f"{config['dataset_name']}", f"{config['lang_src']}-{config['lang_trg']}", split='all')
        ds = BilingualDataset(ds, tokenizer_src, tokenizer_trg, config['seq_len'])
        sentence = ds[id]['src_text']
        label = ds[id]["trg_text"]
    seq_len = config['seq_len']

    # translate the sentence
    model.eval()
    with torch.no_grad():
        # Precompute the encoder output and reuse it for every generation step
        source = tokenizer_src.encode(sentence)
        source = torch.cat(
            tensors=[
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64), 
            torch.tensor(source.ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2), dtype=torch.int64)
        ], dim=0).to(device)

        source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
        
        encoder_output = model.encode(source, source_mask)

        # Initialize the decoder input with the sos token
        decoder_input = torch.empty(1, 1).fill_(tokenizer_trg.token_to_id('[SOS]')).type_as(source).to(device)

        # Print the source sentence and target start prompt
        if label != "": print(f"{f'ID: ':>12}{id}") 
        print(f"{f'SOURCE: ':>12}{sentence}")
        if label != "": print(f"{f'TARGET: ':>12}{label}") 
        print(f"{f'PREDICTED: ':>12}", end='')

        # Generate the translation word by word
        while decoder_input.size(1) < seq_len:
            # build mask for target and calculate output
            decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int).type_as(source_mask).to(device)
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

            # project next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                tensors=[
                    decoder_input,
                    torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)
                ], 
                dim=1
            )

            # print the translated word
            print(f"{tokenizer_trg.decode([next_word.item()])}", end=' ')

            # break if we predict the end of sentence token
            if next_word == tokenizer_trg.token_to_id('[EOS]'):
                break

    # convert ids to tokens
    return tokenizer_trg.decode(decoder_input[0].tolist())
    
#read sentence from argument
translate(sys.argv[1] if len(sys.argv) > 1 else "I am not a very good a student.")
