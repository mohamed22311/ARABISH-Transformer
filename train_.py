import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_config, get_weights_file_path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import warnings
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute encoder output and reuse it for every token we get from the decoder 
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with sos token 
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device) # B, tokens_of_decoder_input
    while True:
        if decoder_input.size(1) == max_len:
            break
        # biuld a mask for the target (decoder_input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        # Calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get the next token
        prob = model.project(out[:,-1])
        _, next_word = torch.max(prob,dim=1)
        # Add the next word to the next decoder input
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval()
    count = 0
    source_texts = []
    expected = []
    predicted = []
    console_width = 80

    with torch.inference_mode():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            assert encoder_input.size(0) == 1, 'Batch size must be 1 for validation'
            model_out  = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            print_msg('-'*console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')

            if count == num_examples:
                break
    if writer:
        pass


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer =Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    
    # Build a tokenizer 
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    
    # train test split
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0
     
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max Length of source sentence: {max_len_src}\nMax Length of Target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(dataset=train_ds,
                                  batch_size=config['batch_size'],
                                  shuffle=True)
    val_dataloader = DataLoader(dataset=val_ds,
                                  batch_size=1,
                                  shuffle=True)
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(src_vocab_size=vocab_src_len, tgt_vocab_size=vocab_tgt_len, src_seq_len=config['seq_len'], tgt_seq_len=config['seq_len'], d_model=config['d_model'])
    return model

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    Path(f"{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config,tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr'], eps=1e-9)
    
    
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading Model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # B, seq_len
            decoder_input = batch['decoder_input'].to(device) # B, seq_len
            encoder_mask = batch['encoder_mask'].to(device) # B, 1, 1, seq_len
            decoder_mask = batch['decoder_mask'].to(device) # B, 1, seq_len, seq_len

            encoder_output = model.encode(encoder_input, encoder_mask) # B, seq_len, d_model
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # B, seq_len, d_model
            proj_output = model.project(decoder_output) # B, seq_len, tgt_vocab_size
            label = batch['label'].to(device) # B, seq_len

            # B, seq_len, tgt_vocab_size ---> B*seq_len, tgt_vocab_size 
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({'loss': f'{loss.item():6.3f}'})

            writer.add_scalar('Train_loss', loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch, 
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        },model_filename)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)