import torch 
import torch.nn as nn
from dataset.dataset_loader import dataset_loader
from model.build_transformer import build_transformer
from train.run_validation import run_validation
from train.save_model_state import save_model_state
from config import get_config, get_weights_file_path, latest_weights_file_path
from torch.utils.tensorboard import SummaryWriter
from typing import Dict
import wandb
from tqdm import tqdm
from pathlib import Path

def train_model(config: Dict,
                writer_: bool,
                wandb_: bool):
    """
    function used to make instance of the model and train it to the given dataset

    Args:
        config: Dict
            A dictonary of all neccsary vlaues 
        writer_: bool 
            True show results on tensorboard
        wandb_: bool
            True log results on weights&biases

    """
   
    # Define the device through device-agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    # load the training and validation dataloaders and tokenizers 
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg = dataset_loader(conf=config)

    # Instaniate the Transformer model
    model = build_transformer(src_vocab_size=tokenizer_src.get_vocab_size(),
                              trg_vocab_size=tokenizer_trg.get_vocab_size(),
                              src_seq_len=config['seq_len'],
                              trg_seq_len=config['seq_len'],
                              d_model=config['d_model'],
                              N=config['number_of_layers'],
                              h=config['number_of_heads'],
                              dropout=config['dropout'],
                              d_ff=config['d_ff'])
    
    
    # Optimimzer to optimize the weights 
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=config['lr'],
                                 eps=1e-9)

    # Loss function to calculate the loss (ignore the padding token from the loss calculations) (smoothing to add bit of randomness)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'),
                                  label_smoothing=0.1).to(device)
    
    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = None
    if preload == 'latest':
        model_filename = latest_weights_file_path(config)
    elif preload:
        get_weights_file_path(config, preload)
    
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        del state
    else:
        print('No model to preload, starting from scratch')

    if writer_:

        # Tensorboard writer to show summaries of the training 
        writer = SummaryWriter(config['experiment_name'])

    if wandb_:
        # define our custom x axis metric
        wandb.define_metric("global_step")
        # define which metrics will be plotted against it
        wandb.define_metric("validation/*", step_metric="global_step")
        wandb.define_metric("train/*", step_metric="global_step")

    for epoch in range(initial_epoch, config['num_epochs']):
        # clear the GPU memory 
        torch.cuda.empty_cache()
        # put the model in training mode
        model.train()
        # create tqdm bar indicator
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")

        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (batch, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(src=encoder_input,
                                          src_mask=encoder_mask) # (B, seq_len, d_model)
            
            decoder_output = model.decode(encoder_output=encoder_output,
                                          src_mask=encoder_mask,
                                          trg=decoder_input,
                                          trg_mask=decoder_mask) # (B, seq_len, d_model)
            
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_trg.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            if writer_:
                writer.add_scalar('train loss', loss.item(), global_step)
                writer.flush()
            
            if wandb_:
                wandb.log({'train/loss': loss.item(), 'global_step': global_step})

            optimizer.zero_grad(set_to_none=True)

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(model=model,
                       validation_ds=val_dataloader,
                       tokenizer_src=tokenizer_src,
                       tokenizer_trg=tokenizer_trg,
                       max_len=config['seq_len'],
                       device=device,
                       print_msg=lambda msg: batch_iterator.write(msg),
                       global_step=global_step,
                       writer=writer,
                       wandb_=wandb_)
        
        # Save the model at the end of every epoch
        save_model_state(model=model,
                         optimizer=optimizer,
                         global_step=global_step,
                         config=config,
                         epoch=epoch)
