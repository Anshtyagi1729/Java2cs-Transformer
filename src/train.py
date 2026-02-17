import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

from src.model import build_transformer
from dataset import java2csDataset, get_or_build_tokenizer
from utils import create_masks

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Downloading dataset...")
    dataset = load_dataset("google/code_x_glue_cc_code_to_code_trans", split="train")
    
    tokenizer = get_or_build_tokenizer(dataset)
    vocab_len = tokenizer.get_vocab_size()
    
    ds_train = java2csDataset(dataset, tokenizer)
    train_loader = DataLoader(ds_train, batch_size=32, shuffle=True)
    
    model = build_transformer(
        src_vocab_size=vocab_len, 
        tgt_vocab_size=vocab_len,
        src_seq_len=256,
        tgt_seq_len=256,
        d_model=256,  
        N=2,         
        h=4,          
        dropout=0.1,
        d_ff=512
    )
    
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    pad_token_id = tokenizer.token_to_id("[PAD]")
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    num_epochs = 30
    model.train()
    print(f"Starting training on {device}...")
    
    for epoch in range(num_epochs):
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0
        for batch in batch_iterator:
            encoder_input = batch['java'].to(device) 
            tgt = batch['cs'].to(device)           
            decoder_input = tgt[:, :-1]
            ground_truth = tgt[:, 1:]
            
            src_mask, tgt_mask = create_masks(encoder_input, decoder_input, device)
            
            optimizer.zero_grad()
            proj_output = model(encoder_input, decoder_input, src_mask, tgt_mask)
            
            loss = loss_fn(
                proj_output.view(-1, proj_output.shape[-1]), 
                ground_truth.contiguous().view(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f}")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f"java2cs_model_epoch_{epoch}.pt")

if __name__ == "__main__":
    train_model()