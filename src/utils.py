import torch

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0

def create_masks(src, tgt, device):
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2).int().to(device)
    tgt_seq_len = tgt.shape[1]
    tgt_causal_mask = causal_mask(tgt_seq_len).to(device)
    tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2).int().to(device)
    tgt_mask = tgt_causal_mask & tgt_padding_mask
    return src_mask, tgt_mask