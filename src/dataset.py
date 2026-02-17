import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import os

class java2csDataset(Dataset):
    def __init__(self, ds, tokenizer, max_length=256):
        super().__init__()
        self.ds = ds
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sos_token = tokenizer.token_to_id('[SOS]')
        self.eos_token = tokenizer.token_to_id('[EOS]')
        self.pad_token = tokenizer.token_to_id('[PAD]')

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        java_text = self.ds[idx]['java']
        cs_text = self.ds[idx]['cs']
        
        src_ids = self.tokenizer.encode(java_text).ids
        tgt_ids = self.tokenizer.encode(cs_text).ids
        
        src_tensor = [self.sos_token] + src_ids + [self.eos_token]        
        tgt_tensor = [self.sos_token] + tgt_ids + [self.eos_token]

        pad_len_src = self.max_length - len(src_tensor)
        pad_len_tgt = self.max_length - len(tgt_tensor)

        if pad_len_src > 0: 
            src_tensor = src_tensor + ([self.pad_token] * pad_len_src)
        if pad_len_tgt > 0: 
            tgt_tensor = tgt_tensor + ([self.pad_token] * pad_len_tgt)

        if pad_len_src < 0: 
            src_tensor = src_tensor[:self.max_length]
        if pad_len_tgt < 0: 
            tgt_tensor = tgt_tensor[:self.max_length]

        return {
            "java": torch.tensor(src_tensor, dtype=torch.long),
            "cs": torch.tensor(tgt_tensor, dtype=torch.long)
        }

def get_or_build_tokenizer(dataset, vocab_size=30000):
    if os.path.exists("java2cs_tokenizer.json"):
        print("Tokenizer loaded from file.")
        tokenizer = Tokenizer.from_file("java2cs_tokenizer.json")
    else:
        print("Building Tokenizer...")
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, 
            special_tokens=["[PAD]", "[SOS]", "[EOS]", "[UNK]"]
        )
        
        def get_training_corpus():
            for i in range(0, len(dataset), 1000):
                chunk = dataset[i : i + 1000]
                for text in chunk['java']: yield text
                for text in chunk['cs']: yield text

        tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
        tokenizer.save("java2cs_tokenizer.json")
        print("Tokenizer built and saved!")
    return tokenizer