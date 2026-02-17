import torch
from tokenizers import Tokenizer
from model import build_transformer
from utils import create_masks

def translate_robust(model, sentence, tokenizer, max_len=256, device='cuda'):
    model.eval()
    input_ids = tokenizer.encode(sentence).ids
    src = torch.tensor(input_ids).unsqueeze(0).to(device)
    
    sos_token = tokenizer.token_to_id("[SOS]")
    eos_token = tokenizer.token_to_id("[EOS]")
    
    tgt = torch.tensor([[sos_token]]).to(device)
    
    for _ in range(max_len):
        src_mask, tgt_mask = create_masks(src, tgt, device)
        
        with torch.no_grad():
            output = model(src, tgt, src_mask, tgt_mask)
            logits = output[:, -1, :]
            if tgt.size(1) > 1:
                last_token = tgt[0, -1]
                logits[0, last_token] = -float('inf') 
            
            next_token_id = logits.argmax()
            
            if next_token_id.item() == eos_token:
                break
            tgt = torch.cat([tgt, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
    return tokenizer.decode(tgt[0].tolist()[1:])

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = Tokenizer.from_file("java2cs_tokenizer.json")
    vocab_len = tokenizer.get_vocab_size()
    
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
    
    checkpoint = torch.load("java2cs_model_epoch_29.pt", map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)

    java_code = "public static void main(String[] args) { int a = 10; int b = 20; int sum = a - b; System.out.println(sum); }"
    print("Input:", java_code)
    output = translate_robust(model, java_code, tokenizer, device=device)
    print("Output:", output)