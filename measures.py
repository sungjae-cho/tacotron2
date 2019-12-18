import torch

def forward_attention_ratio(attn_map, hop_size=5):
    '''
    Params
    -----
    - attn_map: Attention map. torch.Tensor. Shape: [batch_size, mel_steps, txt_steps]
    '''
    # torch.Tensor. Shape: [batch_size, mel_steps]
    max_attn_map = torch.argmax(attn_map, dim=2)

    pre_attn_map = attn_map[:,:-hop_size]
    post_attn_map = attn_map[:,hop_size:]
    is_increment = (pre_attn_map <= post_attn_map).type(torch.DoubleTensor)

    mean_forward_attention_ratio = torch.mean(is_increment).item()
    batch_forward_attention_ratio = torch.mean(is_increment, dim=1)

    return mean_forward_attention_ratio, batch_forward_attention_ratio
