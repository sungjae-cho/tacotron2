
import torch

#def get_mel_length(alignments, batch_i, text_length, last_steps=5):
def get_mel_length(gate_output):
    '''
    Prams
    -----
    gate_output: torch.Tensor.
    - size: [max_mel_len].

    Return
    -----
    mel_length: int.
    - Size == [batch_size].
    '''
    #mel_length = torch.max(torch.argmax(alignments[batch_i,:,text_length-last_steps:text_length],dim=0))
    #mel_length = mel_length.item()
    is_positive_output = (gate_output > 0).tolist()
    if True in is_positive_output:
        mel_length = is_positive_output.index(True)
    else:
        mel_length = len(is_positive_output)

    return mel_length


def forward_attention_ratio(alignments, text_lengths, gate_outputs, hop_size=1):
    '''
    Params
    -----
    alignments: Attention map. torch.Tensor. Shape: [batch_size, mel_steps, txt_steps]
    text_lengths: torch.Tensor. a 1-D tensor that keeps input text lengths.
    hop_size: int. hopping size to determine increment.

    Returns
    -----
    mean_forward_attention_ratio: float. torch.mean(batch_forward_attention_ratio).
    - The value is the mean of the forward attention ratio of all batch samples.
    batch_forward_attention_ratio: torch.Tensor((batch_size),dtype=torch.float).
    - Each element is the forward attention ratio of each batch sample.
    '''
    # torch.Tensor. Shape: [batch_size, mel_steps]
    max_alignments = torch.argmax(alignments, dim=2)

    pre_alignments = max_alignments[:,:-hop_size]
    post_alignments = max_alignments[:,hop_size:]
    is_increment = (pre_alignments <= post_alignments).type(torch.DoubleTensor)

    batch_size = alignments.size(0)
    batch_forward_attention_ratio = torch.empty((batch_size), dtype=torch.float)
    for i in range(batch_size):
        text_length = text_lengths[i]
        gate_output = gate_outputs[i]
        mel_len = get_mel_length(gate_output)
        if mel_len-hop_size > 0:
            forward_attention_ratio = torch.mean(is_increment[i,:mel_len-hop_size]).item()
        else:
            forward_attention_ratio = 0
        batch_forward_attention_ratio[i] = forward_attention_ratio

    mean_forward_attention_ratio = torch.mean(batch_forward_attention_ratio).item()

    return mean_forward_attention_ratio, batch_forward_attention_ratio
