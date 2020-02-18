
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
    argmax_alignments = torch.argmax(alignments, dim=2)

    pre_alignments = argmax_alignments[:,:-hop_size]
    post_alignments = argmax_alignments[:,hop_size:]
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


def attention_ratio(alignments, text_lengths, gate_outputs):
    '''
    Attention ratio is a measure for
    "how much encoding steps are attended over all encoding steps".

    Params
    -----
    alignments: Attention map. torch.Tensor. Shape: [batch_size, mel_steps, txt_steps].
    text_lengths: torch.Tensor. A 1-D tensor that keeps input text lengths.
    gate_outputs: torch.Tensor. Shape: [batch_size, stop_token_seq].
    - A 2-D tensor that is a predicted sequence of the stopping decoding step
    - 0 indicates a signal to generate the next decoding step.
    - 1 indicates a signal to generate this decoding step and stop generating the next stop.

    Returns
    -----
    mean_attention_ratio
    - float. torch.mean(batch_forward_attention_ratio).
    batch_attention_ratio
    - torch.Tensor((batch_size),dtype=torch.float).
    '''
    batch_size = alignments.size(0)
    batch_attention_ratio = torch.empty((batch_size), dtype=torch.float)
    sum_attention_ratio = 0
    for i in range(batch_size):
        text_length = text_lengths[i].item()
        gate_output = gate_outputs[i]
        mel_length = get_mel_length(gate_output)
        alignment = alignments[i,:mel_length,:text_length]
        argmax_alignment = torch.argmax(alignment, dim=1)
        n_unique_argmax = torch.unique(argmax_alignment).size(0)
        sample_attention_ratio = n_unique_argmax / text_length
        batch_attention_ratio[i] = sample_attention_ratio
    mean_attention_ratio = batch_attention_ratio.mean().item()

    return mean_attention_ratio, batch_attention_ratio


def multiple_attention_ratio(alignments, text_lengths, gate_outputs):
    '''
    Multiple attention ratio is a measure for
    "how much encoding steps are attended multiple times over all encoding steps".

    Params
    -----
    alignments: Attention map. torch.Tensor. Shape: [batch_size, mel_steps, txt_steps].
    text_lengths: torch.Tensor. A 1-D tensor that keeps input text lengths.
    gate_outputs: torch.Tensor. Shape: [batch_size, stop_token_seq].
    - A 2-D tensor that is a predicted sequence of the stopping decoding step
    - 0 indicates a signal to generate the next decoding step.
    - 1 indicates a signal to generate this decoding step and stop generating the next stop.

    Returns
    -----
    mean_multiple_attention_ratio
    - float. torch.mean(batch_forward_attention_ratio).
    batch_multiple_attention_ratio
    - torch.Tensor((batch_size),dtype=torch.float).
    '''
    batch_size = alignments.size(0)
    batch_multiple_attention_ratio = torch.empty((batch_size), dtype=torch.float)

    for i in range(batch_size):
        text_length = text_lengths[i].item()
        gate_output = gate_outputs[i]
        mel_length = get_mel_length(gate_output)
        alignment = alignments[i,:mel_length,:text_length]
        argmax_alignment = torch.argmax(alignment, dim=1)
        argmax_alignment = argmax_alignment.tolist()

        for j in range((mel_length-2), -1, -1):
            j_prev = j + 1
            if argmax_alignment[j] == argmax_alignment[j_prev]:
                del argmax_alignment[j_prev]

        n_multiple_attention = 0
        for argmax in set(argmax_alignment):
            if argmax_alignment.count(argmax) > 1:
                n_multiple_attention += 1
        sample_multiple_attention_ratio = n_multiple_attention / text_length
        batch_multiple_attention_ratio[i] = sample_multiple_attention_ratio

    mean_multiple_attention_ratio = batch_multiple_attention_ratio.mean().item()

    return mean_multiple_attention_ratio, batch_multiple_attention_ratio
