
import torch
from text import sequence_to_text
from text.find_symbol_location import find_letter_locations, find_punctuation_locations, find_blank_locations

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


def forward_attention_ratio(alignments, input_lengths,
        output_lengths=None, gate_outputs=None, mode_mel_length="stop_token",
        hop_size=1):
    '''
    Params
    -----
    alignments: Attention map. torch.Tensor. Shape: [batch_size, max_batch_mel_steps, max_batch_txt_steps].
    input_lengths: torch.Tenor. A 1-D tensor that keeps input text lengths. Shape: [batch_size].
    output_lengths: torch.Tensor. A 1-D tensor that keeps output mel lengths. Shape: [batch_size].
    gate_outputs: torch.Tensor. Shape: [batch_size, stop_token_seq].
    - A 2-D tensor that is a predicted sequence of the stopping decoding step
    - 0 indicates a signal to generate the next decoding step.
    - 1 indicates a signal to generate this decoding step and stop generating the next stop.
    mode_mel_length: str.
    - "ground_truth"
    -- This option requires output_lengths.
    - "stop_token"
    -- This option requires gate_outputs.
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
        if mode_mel_length == "ground_truth":
            mel_length = output_lengths[i].item()
        if mode_mel_length == "stop_token":
            gate_output = gate_outputs[i]
            mel_length = get_mel_length(gate_output)
            if mel_length == 0:
                batch_forward_attention_ratio[i] = 0
                continue
        if mel_length-hop_size > 0:
            forward_attention_ratio = torch.mean(is_increment[i,:mel_length-hop_size]).item()
        else:
            forward_attention_ratio = 0
        batch_forward_attention_ratio[i] = forward_attention_ratio

    mean_forward_attention_ratio = torch.mean(batch_forward_attention_ratio).item()

    return mean_forward_attention_ratio, batch_forward_attention_ratio


def attention_ratio(alignments, input_lengths, text_padded,
        output_lengths=None, gate_outputs=None, mode_mel_length="stop_token"):
    '''
    Attention ratio is a measure for
    "how much encoding steps are attended over all encoding steps".

    Params
    -----
    alignments: Attention map. torch.Tensor. Shape: [batch_size, max_batch_mel_steps, max_batch_txt_steps].
    input_lengths: torch.Tenor. A 1-D tensor that keeps input text lengths. Shape: [batch_size].
    text_padded: torch.LongTensor. Shape: [batch_size, max_batch_input_len].
    output_lengths: torch.Tensor. A 1-D tensor that keeps output mel lengths. Shape: [batch_size].
    gate_outputs: torch.Tensor. Shape: [batch_size, stop_token_seq].
    - A 2-D tensor that is a predicted sequence of the stopping decoding step
    - 0 indicates a signal to generate the next decoding step.
    - 1 indicates a signal to generate this decoding step and stop generating the next stop.
    mode_mel_length: str.
    - "ground_truth"
    -- This option requires output_lengths.
    - "stop_token"
    -- This option requires gate_outputs.

    Returns
    -----
    mean_attention_ratio
    - float. torch.mean(batch_forward_attention_ratio).
    batch_attention_ratio
    - torch.Tensor((batch_size),dtype=torch.float).
    '''
    batch_size = alignments.size(0)
    batch_attention_ratio = torch.empty((batch_size), dtype=torch.float)
    batch_letter_attention_ratio = torch.empty((batch_size), dtype=torch.float)
    batch_punct_attention_ratio = torch.empty((batch_size), dtype=torch.float)
    batch_blank_attention_ratio = torch.empty((batch_size), dtype=torch.float)
    sum_attention_ratio = 0

    for i in range(batch_size):
        if mode_mel_length == "ground_truth":
            mel_length = output_lengths[i].item()
        if mode_mel_length == "stop_token":
            gate_output = gate_outputs[i]
            mel_length = get_mel_length(gate_output)
            if mel_length == 0:
                batch_attention_ratio[i] = 0
                continue
        text_length = input_lengths[i].item()
        text_sequence = text_padded[i,:text_length].view(1, -1)
        text_string = sequence_to_text(text_sequence.squeeze().tolist())

        alignment = alignments[i,:mel_length,:text_length]
        argmax_alignment = torch.argmax(alignment, dim=1)
        n_unique_argmax = torch.unique(argmax_alignment).size(0)
        sample_attention_ratio = n_unique_argmax / text_length
        batch_attention_ratio[i] = sample_attention_ratio

        argmax_alignment_list = argmax_alignment.tolist()

        cnt_letters = 0
        letter_locations = find_letter_locations(text_string)
        for idx in letter_locations:
            if idx in argmax_alignment_list:
                cnt_letters += 1
                continue
        if len(letter_locations) == 0:
            letter_attention_ratio = 0
        else:
            letter_attention_ratio = cnt_letters / len(letter_locations)
        batch_letter_attention_ratio[i] = letter_attention_ratio

        cnt_punct = 0
        punct_locations = find_punctuation_locations(text_string)
        for idx in punct_locations:
            if idx in argmax_alignment_list:
                cnt_punct += 1
                continue
        if len(punct_locations) == 0:
            punct_attention_ratio = 0
        else:
            punct_attention_ratio = cnt_punct / len(punct_locations)
        batch_punct_attention_ratio[i] = punct_attention_ratio

        cnt_blanks = 0
        blank_locations = find_blank_locations(text_string)
        for idx in blank_locations:
            if idx in argmax_alignment_list:
                cnt_blanks += 1
                continue
        if len(blank_locations) == 0:
            blank_attention_ratio = 0
        else:
            blank_attention_ratio = cnt_blanks / len(blank_locations)
        batch_blank_attention_ratio[i] = blank_attention_ratio

    mean_attention_ratio = batch_attention_ratio.mean().item()
    mean_letter_attention_ratio = batch_letter_attention_ratio.mean().item()
    mean_punct_attention_ratio = batch_punct_attention_ratio.mean().item()
    mean_blank_attention_ratio = batch_blank_attention_ratio.mean().item()

    return ((mean_attention_ratio, batch_attention_ratio),
            (mean_letter_attention_ratio, batch_letter_attention_ratio),
            (mean_punct_attention_ratio, batch_punct_attention_ratio),
            (mean_blank_attention_ratio, batch_blank_attention_ratio))


def attention_range_ratio(alignments, input_lengths,
        output_lengths=None, gate_outputs=None, mode_mel_length="stop_token"):
    '''
    Attention ratio is a measure for
    "how much encoding steps are attended over all encoding steps".

    Params
    -----
    alignments: Attention map. torch.Tensor. Shape: [batch_size, max_batch_mel_steps, max_batch_txt_steps].
    input_lengths: torch.Tenor. A 1-D tensor that keeps input text lengths. Shape: [batch_size].
    output_lengths: torch.Tensor. A 1-D tensor that keeps output mel lengths. Shape: [batch_size].
    gate_outputs: torch.Tensor. Shape: [batch_size, stop_token_seq].
    - A 2-D tensor that is a predicted sequence of the stopping decoding step
    - 0 indicates a signal to generate the next decoding step.
    - 1 indicates a signal to generate this decoding step and stop generating the next stop.
    mode_mel_length: str.
    - "ground_truth"
    -- This option requires output_lengths.
    - "stop_token"
    -- This option requires gate_outputs.

    Returns
    -----
    mean_attention_ratio
    - float. torch.mean(batch_forward_attention_ratio).
    batch_attention_ratio
    - torch.Tensor((batch_size),dtype=torch.float).
    '''
    batch_size = alignments.size(0)
    batch_attention_range_ratio = torch.empty((batch_size), dtype=torch.float)

    for i in range(batch_size):
        if mode_mel_length == "ground_truth":
            mel_length = output_lengths[i].item()
        if mode_mel_length == "stop_token":
            gate_output = gate_outputs[i]
            mel_length = get_mel_length(gate_output)
            if mel_length == 0:
                batch_attention_range_ratio[i] = 0
                continue
        text_length = input_lengths[i].item()
        alignment = alignments[i,:mel_length,:text_length]
        argmax_alignment = torch.argmax(alignment, dim=1)
        unique_argmax_set = torch.unique(argmax_alignment)
        range_length = torch.max(unique_argmax_set) - torch.min(unique_argmax_set) + 1
        range_length = range_length.item()
        range_ratio = range_length / text_length
        batch_attention_range_ratio[i] = range_ratio
    mean_attention_range_ratio = batch_attention_range_ratio.mean().item()

    return mean_attention_range_ratio, batch_attention_range_ratio


def multiple_attention_ratio(alignments, input_lengths,
        output_lengths=None, gate_outputs=None, mode_mel_length="stop_token"):
    '''
    Multiple attention ratio is a measure for
    "how much encoding steps are attended multiple times over all encoding steps".

    Params
    -----
    alignments: Attention map. torch.Tensor. Shape: [batch_size, max_batch_mel_steps, max_batch_txt_steps].
    input_lengths: torch.Tenor. A 1-D tensor that keeps input text lengths. Shape: [batch_size].
    output_lengths: torch.Tensor. A 1-D tensor that keeps output mel lengths. Shape: [batch_size].
    gate_outputs: torch.Tensor. Shape: [batch_size, stop_token_seq].
    - A 2-D tensor that is a predicted sequence of the stopping decoding step
    - 0 indicates a signal to generate the next decoding step.
    - 1 indicates a signal to generate this decoding step and stop generating the next stop.
    mode_mel_length: str.
    - "ground_truth"
    -- This option requires output_lengths.
    - "stop_token"
    -- This option requires gate_outputs.

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
        if mode_mel_length == "ground_truth":
            mel_length = output_lengths[i].item()
        if mode_mel_length == "stop_token":
            gate_output = gate_outputs[i]
            mel_length = get_mel_length(gate_output)
            if mel_length == 0:
                batch_multiple_attention_ratio[i] = 1
                continue
        text_length = input_lengths[i].item()
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
