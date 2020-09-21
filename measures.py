
import torch
import math
import numpy as np
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
    '''
    #mel_length = torch.max(torch.argmax(alignments[batch_i,:,text_length-last_steps:text_length],dim=0))
    #mel_length = mel_length.item()
    is_positive_output = (gate_output > 0).tolist()
    if True in is_positive_output:
        mel_length = is_positive_output.index(True)
    else:
        mel_length = len(is_positive_output)

    return mel_length

def get_mel_lengths(gate_outputs):
    '''
    Prams
    -----
    gate_output: torch.Tensor.
    - size: [batch_size, max_mel_len].

    Return
    -----
    mel_length: int.
    - Size == [batch_size].
    '''
    batch_size = gate_outputs.size(0)
    mel_lengths = torch.LongTensor(batch_size)
    for i in range(batch_size):
        gate_output = gate_outputs[i,:]
        mel_lengths[i] = get_mel_length(gate_output)

    return mel_lengths


def get_attention_quality(
        batch_forward_attention_ratio,
        batch_multiple_attention_ratio,
        batch_letter_attention_ratio):

    batch_attention_quality = batch_forward_attention_ratio \
        * (1 - batch_multiple_attention_ratio) \
        * batch_letter_attention_ratio

    return batch_attention_quality


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
    alignments = alignments.detach().cpu()
    input_lengths = input_lengths.detach().cpu()

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
        output_lengths=None, gate_outputs=None, mode_mel_length="stop_token",
        top_k=5):
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
    alignments = alignments.detach().cpu()
    input_lengths = input_lengths.detach().cpu()
    text_padded = text_padded.detach().cpu()
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
        #text_string = sequence_to_text(text_sequence.squeeze().tolist())
        text_sequence_list = text_sequence.squeeze().tolist()

        alignment = alignments[i,:mel_length,:text_length]
        #argmax_alignment = torch.argmax(alignment, dim=1)
        argmax_alignment = torch.argsort(alignment, dim=1, descending=True)[:,:top_k]
        argmax_alignment = torch.unique(argmax_alignment.reshape(-1))
        n_unique_argmax = argmax_alignment.size(0)
        sample_attention_ratio = n_unique_argmax / text_length
        batch_attention_ratio[i] = sample_attention_ratio

        argmax_alignment_list = argmax_alignment.tolist()

        cnt_letters = 0
        letter_locations = find_letter_locations(text_sequence_list)
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
        punct_locations = find_punctuation_locations(text_sequence_list)
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
        blank_locations = find_blank_locations(text_sequence_list)
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
        output_lengths=None, gate_outputs=None, mode_mel_length="stop_token",
        top_k=3):
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
    alignments = alignments.detach().cpu()
    input_lengths = input_lengths.detach().cpu()
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
        #argmax_alignment = torch.argmax(alignment, dim=1)
        argmax_alignment = torch.argsort(alignment, dim=1, descending=True)[:,:top_k]
        unique_argmax_set = torch.unique(argmax_alignment)
        range_length = torch.max(unique_argmax_set) - torch.min(unique_argmax_set) + 1
        range_length = range_length.item()
        range_ratio = range_length / text_length
        batch_attention_range_ratio[i] = range_ratio
    mean_attention_range_ratio = batch_attention_range_ratio.mean().item()

    return mean_attention_range_ratio, batch_attention_range_ratio


def multiple_attention_ratio(alignments, input_lengths, text_padded=None,
        output_lengths=None, gate_outputs=None, mode_mel_length="stop_token",
        enc_element='letter', top_k=3):
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
    alignments = alignments.detach().cpu()
    text_padded = text_padded.detach().cpu()
    batch_multiple_attention_ratio = torch.zeros((batch_size), dtype=torch.float)

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
        if text_length > 0:
            sample_multiple_attention_ratio = n_multiple_attention / text_length
        else:
            sample_multiple_attention_ratio = 0
        if math.isnan(sample_multiple_attention_ratio):
            print("sample_multiple_attention_ratio is nan!")
        batch_multiple_attention_ratio[i] = sample_multiple_attention_ratio

    mean_multiple_attention_ratio = batch_multiple_attention_ratio.mean().item()

    return mean_multiple_attention_ratio, batch_multiple_attention_ratio

class SecondStopPredictor():
    def __init__(self, max_decoder_steps, letter_att_top_k=5):
        self.max_decoder_steps = max_decoder_steps
        self.letter_att_top_k = letter_att_top_k
        self.stop_same_forward_steps = 20 # 10 *
        self.lar_lb = 0.95 # the lower bound of letter attention ratio

    def initialize(self, text_padded, input_lengths):
        '''
        Params
        -----
        text_padded: torch.LongTensor. Shape: [batch_size, max_batch_input_len].
        input_lengths: torch.Tenor. A 1-D tensor that keeps input text lengths. Shape: [batch_size].
        '''
        self.text_padded = text_padded.detach().cpu()
        self.batch_size = text_padded.size(0)
        self.max_input_length = text_padded.size(1)
        self.input_lengths = input_lengths.detach().cpu()
        self.dec_step = -1
        self.forward_steps = torch.zeros((self.batch_size)) # forward step: forwarding and remaining steps
        self.backward_steps = torch.zeros((self.batch_size))
        self.prev_forward_steps = torch.zeros((self.batch_size))
        self.prev_backward_steps = torch.zeros((self.batch_size))
        self.prev_argmax_att_w = torch.zeros((self.batch_size))
        self.prev_attention_weights = torch.zeros(
            (self.batch_size, self.max_input_length))
        self.prev_letters_top_k_att_w = list()
        self.att_letter_sets = list()
        self.letter_locations = list()
        #self.max_aq = torch.zeros((self.batch_size))
        self.cnt_same_argmax_att = torch.zeros((self.batch_size))
        self.end_points = [self.max_decoder_steps] * self.batch_size
        self.end_found = [False] * self.batch_size
        self.forward_attention_ratio = torch.zeros((self.batch_size))
        self.letter_attention_ratio = torch.zeros((self.batch_size))

        for i in range(self.batch_size):
            text_length = input_lengths[i].item()
            text_sequence = text_padded[i,:text_length].view(1, -1)
            text_sequence_list = text_sequence.squeeze().tolist()
            self.prev_letters_top_k_att_w.append(set())
            self.att_letter_sets.append(set())
            self.letter_locations.append(set(find_letter_locations(text_sequence_list)))


    def get_attention_measures(self):
        return self.forward_attention_ratio, self.letter_attention_ratio

    def get_end_points(self):
        return self.end_points

    def predict(self, attention_weights):
        '''
        Params
        -----
        attention_weights: Attention map.
        - Type: torch.Tensor.
        - Shape: [batch_size, max_batch_txt_steps].

        Returns
        -----
        end_decoding: Whether to end the decoding loop.
        - Type: bool.
        - If True, then end the decoding loop. Otherwise, keep the loop.
        end_points
        - Type: list. Element type: int.
        - Length: batch_size.
        '''
        self.dec_step += 1
        attention_weights = attention_weights.detach().cpu().float()
        forward_attention_ratio = torch.zeros((self.batch_size))
        letter_attention_ratio = torch.zeros((self.batch_size))

        for i in range(self.batch_size):
            if self.end_found[i]:
                continue
            text_length = self.input_lengths[i]
            attention_weight = attention_weights[i,:text_length]

            #arg_top_k_att_w = torch.argsort(attention_weight, descending=True)[:self.letter_att_top_k].tolist()
            arg_top_k_att_w = torch.topk(attention_weight, self.letter_att_top_k)[1].tolist()
            #argmax_att_w = torch.argmax(attention_weight)
            argmax_att_w = arg_top_k_att_w[0]

            letters_top_k_att_w = self.letter_locations[i].intersection(arg_top_k_att_w)

            one_step_diff = False
            if self.dec_step > 0:
                one_step_diff = True if abs(argmax_att_w - self.prev_argmax_att_w[i]) == 1 else False
                if argmax_att_w >= self.prev_argmax_att_w[i]:
                    self.forward_steps[i] += 1
                else:
                    self.backward_steps[i] += 1
                forward_attention_ratio[i] = self.forward_steps[i] / self.dec_step

                prev_letters_top_k_att_w = self.prev_letters_top_k_att_w[i]
                prev_att_letter_set = self.att_letter_sets[i]

            self.att_letter_sets[i] = self.att_letter_sets[i].union(letters_top_k_att_w)
            letter_attention_ratio[i] = len(self.att_letter_sets[i]) / len(self.letter_locations[i])
            '''
            # Attention quality
            aq = forward_attention_ratio[i] * letter_attention_ratio[i]

            # Record max attention quality
            pre_max_aq = self.max_aq[i]
            if aq > self.max_aq[i]:
                self.max_aq[i] = aq
            '''
            # Count the same argmax attention continuously
            if (argmax_att_w == self.prev_argmax_att_w[i]) and \
                letter_attention_ratio[i] > self.lar_lb and not self.end_found[i]:
                self.cnt_same_argmax_att[i] += 1
            else:
                self.cnt_same_argmax_att[i] = 0


            # Determine the end points of the given batch samples.
            # End point: The end of speech. This should not included in speech.
            # - speech[:self.end_points[i]] is the right range for slicing.
            if (letter_attention_ratio[i] > self.lar_lb and not self.end_found[i]) \
                and max(self.letter_locations[i]) not in arg_top_k_att_w \
                and (self.cnt_same_argmax_att[i] > self.stop_same_forward_steps \
                    or (self.backward_steps[i] > self.prev_backward_steps[i] and not one_step_diff)):
                    # or self.max_aq[i] > aq):
                if self.cnt_same_argmax_att[i] > self.stop_same_forward_steps:
                    print("Stopped by self.cnt_same_argmax_att[i] > self.stop_same_forward_steps")
                if self.backward_steps[i] > self.prev_backward_steps[i] and not one_step_diff:
                    print("Stopped by self.backward_steps[i] > self.prev_backward_steps[i] and not one_step_diff")
                # If an enpoint is found, ...
                self.end_points[i] = self.dec_step
                self.end_found[i] = True

            self.forward_attention_ratio[i] = forward_attention_ratio[i]
            self.letter_attention_ratio[i] = letter_attention_ratio[i]

            self.prev_forward_steps[i] = self.forward_steps[i]
            self.prev_backward_steps[i] = self.backward_steps[i]
            self.prev_argmax_att_w[i] = argmax_att_w
            self.prev_attention_weights[i,:text_length] = attention_weight
            self.prev_letters_top_k_att_w[i] = letters_top_k_att_w

        end_decoding = True if sum(self.end_found) == self.batch_size else False
        end_points = self.end_points

        return end_decoding, end_points
