import torch
from torch import nn
from measures import forward_attention_ratio


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss


def forward_attention_loss(alignments, gate_outputs, hop_size=1):
    '''
    Params
    -----
    alignments: Attention map. torch.Tensor. Shape: [batch_size, mel_steps, txt_steps]
    hop_size: int. hopping size to determine increment.

    Returns
    -----
    mean_forward_attention_ratio: float. torch.mean(batch_forward_attention_ratio).
    - The value is the mean of the forward attention ratio of all batch samples.
    batch_forward_attention_ratio: torch.Tensor((batch_size),dtype=torch.float).
    - Each element is the forward attention ratio of each batch sample.
    '''
    _, batch_forward_attention_ratio = forward_attention_ratio(alignments, gate_outputs)
    mean_forward_attention_loss = torch.mean(-torch.log(batch_forward_attention_ratio))

    return mean_forward_attention_loss
