import torch
from torch import nn


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
        taco2_loss = mel_loss + gate_loss

        return taco2_loss, mel_loss, gate_loss


def KLD_loss(mu, logvar):
    '''
    References
    - https://github.com/pytorch/examples/blob/master/vae/main.py#L73
    - https://github.com/hwalsuklee/tensorflow-mnist-VAE/blob/master/vae.py#L82
    - https://github.com/hwalsuklee/tensorflow-mnist-VAE/blob/master/vae.py#L85
    '''
    log_lb = 1e-8 # log lower bound
    KLD = 0.5 * torch.sum(logvar.exp() + mu.pow(2) - 1 - torch.log(log_lb + logvar.exp()), dim=1)
    loss_KLD = KLD.mean()

    return loss_KLD
