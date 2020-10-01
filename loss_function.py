import torch
from torch import nn
from utils import get_KLD_weight

class TotalLoss(nn.Module):
    def __init__(self, hparams):
        super(TotalLoss, self).__init__()
        self.hparams = hparams

        self.Taco2Loss_fn = Tacotron2Loss()
        self.L1Loss_fn = nn.L1Loss()
        self.CELoss_fn = nn.CrossEntropyLoss()
        self.MSELoss_fn = nn.MSELoss()

    def forward(self,
            mel_outputs, mel_outputs_postnet, mel_padded,
            gate_outputs, gate_padded,
            y_pred, y,
            mu, logvar,
            prosody_pred, prosody_ref,
            logit_speakers, speakers,
            logit_emotions, emotion_targets,
            att_means, input_lengths,
            iteration):

        # Caculate Tacotron2 losses.
        loss_taco2, loss_mel, loss_gate = self.Taco2Loss_fn(y_pred, y)

        # Calculate the KL-divergence loss.
        if self.hparams.residual_encoder:
            loss_KLD = KLD_loss(mu, logvar)
        else:
            loss_KLD = torch.zeros(1).cuda()

        # Fitting prosody_pred to prosody_ref
        if self.hparams.prosody_predictor:
            fixed_prosody_ref = prosody_ref.detach()
            loss_ref_enc = self.L1Loss_fn(prosody_pred, fixed_prosody_ref)
        else:
            loss_ref_enc = torch.zeros(1).cuda()

        # Forward the speaker adversarial training module.
        if self.hparams.speaker_adversarial_training:
            spk_adv_targets = get_spk_adv_targets(speakers, input_lengths)
            # Compute the speaker adversarial loss
            loss_spk_adv = self.CELoss_fn(logit_speakers, spk_adv_targets)
        else:
            loss_spk_adv = torch.zeros(1).cuda()

        # Forward the emotion adversarial training module.
        if self.hparams.emotion_adversarial_training:
            emo_adv_targets = get_emo_adv_targets(emotion_targets, input_lengths)
            # Compute the emotion adversarial loss
            loss_emo_adv = self.CELoss_fn(logit_emotions, emo_adv_targets)
        else:
            loss_emo_adv = torch.zeros(1).cuda()

        if self.hparams.monotonic_attention:
            loss_att_means = self.MSELoss_fn(att_means, input_lengths.float())
        else:
            loss_att_means = torch.zeros(1).cuda()

        self.hparams.res_en_KLD_weight = get_KLD_weight(iteration, self.hparams)
        loss = loss_taco2 + \
            self.hparams.res_en_KLD_weight * loss_KLD + \
            self.hparams.loss_ref_enc_weight * loss_ref_enc + \
            self.hparams.speaker_adv_weight * loss_spk_adv + \
            self.hparams.emotion_adv_weight * loss_emo_adv + \
            self.hparams.loss_att_means_weight * loss_att_means

        return (loss, loss_taco2, loss_mel, loss_gate, loss_KLD, loss_ref_enc,
            loss_spk_adv, loss_emo_adv, loss_att_means)

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out = model_output[0]
        mel_out_postnet = model_output[1]
        gate_out = model_output[2]

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
