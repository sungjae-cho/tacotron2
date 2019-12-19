import random
import torch
import numpy as np
import wandb
import sys
sys.path.append('waveglow/')
import warnings
warnings.filterwarnings('ignore')
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy
from measures import forward_attention_ratio
from text import sequence_to_text
from denoiser import Denoiser


class Tacotron2Logger(SummaryWriter):
    def __init__(self, run_name, prj_name, logdir, resume):
        self.run_name = run_name
        wandb.init(name=run_name, project=prj_name, resume=resume)
        super(Tacotron2Logger, self).__init__(logdir)
        self.waveglow = self.load_waveglow('/data2/sungjaecho/pretrained/waveglow_256channels_ljs_v2.pt')

    def load_waveglow(self, waveglow_path):
        waveglow = torch.load(waveglow_path)['model']
        waveglow.cuda().eval().half()
        for k in waveglow.convinv:
            k.float()

        return waveglow

    def mel2wav(self, mel_outputs_postnet, with_denoiser=False):
        with torch.no_grad():
            audio = self.waveglow.infer(mel_outputs_postnet, sigma=0.666)
        if with_denoiser:
            np_wav = denoiser(audio, strength=0.01)[:, 0].cpu().numpy()
        else:
            np_wav = audio[0].data.cpu().numpy()

        return np_wav


    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     y_pred, iteration, epoch):
            self.add_scalar("training.loss", reduced_loss, iteration)
            self.add_scalar("grad.norm", grad_norm, iteration)
            self.add_scalar("learning.rate", learning_rate, iteration)
            self.add_scalar("duration", duration, iteration)

            # wandb log
            wandb.log({"epoch": epoch,
                       "training.loss": reduced_loss,
                       "grad.norm": grad_norm,
                       "learning.rate": learning_rate,
                       "duration": duration})

            _, mel_outputs, gate_outputs, alignments = y_pred

            hop_list = [10, 20, 50, 100]
            for hop_size in hop_list:
                mean_far, batch_far = forward_attention_ratio(alignments, hop_size)
                log_name = "mean_forward_attention_ratio(hop_size={})".format(hop_size)
                wandb.log({log_name:mean_far})
                log_name = "forward_attention_ratio(hop_size={})".format(hop_size)
                wandb.log({log_name:wandb.Histogram(batch_far.data.cpu().numpy())})


    def log_validation(self,
        reduced_loss, model, x, y, y_pred, iteration, epoch, sample_rate):
        text_padded, input_lengths, mel_padded, max_len, output_lengths = x

        self.add_scalar("validation.loss", reduced_loss, iteration)
        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            '''self.add_histogram(tag, value.data.cpu().numpy(), iteration)'''
            wandb.log({tag:wandb.Histogram(value.data.cpu().numpy())})

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        text_len = input_lengths[idx].item()
        text_string = sequence_to_text(text_padded[idx].tolist())[:text_len]
        mel_len = torch.max(torch.argmax(alignments[idx,:,text_len-5:text_len],dim=0)) # alignments.size(): [batch_size, mel_steps, txt_steps
        print("mel_len: {}".format(mel_len))

        mel = mel_outputs[idx:idx+1,:,:mel_len]

        np_wav = self.mel2wav(mel.type('torch.cuda.HalfTensor'))

        np_alignment = plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T)
        '''self.add_image(
            "alignment",
            np_alignment,
            iteration, dataformats='HWC')'''

        np_mel_target = plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy())
        '''self.add_image(
            "mel_target",
            np_mel_target,
            iteration, dataformats='HWC')'''

        np_mel_predicted = plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy())
        '''self.add_image(
            "mel_predicted",
            np_mel_predicted,
            iteration, dataformats='HWC')'''

        np_gate = plot_gate_outputs_to_numpy(
            gate_targets[idx].data.cpu().numpy(),
            torch.sigmoid(gate_outputs[idx]).data.cpu().numpy())
        '''self.add_image(
            "gate",
            np_gate,
            iteration, dataformats='HWC')'''

        # wandb log
        wandb.log({"epoch": epoch, "val.loss": reduced_loss})
        wandb.log({"val.alignment": [wandb.Image(np_alignment, caption=text_string)]})
        wandb.log({"val.audio": [wandb.Audio(np_wav.astype(np.float32), caption=text_string, sample_rate=sample_rate)]})
        wandb.log({"val.mel_target": [wandb.Image(np_mel_target)]})
        wandb.log({"val.mel_predicted": [wandb.Image(np_mel_predicted)]})
        wandb.log({"val.gate": [wandb.Image(np_gate)]})

        # foward attention ratio
        hop_list = [10, 20, 50, 100]
        for hop_size in hop_list:
            mean_far, batch_far = forward_attention_ratio(alignments, hop_size)
            log_name = "val.mean_forward_attention_ratio(hop_size={})".format(hop_size)
            wandb.log({log_name:mean_far})
            log_name = "val.forward_attention_ratio(hop_size={})".format(hop_size)
            wandb.log({log_name:wandb.Histogram(batch_far.data.cpu().numpy())})
