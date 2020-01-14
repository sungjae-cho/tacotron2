import random
import torch
import numpy as np
import wandb
import sys
sys.path.append('waveglow/')
import warnings
warnings.filterwarnings('ignore')
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy, \
    plot_embeddings_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy
from measures import forward_attention_ratio, get_mel_length
from text import sequence_to_text
from denoiser import Denoiser
from text import text_to_sequence
from utils import to_gpu


class Tacotron2Logger(SummaryWriter):
    def __init__(self, run_name, prj_name, logdir, resume):
        self.run_name = run_name
        if resume:
            wandb.init(project=prj_name, resume=run_name)
        else:
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

    def get_embeddings(self, valset, model):
        speaker_list = list()
        for speaker in valset.speaker_list:
            speaker_int = valset.speaker2int(speaker)
            speaker_list.append(speaker_int)
        speaker_tensors = torch.tensor(speaker_list)

        emotion_vectors = list()
        for emotion in valset.emotion_list:
            emotion_vector = valset.get_emotion(emotion)
            emotion_vectors.append(emotion_vector)
        emotion_tensors = torch.stack(emotion_vectors)

        speaker_tensors = to_gpu(speaker_tensors).long()
        emotion_tensors = to_gpu(emotion_tensors).float()
        print("speaker_tensors.size()",speaker_tensors.size())
        print("emotion_tensors.size()",emotion_tensors.size())

        speaker_embeddings, emotion_embeddings = model.get_embeddings(speaker_tensors, emotion_tensors)

        return speaker_embeddings, emotion_embeddings


    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     x, etc, y_pred, iteration, epoch, forward_attention_loss=None):
            text_padded, input_lengths, mel_padded, max_len, output_lengths = x

            # Tensorbard log
            #self.add_scalar("training.loss", reduced_loss, iteration)
            #self.add_scalar("grad.norm", grad_norm, iteration)
            #self.add_scalar("learning.rate", learning_rate, iteration)
            #self.add_scalar("duration", duration, iteration)

            # wandb log
            wandb.log({"train/loss": reduced_loss,
                       "train/grad_norm": grad_norm,
                       "train/learning_rate": learning_rate,
                       "train/iter_duration": duration,
                       "epoch": epoch,
                       "iteration":iteration}
                       , step=iteration)
            if forward_attention_loss is not None:
                wandb.log({"train/forward_attention_loss": forward_attention_loss}
                           , step=iteration)
            _, mel_outputs, gate_outputs, alignments = y_pred

            hop_list = [1]
            for hop_size in hop_list:
                mean_far, batch_far = forward_attention_ratio(alignments, input_lengths, hop_size)
                log_name = "mean_forward_attention_ratio.train/hop_size={}".format(hop_size)
                wandb.log({log_name:mean_far, "epoch": epoch, "iteration":iteration}, step=iteration)
                log_name = "forward_attention_ratio.train/hop_size={}".format(hop_size)
                wandb.log({log_name:wandb.Histogram(batch_far.data.cpu().numpy()), "epoch": epoch, "iteration":iteration}, step=iteration)

    #TODO: add hparams indead of sample_rate
    def log_validation(self, valset,
        reduced_loss,  far_pair,
        model, x, y, etc, y_pred, iteration, epoch, sample_rate):
        text_padded, input_lengths, mel_padded, max_len, output_lengths = x
        speakers, sex, emotion_vectors, lang = etc

        #self.add_scalar("validation.loss", reduced_loss, iteration) # Tensorboard log
        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y
        mean_far, batch_far = far_pair

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            # self.add_histogram(tag, value.data.cpu().numpy(), iteration) # Tensorboard log
            wandb.log({tag:wandb.Histogram(value.data.cpu().numpy()), "epoch": epoch, "iteration":iteration}, step=iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)

        speaker = valset.int2speaker(speakers[idx].item())
        str_emotion = valset.emotion_tensor2str_emotion(emotion_vectors[idx])

        text_len = input_lengths[idx].item()
        text_string = sequence_to_text(text_padded[idx].tolist())[:text_len]

        mel_len = get_mel_length(alignments, idx, text_len)
        mel = mel_outputs[idx:idx+1,:,:mel_len]

        np_wav = self.mel2wav(mel.type('torch.cuda.HalfTensor'))

        np_alignment = plot_alignment_to_numpy(
            alignments[idx].data.cpu().numpy().T,
            decoding_len=mel_len)
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
        caption_string = '[{speaker}|{emotion}] {text}'.format(
            speaker=speaker,
            emotion=str_emotion,
            text=text_string
        )
        wandb.log({"val/loss": reduced_loss,
                   "val/alignment": [wandb.Image(np_alignment, caption=caption_string)],
                   "val/audio": [wandb.Audio(np_wav.astype(np.float32), caption=caption_string, sample_rate=sample_rate)],
                   "val/mel_target": [wandb.Image(np_mel_target)],
                   "val/mel_predicted": [wandb.Image(np_mel_predicted)],
                   "val/gate": [wandb.Image(np_gate)],
                   "epoch": epoch,
                   "iteration":iteration}
                   , step=iteration)

        # foward attention ratio
        log_name = "mean_forward_attention_ratio.val"
        wandb.log({log_name:mean_far, "epoch": epoch, "iteration":iteration}, step=iteration)
        log_name = "forward_attention_ratio.val"
        wandb.log({log_name:wandb.Histogram(batch_far.data.cpu().numpy()), "epoch": epoch, "iteration":iteration}, step=iteration)

        speaker_embeddings, emotion_embeddings = self.get_embeddings(valset, model)
        np_plot_speaker_embeddings = plot_embeddings_to_numpy(valset.speaker_list, speaker_embeddings.data.cpu().numpy())
        np_plot_emotion_embeddings = plot_embeddings_to_numpy(valset.emotion_list, emotion_embeddings.data.cpu().numpy())

        wandb.log({"speaker_embeddings": [wandb.Image(np_plot_speaker_embeddings)],
                   "emotion_embeddings": [wandb.Image(np_plot_emotion_embeddings)]}
                   , step=iteration)

        text = "KAIST is a national research university located in Daedeok Innopolis, Daejeon, South Korea."
        for speaker in valset.speaker_list:
            for emotion in valset.emotion_list:
                sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
                sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
                speaker_int = valset.speaker2int(speaker)
                emotion_vector = valset.get_emotion(emotion)
                speaker_tensor = to_gpu(torch.tensor(speaker_int).view(1)).long()
                emotion_tensor = to_gpu(torch.tensor(emotion_vector).view(1,-1)).float()

                _, mel_outputs_postnet, _, alignments = model.inference(sequence, speaker_tensor, emotion_tensor)

                np_wav = self.mel2wav(mel_outputs_postnet.type('torch.cuda.HalfTensor'))
                np_alignment = plot_alignment_to_numpy(alignments[0].data.cpu().numpy().T)
                np_mel_predicted = plot_spectrogram_to_numpy(mel_outputs_postnet[0].data.cpu().numpy())

                group_log_name = "Inference_test/{speaker}_{emotion}/".format(
                    speaker=speaker, emotion=emotion
                )
                wandb.log({
                    "{}wav".format(group_log_name): [wandb.Audio(np_wav.astype(np.float32), caption=text, sample_rate=sample_rate)],
                    "{}alignment".format(group_log_name): [wandb.Image(np_alignment)],
                    "{}mel_predicted".format(group_log_name): [wandb.Image(np_mel_predicted)]
                }, step=iteration)
