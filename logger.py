import random
import torch
import numpy as np
import wandb
import sys
sys.path.append('waveglow/')
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter

from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy, \
    plot_embeddings_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy
from measures import forward_attention_ratio, get_mel_length, attention_ratio, attention_range_ratio, multiple_attention_ratio
from text import sequence_to_text
from denoiser import Denoiser
from text import text_to_sequence
from utils import to_gpu, get_spk_adv_targets


class Tacotron2Logger(SummaryWriter):
    def __init__(self, hparams, run_name, prj_name, logdir, resume):
        self.hparams = hparams
        self.run_name = run_name
        if resume == "":
            wandb.init(name=run_name, project=prj_name, resume=resume)
        else:
            wandb.init(project=prj_name, resume=resume)
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

        speaker_embeddings, emotion_embeddings = model.get_embeddings(speaker_tensors, emotion_tensors)

        return speaker_embeddings, emotion_embeddings


    def log_training(self, losses, grad_norm, learning_rate, duration,
            x, etc, y_pred, pred_speakers, iteration, epoch,
            forward_attention_loss=None):

            loss, loss_mel, loss_spk_adv, loss_att_means = losses
            text_padded, input_lengths, mel_padded, max_len, output_lengths = x
            speakers, sex, emotion_vectors, lang = etc
            _, mel_outputs, gate_outputs, alignments = y_pred

            # Compute forward_attention_ratio.
            hop_size = 1
            mean_far, batch_far = forward_attention_ratio(alignments, input_lengths, output_lengths, gate_outputs, hop_size)
            mean_ar, batch_ar = attention_ratio(alignments, input_lengths, output_lengths, gate_outputs)
            mean_arr, batch_arr = attention_range_ratio(alignments, input_lengths, output_lengths, gate_outputs)
            mean_mar, batch_mar = multiple_attention_ratio(alignments, input_lengths, output_lengths, gate_outputs)
            mean_attention_quality = mean_far * mean_ar * mean_arr * (1 - mean_mar)
            batch_attention_quality = batch_far * batch_ar * batch_arr * (1 - batch_mar)

            # wandb log
            wandb.log({"epoch": epoch,
                       "iteration":iteration,
                       "train/loss": loss,
                       "train/loss_mel": loss_mel,
                       "train/grad_norm": grad_norm,
                       "train/learning_rate": learning_rate,
                       "train/iter_duration": duration,
                       "train/mean_forward_attention_ratio":mean_far,
                       "train/mean_attention_ratio":mean_ar,
                       "train/mean_attention_range_ratio":mean_arr,
                       "train/mean_multiple_attention_ratio":mean_mar,
                       "train/mean_attention_quality":mean_attention_quality,
                       "train/forward_attention_ratio":wandb.Histogram(batch_far.data.cpu().numpy()),
                       "train/attention_ratio":wandb.Histogram(batch_ar.data.cpu().numpy()),
                       "train/attention_range_ratio":wandb.Histogram(batch_arr.data.cpu().numpy()),
                       "train/multiple_attention_ratio":wandb.Histogram(batch_mar.data.cpu().numpy()),
                       "train/attention_quality":wandb.Histogram(batch_attention_quality.data.cpu().numpy())
                       }, step=iteration)

            # Calculate accuracy of the speaker adversarial training module.
            if self.hparams.speaker_adversarial_training:
                np_speakers = get_spk_adv_targets(speakers, input_lengths).cpu().numpy()
                np_pred_speakers = pred_speakers.cpu().numpy()
                spk_adv_accuracy = accuracy_score(np_speakers, np_pred_speakers)

                wandb.log({"train/loss_spk_adv": loss_spk_adv,
                           "train/spk_adv_accuracy": spk_adv_accuracy}
                           , step=iteration)

            # Logging forward_attention_loss.
            if forward_attention_loss is not None:
                wandb.log({"train/forward_attention_loss": forward_attention_loss}
                           , step=iteration)

           # Logging loss_monotonic_attention_MSE.
            if self.hparams.monotonic_attention:
                wandb.log({"train/loss_monotonic_attention_MSE": loss_att_means}
                           , step=iteration)



    def log_validation(self, valset, val_type,
        losses, far_pair, ar_pair, arr_pair, mar_pair,
        model, x, y, etc, y_pred, pred_speakers, iteration, epoch, hparams):

        # Validation type: {('all', 'all'), ('speaker1', 'emotion1'), ...}
        (val_speaker, val_emotion) = val_type

        loss, loss_mel, loss_spk_adv, loss_att_means = losses
        text_padded, input_lengths, mel_padded, max_len, output_lengths = x
        speakers, sex, emotion_vectors, lang = etc

        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y
        mean_far, batch_far = far_pair
        mean_ar, batch_ar = ar_pair
        mean_arr, batch_arr = arr_pair
        mean_mar, batch_mar = mar_pair
        mean_attention_quality = mean_far * mean_ar * mean_arr * (1 - mean_mar)
        batch_attention_quality = batch_far * batch_ar * batch_arr * (1 - batch_mar)

        # [#1] Logging for all val_type
        log_prefix = "val/{speaker}/{emotion}".format(speaker=val_speaker, emotion=val_emotion)
        wandb.log({"{}/loss".format(log_prefix): loss,
                   "{}/loss_mel".format(log_prefix): loss_mel,
                   "{}/mean_forward_attention_ratio".format(log_prefix):mean_far,
                   "{}/mean_attention_ratio".format(log_prefix):mean_ar,
                   "{}/mean_attention_range_ratio".format(log_prefix):mean_arr,
                   "{}/mean_multiple_attention_ratio".format(log_prefix):mean_mar,
                   "{}/mean_attention_quality".format(log_prefix):mean_attention_quality,
                   "{}/forward_attention_ratio".format(log_prefix):wandb.Histogram(batch_far.data.cpu().numpy()),
                   "{}/attention_ratio".format(log_prefix):wandb.Histogram(batch_ar.data.cpu().numpy()),
                   "{}/attention_range_ratio".format(log_prefix):wandb.Histogram(batch_arr.data.cpu().numpy()),
                   "{}/multiple_attention_ratio".format(log_prefix):wandb.Histogram(batch_mar.data.cpu().numpy()),
                   "{}/attention_quality".format(log_prefix):wandb.Histogram(batch_attention_quality.data.cpu().numpy())
                   } , step=iteration)

        if self.hparams.speaker_adversarial_training:
            # Calculate accuracy of the speaker adversarial training module.
            np_speakers = get_spk_adv_targets(speakers, input_lengths).cpu().numpy()
            np_pred_speakers = pred_speakers.cpu().numpy()
            val_spk_adv_accuracy = accuracy_score(np_speakers, np_pred_speakers)

            wandb.log({"{}/loss_spk_adv".format(log_prefix): loss_spk_adv,
                       "{}/spk_adv_accuracy".format(log_prefix): val_spk_adv_accuracy}
                       , step=iteration)

        # Logging loss_monotonic_attention_MSE.
        if self.hparams.monotonic_attention:
             wandb.log({"{}/loss_monotonic_attention_MSE".format(log_prefix): loss_att_means}
                        , step=iteration)

        # [#2] Logging for all val_type except ('all', 'all')
        if (val_speaker, val_emotion) != ('all', 'all'):
            # plot alignment, mel target and predicted, gate target and predicted
            idx = random.randint(0, alignments.size(0) - 1)

            speaker_tensor = speakers[idx].view(1)
            emotion_tensor = emotion_vectors[idx].view(1, -1)
            speaker = valset.int2speaker(speaker_tensor.item())
            str_emotion = valset.emotion_tensor2str_emotion(emotion_tensor)

            text_len = input_lengths[idx].item()
            text_sequence = text_padded[idx,:text_len].view(1, -1)
            text_string = sequence_to_text(text_sequence.squeeze().tolist())

            mel_len = get_mel_length(gate_outputs[idx])
            mel = mel_outputs[idx:idx+1,:,:mel_len]
            mel_target_len = output_lengths[idx].item()
            mel_target = mel_targets[idx:idx+1,:,:mel_target_len]

            _, mel_outputs_postnet_inf, _, alignments_inf = model.inference(text_sequence, speaker_tensor, emotion_tensor)

            np_wav = self.mel2wav(mel.type('torch.cuda.HalfTensor'))
            np_wav_target = self.mel2wav(mel_target.type('torch.cuda.HalfTensor'))
            np_wav_inf = self.mel2wav(mel_outputs_postnet_inf.type('torch.cuda.HalfTensor'))

            np_alignment = plot_alignment_to_numpy(
                alignments[idx].data.cpu().numpy().T,
                text_len,
                decoding_len=mel_len)
            np_alignment_inf = plot_alignment_to_numpy(
                alignments_inf[0].data.cpu().numpy().T,
                text_len)

            np_mel_target = plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy())
            np_mel_predicted = plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy())
            np_mel_predicted_inf = plot_spectrogram_to_numpy(mel_outputs_postnet_inf[0].data.cpu().numpy())

            np_gate = plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy())

            # wandb log
            caption_string = '[{speaker}|{emotion}] {text}'.format(
                speaker=speaker,
                emotion=str_emotion,
                text=text_string
            )

            log_prefix = "val/{speaker}/{emotion}".format(speaker=val_speaker, emotion=val_emotion)
            wandb.log({"{}/alignment/teacher_forcing".format(log_prefix): [wandb.Image(np_alignment, caption=caption_string)],
                       "{}/alignment/inference".format(log_prefix): [wandb.Image(np_alignment_inf, caption=caption_string)],
                       "{}/audio/target".format(log_prefix): [wandb.Audio(np_wav_target.astype(np.float32), caption=caption_string, sample_rate=hparams.sampling_rate)],
                       "{}/audio/teacher_forcing".format(log_prefix): [wandb.Audio(np_wav.astype(np.float32), caption=caption_string, sample_rate=hparams.sampling_rate)],
                       "{}/audio/inference".format(log_prefix): [wandb.Audio(np_wav_inf.astype(np.float32), caption=caption_string, sample_rate=hparams.sampling_rate)],
                       "{}/mel_target".format(log_prefix): [wandb.Image(np_mel_target)],
                       "{}/mel_predicted/teacher_forcing".format(log_prefix): [wandb.Image(np_mel_predicted)],
                       "{}/mel_predicted/inference".format(log_prefix): [wandb.Image(np_mel_predicted_inf)],
                       "{}/gate".format(log_prefix): [wandb.Image(np_gate)],
                       "{}/mean_forward_attention_ratio".format(log_prefix):mean_far,
                       "{}/mean_attention_ratio".format(log_prefix):mean_ar,
                       "{}/mean_attention_range_ratio".format(log_prefix):mean_arr,
                       "{}/mean_multiple_attention_ratio".format(log_prefix):mean_mar,
                       "{}/mean_attention_quality".format(log_prefix):mean_attention_quality,
                       "{}/forward_attention_ratio".format(log_prefix):wandb.Histogram(batch_far.data.cpu().numpy()),
                       "{}/attention_ratio".format(log_prefix):wandb.Histogram(batch_ar.data.cpu().numpy()),
                       "{}/attention_range_ratio".format(log_prefix):wandb.Histogram(batch_arr.data.cpu().numpy()),
                       "{}/multiple_attention_ratio".format(log_prefix):wandb.Histogram(batch_mar.data.cpu().numpy()),
                       "{}/attention_quality".format(log_prefix):wandb.Histogram(batch_attention_quality.data.cpu().numpy())
                       } , step=iteration)

        # [#3] Loggings only for all validation set.
        else:
            # Log epochs and iterations.
            wandb.log({"epoch": epoch,
                       "iteration":iteration}
                       , step=iteration)

            speaker_embeddings, emotion_embeddings = self.get_embeddings(valset, model)

            if len(valset.speaker_list) > 1:
                np_plot_speaker_embeddings = plot_embeddings_to_numpy(valset.speaker_list, speaker_embeddings.data.cpu().numpy())
                wandb.log({"speaker_embeddings": [wandb.Image(np_plot_speaker_embeddings)]}
                           , step=iteration)

            if len(valset.emotion_list) > 1:
                np_plot_emotion_embeddings = plot_embeddings_to_numpy(valset.emotion_list, emotion_embeddings.data.cpu().numpy())
                wandb.log({"emotion_embeddings": [wandb.Image(np_plot_emotion_embeddings)]}
                           , step=iteration)

            # plot distribution of parameters
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                wandb.log({tag:wandb.Histogram(value.data.cpu().numpy()), "epoch": epoch, "iteration":iteration}, step=iteration)

            # Inference test.
            text = "Hello! This is a synthesized audio without 'teacher-forcing.' Any question?"
            for speaker in valset.speaker_list:
                for emotion in valset.emotion_list:
                    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
                    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
                    text_len = sequence.size(1)
                    speaker_int = valset.speaker2int(speaker)
                    emotion_vector = valset.get_emotion(emotion)
                    speaker_tensor = to_gpu(torch.tensor(speaker_int).view(1)).long()
                    emotion_tensor = to_gpu(torch.tensor(emotion_vector).view(1,-1)).float()

                    _, mel_outputs_postnet, _, alignments = model.inference(sequence, speaker_tensor, emotion_tensor)

                    np_wav = self.mel2wav(mel_outputs_postnet.type('torch.cuda.HalfTensor'))
                    np_alignment = plot_alignment_to_numpy(alignments[0].data.cpu().numpy().T, text_len)
                    np_mel_predicted = plot_spectrogram_to_numpy(mel_outputs_postnet[0].data.cpu().numpy())

                    group_log_name = "Inference_test/{speaker}/{emotion}".format(
                        speaker=speaker, emotion=emotion
                    )
                    wandb.log({
                        "{}/wav".format(group_log_name): [wandb.Audio(np_wav.astype(np.float32), caption=text, sample_rate=hparams.sampling_rate)],
                        "{}/alignment".format(group_log_name): [wandb.Image(np_alignment)],
                        "{}/mel_predicted".format(group_log_name): [wandb.Image(np_mel_predicted)]
                    }, step=iteration)
