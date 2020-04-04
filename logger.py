import random
import torch
import numpy as np
import wandb
import sys
sys.path.append('waveglow/')
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter

from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy, \
    plot_embeddings_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy
from measures import forward_attention_ratio, get_mel_length, get_mel_lengths, \
    get_attention_quality, attention_ratio, attention_range_ratio, \
    multiple_attention_ratio
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
        self.waveglow = None
        self.waveglow_path = hparams.waveglow_path
        self.init_training_epoch_variables()

    def load_waveglow(self, waveglow_path):
        waveglow = torch.load(waveglow_path)['model']
        waveglow.cuda().eval().half()
        for k in waveglow.convinv:
            k.float()

        return waveglow

    def mel2wav(self, mel_outputs_postnet, with_denoiser=False):
        if self.waveglow is None:
            self.waveglow = self.load_waveglow(self.waveglow_path)
        with torch.no_grad():
            audio = self.waveglow.infer(mel_outputs_postnet, sigma=0.666)
        if with_denoiser:
            np_wav = denoiser(audio, strength=0.01)[:, 0].cpu().numpy()
        else:
            np_wav = audio[0].detach().cpu().numpy()

        return np_wav

    def get_embeddings(self, valset, model):
        speaker_list = list()
        for speaker in valset.speaker_list:
            speaker_int = valset.speaker2int(speaker)
            speaker_list.append(speaker_int)
        speaker_tensors = torch.tensor(speaker_list)

        emotion_vectors = list()
        for emotion in valset.emotion_list:
            emotion_vector = valset.get_emotion_input(emotion)
            emotion_vectors.append(emotion_vector)
        emotion_tensors = torch.stack(emotion_vectors)

        speaker_tensors = to_gpu(speaker_tensors).long()
        emotion_tensors = to_gpu(emotion_tensors).float()

        speaker_embeddings, emotion_embeddings = model.get_embeddings(speaker_tensors, emotion_tensors)

        return speaker_embeddings, emotion_embeddings


    def is_first_batch(self, iteration):
        if (iteration % self.batches_per_epoch) == 0:
            return True
        else:
            return False


    def is_last_batch(self, iteration):
        if (iteration % self.batches_per_epoch) == (self.batches_per_epoch - 1):
            return True
        else:
            return False


    def init_training_epoch_variables(self):
        self.sum_loss = 0
        self.sum_loss_mel = 0
        self.sum_loss_gate = 0
        self.sum_loss_spk_adv = 0
        self.sum_loss_emo_adv = 0
        self.sum_loss_att_means = 0

        self.sum_gate_accuracy = 0
        self.sum_gate_mae = 0

        self.sum_grad_norm = 0

        self.sum_mean_far = 0
        self.sum_mean_ar = 0
        self.sum_mean_letter_ar = 0
        self.best_letter_ar = -2
        self.worst_letter_ar = 2
        self.sum_mean_punct_ar = 0
        self.sum_mean_blank_ar = 0
        self.sum_mean_arr = 0
        self.sum_mean_mar = 0
        self.sum_mean_attention_quality = 0
        self.best_attention_quality = -2
        self.worst_attention_quality = 2

        self.sum_spk_adv_accuracy = 0
        self.sum_emo_adv_accuracy = 0


    def get_training_epoch_variables(self):
        dict_vars = dict()
        dict_vars['sum_loss'] = self.sum_loss
        dict_vars['sum_loss_mel'] = self.sum_loss_mel
        dict_vars['sum_loss_gate'] = self.sum_loss_gate
        dict_vars['sum_loss_spk_adv'] = self.sum_loss_spk_adv
        dict_vars['sum_loss_emo_adv'] = self.sum_loss_emo_adv
        dict_vars['sum_loss_att_means'] = self.sum_loss_att_means

        dict_vars['sum_gate_accuracy'] = self.sum_gate_accuracy
        dict_vars['sum_gate_mae'] = self.sum_gate_mae

        dict_vars['sum_grad_norm'] = self.sum_grad_norm

        dict_vars['sum_mean_far'] = self.sum_mean_far
        dict_vars['sum_mean_ar'] = self.sum_mean_ar
        dict_vars['sum_mean_letter_ar'] = self.sum_mean_letter_ar
        dict_vars['best_letter_ar'] = self.best_letter_ar
        dict_vars['worst_letter_ar'] = self.worst_letter_ar
        dict_vars['sum_mean_punct_ar'] = self.sum_mean_punct_ar
        dict_vars['sum_mean_blank_ar'] = self.sum_mean_blank_ar
        dict_vars['sum_mean_arr'] = self.sum_mean_arr
        dict_vars['sum_mean_mar'] = self.sum_mean_mar
        dict_vars['sum_mean_attention_quality'] = self.sum_mean_attention_quality
        dict_vars['best_attention_quality'] = self.best_attention_quality
        dict_vars['worst_attention_quality'] = self.worst_attention_quality

        dict_vars['sum_spk_adv_accuracy'] = self.sum_spk_adv_accuracy
        dict_vars['sum_emo_adv_accuracy'] = self.sum_emo_adv_accuracy

        return dict_vars


    def set_training_epoch_variables(self, dict_vars):
        self.sum_loss = dict_vars['sum_loss']
        self.sum_loss_mel = dict_vars['sum_loss_mel']
        self.sum_loss_gate = dict_vars['sum_loss_gate']
        self.sum_loss_spk_adv = dict_vars['sum_loss_spk_adv']
        self.sum_loss_emo_adv = dict_vars['sum_loss_emo_adv']
        self.sum_loss_att_means = dict_vars['sum_loss_att_means']

        self.sum_gate_accuracy = dict_vars['sum_gate_accuracy']
        self.sum_gate_mae = dict_vars['sum_gate_mae']

        self.sum_grad_norm = dict_vars['sum_grad_norm']

        self.sum_mean_far = dict_vars['sum_mean_far']
        self.sum_mean_ar = dict_vars['sum_mean_ar']
        self.sum_mean_letter_ar = dict_vars['sum_mean_letter_ar']
        self.best_letter_ar = dict_vars['best_letter_ar']
        self.worst_letter_ar = dict_vars['worst_letter_ar']
        self.sum_mean_punct_ar = dict_vars['sum_mean_punct_ar']
        self.sum_mean_blank_ar = dict_vars['sum_mean_blank_ar']
        self.sum_mean_arr = dict_vars['sum_mean_arr']
        self.sum_mean_mar = dict_vars['sum_mean_mar']
        self.sum_mean_attention_quality = dict_vars['sum_mean_attention_quality']
        self.best_attention_quality = dict_vars['best_attention_quality']
        self.worst_attention_quality = dict_vars['worst_attention_quality']

        self.sum_spk_adv_accuracy = dict_vars['sum_spk_adv_accuracy']
        self.sum_emo_adv_accuracy = dict_vars['sum_emo_adv_accuracy']



    def log_synth_dict(self, synth_dict, synth_dict_type,
            valset, val_speaker, val_emotion, iteration):
        '''
        synth_dict_type is one of ['random_pick', 'min_aq_tf', 'min_aq_fr']
        '''
        speaker_tensor = synth_dict['speaker_tensor']
        emotion_input_tensor = synth_dict['emotion_input_tensor']
        speaker = valset.int2speaker(speaker_tensor.item())
        str_emotion = valset.emotion_tensor2str_emotion(emotion_input_tensor)

        text_sequence = synth_dict['text_sequence']
        text_string = sequence_to_text(text_sequence.tolist())

        mel_true = synth_dict['mel_true']
        mel_output_tf = synth_dict['mel_output_tf']
        mel_output_fr = synth_dict['mel_output_fr']
        np_mel_true = plot_spectrogram_to_numpy(mel_true.detach().cpu().numpy())
        np_mel_output_tf = plot_spectrogram_to_numpy(mel_output_tf.detach().cpu().numpy())
        np_mel_output_fr = plot_spectrogram_to_numpy(mel_output_fr.detach().cpu().numpy())
        np_wav_true = self.mel2wav(mel_true.unsqueeze(0).cuda().half())
        np_wav_tf = self.mel2wav(mel_output_tf.unsqueeze(0).cuda().half())
        np_wav_fr = self.mel2wav(mel_output_fr.unsqueeze(0).cuda().half())

        alignment_tf = synth_dict['alignment_tf']
        alignment_fr = synth_dict['alignment_fr']
        np_alignment_tf = plot_alignment_to_numpy(alignment_tf.detach().cpu().numpy().T)
        np_alignment_fr = plot_alignment_to_numpy(alignment_fr.detach().cpu().numpy().T)

        caption_string = '[{speaker}|{emotion}] {text}'.format(
            speaker=speaker,
            emotion=str_emotion,
            text=text_string
        )

        log_prefix = "val/{speaker}/{emotion}/{synth_dict_type}".format(
            speaker=val_speaker, emotion=val_emotion, synth_dict_type=synth_dict_type)
        wandb.log({
            "{}/true/mel".format(log_prefix): [wandb.Image(np_mel_true)],
            "{}/true/wav".format(log_prefix): [wandb.Audio(np_wav_true.astype(np.float32), caption=caption_string, sample_rate=self.hparams.sampling_rate)],
            "{}/teacher_forcing/mel".format(log_prefix): [wandb.Image(np_mel_output_tf)],
            "{}/teacher_forcing/wav".format(log_prefix): [wandb.Audio(np_wav_tf.astype(np.float32), caption=caption_string, sample_rate=self.hparams.sampling_rate)],
            "{}/teacher_forcing/alignment".format(log_prefix): [wandb.Image(np_alignment_tf, caption=caption_string)],
            "{}/teacher_forcing/attention_ratio".format(log_prefix): synth_dict['ar_tf'],
            "{}/teacher_forcing/letter_attention_ratio".format(log_prefix): synth_dict['letter_ar_tf'],
            "{}/teacher_forcing/punctuation_attention_ratio".format(log_prefix): synth_dict['punct_ar_tf'],
            "{}/teacher_forcing/blank_attention_ratio".format(log_prefix): synth_dict['blank_ar_tf'],
            "{}/teacher_forcing/attention_range_ratio".format(log_prefix): synth_dict['arr_tf'],
            "{}/teacher_forcing/multiple_attention_ratio".format(log_prefix): synth_dict['mar_tf'],
            "{}/free_running/mel".format(log_prefix): [wandb.Image(np_mel_output_fr)],
            "{}/free_running/wav".format(log_prefix): [wandb.Audio(np_wav_fr.astype(np.float32), caption=caption_string, sample_rate=self.hparams.sampling_rate)],
            "{}/free_running/alignment".format(log_prefix): [wandb.Image(np_alignment_fr, caption=caption_string)],
            "{}/free_running/attention_ratio".format(log_prefix): synth_dict['ar_fr'],
            "{}/free_running/letter_attention_ratio".format(log_prefix): synth_dict['letter_ar_fr'],
            "{}/free_running/punctuation_attention_ratio".format(log_prefix): synth_dict['punct_ar_fr'],
            "{}/free_running/blank_attention_ratio".format(log_prefix): synth_dict['blank_ar_fr'],
            "{}/free_running/attention_range_ratio".format(log_prefix): synth_dict['arr_fr'],
            "{}/free_running/multiple_attention_ratio".format(log_prefix): synth_dict['mar_fr'],
        }, step=iteration)


    def log_training(self, trainset, hparams, dict_log_values, batches_per_epoch):

        iteration = dict_log_values['iteration']
        epoch = dict_log_values['epoch']
        losses = dict_log_values['losses']
        grad_norm = dict_log_values['grad_norm']
        learning_rate = dict_log_values['learning_rate']
        duration = dict_log_values['duration']
        x = dict_log_values['x']
        etc = dict_log_values['etc']
        y_pred = dict_log_values['y_pred']
        int_pred_speakers = dict_log_values['int_pred_speakers']
        gate_accuracy = dict_log_values['gate_accuracy']
        gate_mae = dict_log_values['gate_mae']
        att_measures = dict_log_values['att_measures']

        if self.hparams.residual_encoder:
            residual_encoding = dict_log_values['residual_encoding']
            mu = dict_log_values['mu']
            logvar = dict_log_values['logvar']
        if self.hparams.speaker_adversarial_training:
            spk_adv_accuracy = dict_log_values['spk_adv_accuracy']
            speaker_clsf_report = dict_log_values['speaker_clsf_report']
        if self.hparams.emotion_adversarial_training:
            emo_adv_accuracy = dict_log_values['emo_adv_accuracy']
            emotion_clsf_report = dict_log_values['emotion_clsf_report']

        self.batches_per_epoch = batches_per_epoch
        loss, loss_mel, loss_gate, loss_KLD, loss_spk_adv, loss_emo_adv, loss_att_means = losses
        text_padded, input_lengths, mel_padded, max_len, output_lengths = x
        speakers, sex, emotion_input_vectors, emotion_target_vectors, lang = etc
        _, mel_outputs, gate_outputs, alignments = y_pred
        mean_far, batch_far = att_measures[0]
        mean_ar, batch_ar = att_measures[1]
        mean_letter_ar, batch_letter_ar, best_letter_ar, worst_letter_ar = att_measures[2]
        mean_punct_ar, batch_punct_ar = att_measures[3]
        mean_blank_ar, batch_blank_ar = att_measures[4]
        mean_arr, batch_arr = att_measures[5]
        mean_mar, batch_mar = att_measures[6]
        (mean_attention_quality, batch_attention_quality, best_attention_quality,
            worst_attention_quality) = att_measures[7]

        # Initialize training_epoch_variables
        if self.is_first_batch(iteration):
            self.init_training_epoch_variables()

        # Update training_epoch_variables
        self.sum_loss += loss
        self.sum_loss_mel += loss_mel
        self.sum_loss_gate += loss_gate
        self.sum_loss_spk_adv += loss_spk_adv
        self.sum_loss_emo_adv += loss_emo_adv
        self.sum_loss_att_means += loss_att_means

        self.sum_gate_accuracy += gate_accuracy
        self.sum_gate_mae += gate_mae

        self.sum_grad_norm += grad_norm

        self.sum_mean_far += mean_far
        self.sum_mean_ar += mean_ar
        self.sum_mean_letter_ar += mean_letter_ar
        self.best_letter_ar = np.max([self.best_letter_ar, best_letter_ar])
        self.worst_letter_ar = np.min([self.worst_letter_ar, worst_letter_ar])
        self.sum_mean_punct_ar += mean_punct_ar
        self.sum_mean_blank_ar += mean_blank_ar
        self.sum_mean_arr += mean_arr
        self.sum_mean_mar += mean_mar
        self.sum_mean_attention_quality += mean_attention_quality
        self.best_attention_quality = np.max([self.best_attention_quality, best_attention_quality])
        self.worst_attention_quality = np.min([self.worst_attention_quality, worst_attention_quality])

        # wandb log
        wandb.log({"epoch": epoch,
                   "iteration":iteration,
                   "train/loss": loss,
                   "train/loss_mel": loss_mel,
                   "train/loss_gate": loss_gate,
                   "train/gate_accuracy": gate_accuracy,
                   "train/gate_mean_absolute_error":gate_mae,
                   "train/grad_norm": grad_norm,
                   "train/learning_rate": learning_rate,
                   "train/iter_duration": duration,
                   "train/mean_forward_attention_ratio":mean_far,
                   "train/mean_attention_ratio":mean_ar,
                   "train/mean_letter_attention_ratio":mean_letter_ar,
                   "train/best_letter_attention_ratio":best_letter_ar,
                   "train/worst_letter_attention_ratio":worst_letter_ar,
                   "train/mean_punctuation_attention_ratio":mean_punct_ar,
                   "train/mean_blank_attention_ratio":mean_blank_ar,
                   "train/mean_attention_range_ratio":mean_arr,
                   "train/mean_multiple_attention_ratio":mean_mar,
                   "train/mean_attention_quality":mean_attention_quality,
                   "train/best_attention_quality":best_attention_quality,
                   "train/worst_attention_quality":worst_attention_quality,
                   "train/forward_attention_ratio":wandb.Histogram(batch_far.detach().cpu().numpy()),
                   "train/attention_ratio":wandb.Histogram(batch_ar.detach().cpu().numpy()),
                   "train/letter_attention_ratio":wandb.Histogram(batch_letter_ar.detach().cpu().numpy()),
                   "train/punct_attention_ratio":wandb.Histogram(batch_punct_ar.detach().cpu().numpy()),
                   "train/blank_attention_ratio":wandb.Histogram(batch_blank_ar.detach().cpu().numpy()),
                   "train/attention_range_ratio":wandb.Histogram(batch_arr.detach().cpu().numpy()),
                   "train/multiple_attention_ratio":wandb.Histogram(batch_mar.detach().cpu().numpy()),
                   "train/attention_quality":wandb.Histogram(batch_attention_quality.detach().cpu().numpy())
                   }, step=iteration)

        # Logging values concerning the residual encoder.
        if self.hparams.residual_encoder:
            wandb.log({"train/res_en/loss_KLD": loss_KLD,
                       "train/res_en/residual_encoding": wandb.Histogram(residual_encoding.detach().cpu().numpy()),
                       "train/res_en/mu": wandb.Histogram(mu.detach().cpu().numpy()),
                       "train/res_en/var": wandb.Histogram(logvar.exp().detach().cpu().numpy()),
                       "train/res_en/mean_mu": mu.mean().detach().cpu().numpy(),
                       "train/res_en/mean_var": logvar.exp().mean().detach().cpu().numpy(),
                       }, step=iteration)

        # Logging values concerning speaker adversarial training.
        if self.hparams.speaker_adversarial_training:
            # Update training_epoch_variables
            self.sum_spk_adv_accuracy += spk_adv_accuracy
            wandb.log({"train/spk_adv/loss": loss_spk_adv,
                       "train/spk_adv/accuracy": spk_adv_accuracy}
                       , step=iteration)

            for str_speaker in hparams.speakers:
                spk_measure_dict = speaker_clsf_report[str_speaker]
                for measure, m_value in spk_measure_dict.items():
                    if measure == 'support':
                        continue
                    wandb.log({"train/spk_adv/{speaker}/{measure}".format(
                            speaker=str_speaker, measure=measure): m_value
                        }, step=iteration)

        # Logging values concerning emotion adversarial training.
        if self.hparams.emotion_adversarial_training:
            # Update training_epoch_variables
            self.sum_emo_adv_accuracy += emo_adv_accuracy
            wandb.log({"train/emo_adv/loss": loss_emo_adv,
                       "train/emo_adv/accuracy": emo_adv_accuracy}
                       , step=iteration)

            for str_emotion in hparams.emotions:
                emo_measure_dict = emotion_clsf_report[str_emotion]
                for measure, m_value in emo_measure_dict.items():
                    if measure == 'support':
                        continue
                    wandb.log({"train/emo_adv/{emotion}/{measure}".format(
                            emotion=str_emotion, measure=measure): m_value
                        }, step=iteration)

        # Logging loss_monotonic_attention_MSE.
        if self.hparams.monotonic_attention:
            wandb.log({"train/loss_monotonic_attention_MSE": loss_att_means}
                       , step=iteration)

        # Log training epoch variables
        if self.is_last_batch(iteration):
            # wandb log
            wandb.log({"train_epoch/loss": (self.sum_loss / self.batches_per_epoch),
                       "train_epoch/loss_mel": (self.sum_loss_mel / self.batches_per_epoch),
                       "train_epoch/loss_gate": (self.sum_loss_gate / self.batches_per_epoch),
                       "train_epoch/gate_accuracy": (self.sum_gate_accuracy / self.batches_per_epoch),
                       "train_epoch/gate_mean_absolute_error": (self.sum_gate_mae / self.batches_per_epoch),
                       "train_epoch/grad_norm": (self.sum_grad_norm / self.batches_per_epoch),
                       "train_epoch/mean_forward_attention_ratio":(self.sum_mean_far / self.batches_per_epoch),
                       "train_epoch/mean_attention_ratio":(self.sum_mean_ar / self.batches_per_epoch),
                       "train_epoch/mean_letter_attention_ratio":(self.sum_mean_letter_ar / self.batches_per_epoch),
                       "train_epoch/best_letter_attention_ratio":self.best_letter_ar,
                       "train_epoch/worst_letter_attention_ratio":self.worst_letter_ar,
                       "train_epoch/mean_punctuation_attention_ratio":(self.sum_mean_punct_ar / self.batches_per_epoch),
                       "train_epoch/mean_blank_attention_ratio":(self.sum_mean_blank_ar / self.batches_per_epoch),
                       "train_epoch/mean_attention_range_ratio":(self.sum_mean_arr / self.batches_per_epoch),
                       "train_epoch/mean_multiple_attention_ratio":(self.sum_mean_mar / self.batches_per_epoch),
                       "train_epoch/mean_attention_quality":(self.sum_mean_attention_quality / self.batches_per_epoch),
                       "train_epoch/best_attention_quality":self.best_attention_quality,
                       "train_epoch/worst_attention_quality":self.worst_attention_quality
                       }, step=iteration)

            if self.hparams.speaker_adversarial_training:
                wandb.log({"train_epoch/spk_adv/loss": (self.sum_loss_spk_adv / self.batches_per_epoch),
                           "train_epoch/spk_adv/accuracy": (self.sum_spk_adv_accuracy / self.batches_per_epoch)
                           }, step=iteration)

                speaker_clsf_report_train_epoch = dict_log_values['speaker_clsf_report_train_epoch']
                for str_speaker in hparams.speakers:
                    spk_measure_dict = speaker_clsf_report_train_epoch[str_speaker]
                    for measure, m_value in spk_measure_dict.items():
                        if measure == 'support':
                            continue
                        wandb.log({"train_epoch/spk_adv/{speaker}/{measure}".format(
                                speaker=str_speaker, measure=measure): m_value
                            }, step=iteration)
                        print("train_epoch/spk_adv/{speaker}/{measure}".format(
                                speaker=str_speaker, measure=measure), m_value)

            if self.hparams.emotion_adversarial_training:
                wandb.log({"train_epoch/emo_adv/loss": (self.sum_loss_emo_adv / self.batches_per_epoch),
                           "train_epoch/emo_adv/accuracy": (self.sum_emo_adv_accuracy / self.batches_per_epoch)
                           }, step=iteration)

                emotion_clsf_report_train_epoch = dict_log_values['emotion_clsf_report_train_epoch']
                for str_emotion in hparams.emotions:
                    emo_measure_dict = emotion_clsf_report_train_epoch[str_emotion]
                    for measure, m_value in emo_measure_dict.items():
                        if measure == 'support':
                            continue
                        wandb.log({"train_epoch/emo_adv/{emotion}/{measure}".format(
                                emotion=str_emotion, measure=measure): m_value
                            }, step=iteration)
                        print("train_epoch/emo_adv/{emotion}/{measure}".format(
                                emotion=str_emotion, measure=measure), m_value)

            if self.hparams.monotonic_attention:
                wandb.log({"train_epoch/loss_monotonic_attention_MSE": (self.sum_loss_att_means / self.batches_per_epoch)
                           }, step=iteration)


    def log_validation(self, trainset, valset, val_type, hparams, dict_log_values):

        # Validation type: {('all', 'all'), ('speaker1', 'emotion1'), ...}
        (val_speaker, val_emotion) = val_type

        iteration = dict_log_values['iteration']
        epoch = dict_log_values['epoch']

        model = dict_log_values['model']

        text_padded, input_lengths, mel_padded, max_len, output_lengths = dict_log_values['x']
        speakers, sex, emotion_input_vectors, emotion_target_vectors, lang = dict_log_values['etc']
        mel_targets, gate_targets = dict_log_values['y']
        _, mel_outputs, gate_outputs, alignments = dict_log_values['y_pred']
        int_pred_speakers = dict_log_values['int_pred_speakers']

        loss, loss_mel, loss_gate, loss_KLD, loss_spk_adv, loss_emo_adv, loss_att_means = dict_log_values['losses']
        far_pair, ar_pairs, arr_pair, mar_pair = dict_log_values['attention_measures']
        far_fr_pair, ar_fr_pairs, arr_fr_pair, mar_fr_pair = dict_log_values['attention_measures_fr']

        gate_accuracy = dict_log_values['gate_accuracy']
        gate_mae = dict_log_values['gate_mae']

        synth_dict_rand = dict_log_values['synth_dict_rand']
        synth_dict_min_aq_tf = dict_log_values['synth_dict_min_aq_tf']
        synth_dict_min_aq_fr = dict_log_values['synth_dict_min_aq_fr']

        if self.hparams.residual_encoder:
            residual_encoding = dict_log_values['residual_encoding']
            mu = dict_log_values['mu']
            logvar = dict_log_values['logvar']

        if self.hparams.speaker_adversarial_training:
            spk_adv_accuracy = dict_log_values['spk_adv_accuracy']
            if val_type == ('all', 'all'):
                speaker_clsf_report = dict_log_values['speaker_clsf_report']
        if self.hparams.emotion_adversarial_training:
            emo_adv_accuracy = dict_log_values['emo_adv_accuracy']
            if val_type == ('all', 'all'):
                emotion_clsf_report = dict_log_values['emotion_clsf_report']

        # Attention quality measures (teacher forcing)
        mean_far, batch_far = far_pair
        mean_ar, batch_ar = ar_pairs[0]
        mean_letter_ar, batch_letter_ar = ar_pairs[1]
        best_letter_ar = batch_letter_ar.max().item()
        worst_letter_ar = batch_letter_ar.min().item()
        mean_punct_ar, batch_punct_ar = ar_pairs[2]
        mean_blank_ar, batch_blank_ar = ar_pairs[3]
        mean_arr, batch_arr = arr_pair
        mean_mar, batch_mar = mar_pair
        mean_attention_quality = get_attention_quality(mean_far, mean_ar, mean_arr, mean_mar)
        batch_attention_quality = get_attention_quality(batch_far, batch_ar, batch_arr, batch_mar)
        best_attention_quality = batch_attention_quality.max().item()
        worst_attention_quality = batch_attention_quality.min().item()

        # Attention quality measures (free running)
        mean_far_fr, batch_far_fr = far_fr_pair
        mean_ar_fr, batch_ar_fr = ar_fr_pairs[0]
        mean_letter_ar_fr, batch_letter_ar_fr = ar_fr_pairs[1]
        mean_punct_ar_fr, batch_punct_ar_fr = ar_fr_pairs[2]
        mean_blank_ar_fr, batch_blank_ar_fr = ar_fr_pairs[3]
        mean_arr_fr, batch_arr_fr = arr_fr_pair
        mean_mar_fr, batch_mar_fr = mar_fr_pair
        mean_attention_quality_fr = get_attention_quality(mean_far_fr, mean_ar_fr, mean_arr_fr, mean_mar_fr)
        batch_attention_quality_fr = get_attention_quality(batch_far_fr, batch_ar_fr, batch_arr_fr, batch_mar_fr)
        best_attention_quality_fr = batch_attention_quality_fr.max().item()
        worst_attention_quality_fr = batch_attention_quality_fr.min().item()


        # [#1] Logging for all val_type
        # [#1-1] Logging values generated by teacher-forcing
        log_prefix = "val/{speaker}/{emotion}/{dec_type}".format(
            speaker=val_speaker, emotion=val_emotion, dec_type='teacher_forcing')
        wandb.log({"{}/loss".format(log_prefix): loss,
                   "{}/loss_mel".format(log_prefix): loss_mel,
                   "{}/loss_gate".format(log_prefix): loss_gate,
                   "{}/gate_accuracy".format(log_prefix): gate_accuracy,
                   "{}/gate_mean_absolute_error".format(log_prefix): gate_mae,
                   "{}/mean_forward_attention_ratio".format(log_prefix):mean_far,
                   "{}/mean_attention_ratio".format(log_prefix):mean_ar,
                   "{}/mean_letter_attention_ratio".format(log_prefix):mean_letter_ar,
                   "{}/best_letter_attention_ratio".format(log_prefix):best_letter_ar,
                   "{}/worst_letter_attention_ratio".format(log_prefix):worst_letter_ar,
                   "{}/mean_punctuation_attention_ratio".format(log_prefix):mean_punct_ar,
                   "{}/mean_blank_attention_ratio".format(log_prefix):mean_blank_ar,
                   "{}/mean_attention_range_ratio".format(log_prefix):mean_arr,
                   "{}/mean_multiple_attention_ratio".format(log_prefix):mean_mar,
                   "{}/mean_attention_quality".format(log_prefix):mean_attention_quality,
                   "{}/best_attention_quality".format(log_prefix):best_attention_quality,
                   "{}/worst_attention_quality".format(log_prefix):worst_attention_quality,
                   "{}/forward_attention_ratio".format(log_prefix):wandb.Histogram(batch_far.detach().cpu().numpy()),
                   "{}/attention_ratio".format(log_prefix):wandb.Histogram(batch_ar.detach().cpu().numpy()),
                   "{}/letter_attention_ratio".format(log_prefix):wandb.Histogram(batch_letter_ar.detach().cpu().numpy()),
                   "{}/punctuation_attention_ratio".format(log_prefix):wandb.Histogram(batch_punct_ar.detach().cpu().numpy()),
                   "{}/blank_attention_ratio".format(log_prefix):wandb.Histogram(batch_blank_ar.detach().cpu().numpy()),
                   "{}/attention_range_ratio".format(log_prefix):wandb.Histogram(batch_arr.detach().cpu().numpy()),
                   "{}/multiple_attention_ratio".format(log_prefix):wandb.Histogram(batch_mar.detach().cpu().numpy()),
                   "{}/attention_quality".format(log_prefix):wandb.Histogram(batch_attention_quality.detach().cpu().numpy())
                   } , step=iteration)

        # Logging values concerning the residual encoder.
        if self.hparams.residual_encoder:
            residual_encoding = dict_log_values['residual_encoding']
            mu = dict_log_values['mu']
            logvar = dict_log_values['logvar']
            wandb.log({"{}/res_en/loss_KLD".format(log_prefix): loss_KLD,
                       "{}/res_en/residual_encoding".format(log_prefix): wandb.Histogram(residual_encoding.detach().cpu().numpy()),
                       "{}/res_en/mu".format(log_prefix): wandb.Histogram(mu.detach().cpu().numpy()),
                       "{}/res_en/var".format(log_prefix): wandb.Histogram(logvar.exp().detach().cpu().numpy()),
                       "{}/res_en/mean_mu".format(log_prefix): mu.mean().detach().cpu().numpy(),
                       "{}/res_en/mean_var".format(log_prefix): logvar.exp().mean().detach().cpu().numpy(),
                       }, step=iteration)

        # Logging values concerning speaker adversarial training.
        if self.hparams.speaker_adversarial_training:
            wandb.log({"{}/spk_adv/loss".format(log_prefix): loss_spk_adv,
                       "{}/spk_adv/accuracy".format(log_prefix): spk_adv_accuracy}
                       , step=iteration)

        # Logging values concerning emotion adversarial training.
        if self.hparams.emotion_adversarial_training:
            wandb.log({"{}/emo_adv/loss".format(log_prefix): loss_emo_adv,
                       "{}/emo_adv/accuracy".format(log_prefix): emo_adv_accuracy}
                       , step=iteration)

        # Logging loss_monotonic_attention_MSE.
        if self.hparams.monotonic_attention:
             wandb.log({"{}/loss_monotonic_attention_MSE".format(log_prefix): loss_att_means}
                        , step=iteration)

        # [#1-2] Logging values generated by free-running
        log_prefix_fr = "val/{speaker}/{emotion}/{dec_type}".format(
            speaker=val_speaker, emotion=val_emotion, dec_type='free_running')
        wandb.log({"{}/mean_forward_attention_ratio".format(log_prefix_fr):mean_far_fr,
                   "{}/mean_attention_ratio".format(log_prefix_fr):mean_ar_fr,
                   "{}/mean_letter_attention_ratio".format(log_prefix_fr):mean_letter_ar_fr,
                   "{}/best_letter_attention_ratio".format(log_prefix_fr):batch_letter_ar_fr.max().item(),
                   "{}/worst_letter_attention_ratio".format(log_prefix_fr):batch_letter_ar_fr.min().item(),
                   "{}/mean_punctuation_attention_ratio".format(log_prefix_fr):mean_punct_ar_fr,
                   "{}/mean_blank_attention_ratio".format(log_prefix_fr):mean_blank_ar_fr,
                   "{}/mean_attention_range_ratio".format(log_prefix_fr):mean_arr_fr,
                   "{}/mean_multiple_attention_ratio".format(log_prefix_fr):mean_mar_fr,
                   "{}/mean_attention_quality".format(log_prefix_fr):mean_attention_quality_fr,
                   "{}/best_attention_quality".format(log_prefix_fr):best_attention_quality_fr,
                   "{}/worst_attention_quality".format(log_prefix_fr):worst_attention_quality_fr,
                   "{}/forward_attention_ratio".format(log_prefix_fr):wandb.Histogram(batch_far_fr.detach().cpu().numpy()),
                   "{}/attention_ratio".format(log_prefix_fr):wandb.Histogram(batch_ar_fr.detach().cpu().numpy()),
                   "{}/letter_attention_ratio".format(log_prefix_fr):wandb.Histogram(batch_letter_ar_fr.detach().cpu().numpy()),
                   "{}/punctuation_attention_ratio".format(log_prefix_fr):wandb.Histogram(batch_punct_ar_fr.detach().cpu().numpy()),
                   "{}/blank_attention_ratio".format(log_prefix_fr):wandb.Histogram(batch_blank_ar_fr.detach().cpu().numpy()),
                   "{}/attention_range_ratio".format(log_prefix_fr):wandb.Histogram(batch_arr_fr.detach().cpu().numpy()),
                   "{}/multiple_attention_ratio".format(log_prefix_fr):wandb.Histogram(batch_mar_fr.detach().cpu().numpy()),
                   "{}/attention_quality".format(log_prefix_fr):wandb.Histogram(batch_attention_quality_fr.detach().cpu().numpy())
                   } , step=iteration)



        # [#2] Logging for all val_type except ('all', 'all')
        if (val_speaker, val_emotion) != ('all', 'all'):
            # [2-1] synth_dict_rand
            self.log_synth_dict(synth_dict_rand, 'random_pick', valset, val_speaker, val_emotion, iteration)

            # [2-2] synth_dict_min_aq_tf
            self.log_synth_dict(synth_dict_min_aq_tf, 'min_aq_tf', valset, val_speaker, val_emotion, iteration)

            # [2-3] synth_dict_min_aq_fr
            self.log_synth_dict(synth_dict_min_aq_fr, 'min_aq_fr', valset, val_speaker, val_emotion, iteration)

        # [#3] Loggings only for all validation set.
        else:
            # Log epochs and iterations.
            wandb.log({"epoch": epoch,
                       "iteration":iteration}
                       , step=iteration)

            speaker_embeddings, emotion_embeddings = self.get_embeddings(valset, model)

            if len(valset.speaker_list) > 1:
                np_plot_speaker_embeddings = plot_embeddings_to_numpy(valset.speaker_list, speaker_embeddings.detach().cpu().numpy())
                wandb.log({"speaker_embeddings": [wandb.Image(np_plot_speaker_embeddings)]}
                           , step=iteration)

            if len(valset.emotion_list) > 1:
                np_plot_emotion_embeddings = plot_embeddings_to_numpy(valset.emotion_list, emotion_embeddings.detach().cpu().numpy())
                wandb.log({"emotion_embeddings": [wandb.Image(np_plot_emotion_embeddings)]}
                           , step=iteration)

            if self.hparams.speaker_adversarial_training:
                log_prefix = "val/{speaker}/{emotion}".format(
                    speaker=val_speaker, emotion=val_emotion)

                for str_speaker in hparams.speakers:
                    spk_measure_dict = speaker_clsf_report[str_speaker]
                    for measure, m_value in spk_measure_dict.items():
                        if measure == 'support':
                            continue
                        wandb.log({"{prefix}/spk_adv/{speaker}/{measure}".format(
                                prefix=log_prefix, speaker=str_speaker, measure=measure): m_value}
                            , step=iteration)
                        print("{prefix}/spk_adv/{speaker}/{measure}".format(
                                prefix=log_prefix, speaker=str_speaker, measure=measure), m_value)

            if self.hparams.emotion_adversarial_training:
                log_prefix = "val/{speaker}/{emotion}".format(
                    speaker=val_speaker, emotion=val_emotion)

                for str_emotion in hparams.emotions:
                    spk_measure_dict = emotion_clsf_report[str_emotion]
                    for measure, m_value in spk_measure_dict.items():
                        if measure == 'support':
                            continue
                        wandb.log({"{prefix}/emo_adv/{emotion}/{measure}".format(
                                prefix=log_prefix, emotion=str_emotion, measure=measure): m_value}
                            , step=iteration)
                        print("{prefix}/emo_adv/{emotion}/{measure}".format(
                                prefix=log_prefix, emotion=str_emotion, measure=measure), m_value)

            # plot distribution of parameters
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                wandb.log({tag:wandb.Histogram(value.detach().cpu().numpy()), "epoch": epoch, "iteration":iteration}, step=iteration)

            # Free-running test.
            if hparams.log_fr_test:
                text = hparams.test_text
                for speaker in hparams.speakers:
                    for emotion in hparams.emotions:
                        sequence = np.array(text_to_sequence(text, hparams.text_cleaners))[None, :]
                        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
                        text_len = sequence.size(1)
                        speaker_int = valset.speaker2int(speaker)
                        emotion_input_vector = valset.get_emotion_input(emotion)
                        speaker_tensor = to_gpu(torch.tensor(speaker_int).view(1)).long()
                        emotion_input_tensor = to_gpu(torch.tensor(emotion_input_vector).view(1,-1)).float()

                        _, mel_outputs_postnet, _, alignments = model.inference(sequence, speaker_tensor, emotion_input_tensor)

                        np_wav = self.mel2wav(mel_outputs_postnet.cuda().half())
                        np_alignment = plot_alignment_to_numpy(alignments[0].detach().cpu().numpy().T, text_len)
                        np_mel_predicted = plot_spectrogram_to_numpy(mel_outputs_postnet[0].detach().cpu().numpy())

                        group_log_name = "test_free_running/{speaker}/{emotion}".format(
                            speaker=speaker, emotion=emotion
                        )
                        wandb.log({
                            "{}/wav".format(group_log_name): [wandb.Audio(np_wav.astype(np.float32), caption=text, sample_rate=hparams.sampling_rate)],
                            "{}/alignment".format(group_log_name): [wandb.Image(np_alignment)],
                            "{}/mel_predicted".format(group_log_name): [wandb.Image(np_mel_predicted)]
                        }, step=iteration)
