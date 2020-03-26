import random
import numpy as np
import torch
import torch.utils.data

import layers
from utils import load_wav_to_torch, load_filepaths_and_text, \
    load_wavpath_text_speaker_sex_emotion_lang, one_hot_encoding
from text import text_to_sequence


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, hparams, split, speaker=None, emotion=None):
        self.hparams = hparams
        self.split = split
        self.speaker = speaker
        self.emotion = emotion
        self.first_random_seed = 0

        loaded_tuple = load_wavpath_text_speaker_sex_emotion_lang(
            self.hparams, self.split, self.speaker, self.emotion, self.first_random_seed)
        self.wavpath_text_speaker_sex_emotion_lang = loaded_tuple[0]

        self.max_emotions = len(hparams.all_emotions)
        self.max_speakers = len(hparams.all_speakers)
        self.speaker_list = sorted(loaded_tuple[1])
        self.sex_list = sorted(loaded_tuple[2])
        self.emotion_list = sorted(loaded_tuple[3])
        self.lang_list = sorted(loaded_tuple[4])
        self.neutral_zero_vector = hparams.neutral_zero_vector
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)




    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel)

    def get_mel_text_etc_tuple(self, wavpath_text_speaker_sex_emotion_lang):
        wavpath, text, speaker, sex, emotion, lang = wavpath_text_speaker_sex_emotion_lang
        text = self.get_text(text)
        mel = self.get_mel(wavpath)
        speaker = self.get_speaker(speaker)
        sex = self.get_sex(sex)
        emotion_input_vector = self.get_emotion(emotion, is_input=True)
        emotion_target_vector = self.get_emotion(emotion, is_input=False)
        lang = self.get_lang(lang)
        return (text, mel, speaker, sex, emotion_input_vector, emotion_target_vector, lang)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            #audio_norm = audio / self.max_wav_value
            max_wav_value = audio.abs().max()
            audio_norm = audio / max_wav_value * 0.99
            '''max_wav_value = audio.abs().max()
            audio_norm = audio / max_wav_value * 0.95'''
            '''absmean_wav_value = audio.abs().mean()
            audio_norm = audio / absmean_wav_value * 0.5'''
            '''if audio.max() <= 1.0 and audio.min() >= -1.0:
                audio_norm = audio
            else:
                audio_norm = audio / self.max_wav_value'''
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def get_speaker(self, speaker):
        speaker_tensor = torch.IntTensor([self.speaker2int(speaker)])
        return speaker_tensor

    def get_sex(self, sex):
        sex_tensor = self.sex2int(sex)
        return sex_tensor

    def get_emotion(self, emotion, is_input=True):
        if self.neutral_zero_vector and is_input:
            one_hot_vector_size = self.max_emotions - 1
            if emotion == 'neutral':
                emotion_tensor = torch.zeros(one_hot_vector_size)
            else:
                emotion_tensor = one_hot_encoding(
                    self.emotion2int(emotion, is_input), one_hot_vector_size)
        else:
            emotion_tensor = one_hot_encoding(
                self.emotion2int(emotion, is_input), self.max_emotions)
        return emotion_tensor

    def get_lang(self, lang):
        lang_tensor = self.lang2int(sex)
        return lang_tensor

    def get_speaker_size(self):
        return len(self.speaker_list)

    def get_sex_size(self):
        return len(self.sex_list)

    def get_emotion_size(self):
        return len(self.emotion_list)

    def get_lang_size(self):
        return len(self.lang_list)

    def get_lang(self, lang):
        lang_tensor = torch.IntTensor(self.lang2int(lang))
        return lang_tensor

    def emotion_tensor2str_emotion(self, emotion_tensor, is_input=True):
        if self.neutral_zero_vector and is_input:
            if torch.sum(emotion_tensor).item() == 0:
                str_emotion = 'neutral'
            else:
                str_emotion = self.int2emotion(torch.argmax(emotion_tensor).item())
        else:
            str_emotion = self.int2emotion(torch.argmax(emotion_tensor).item())

        return str_emotion

    def speaker2int(self, speaker):
        return self.speaker_list.index(speaker)

    def int2speaker(self, integer):
        return self.speaker_list[integer]

    def sex2int(self, sex):
        return self.sex_list.index(sex)

    def int2sex(self, integer):
        return self.sex_list[integer]

    def emotion2int(self, emotion, is_input=True):
        if self.neutral_zero_vector and is_input:
            if emotion == 'neutral':
                return None
            else:
                nonneutral_emotions = self.hparams.all_emotions.copy()
                nonneutral_emotions.remove('neutral')
                return sorted(nonneutral_emotions).index(emotion)

        else:
            return self.emotion_list.index(emotion)

    def int2emotion(self, integer, is_input=True):
        if self.neutral_zero_vector and is_input:
            if integer is None:
                return 'neutral'
            else:
                nonneutral_emotions = self.hparams.all_emotions.copy()
                nonneutral_emotions.remove('neutral')
                return sorted(nonneutral_emotions)[integer]
        else:
            return self.emotion_list[integer]

    def lang2int(self, lang):
        return self.lang_list.index(lang)

    def int2lang(self, integer):
        return self.lang_list[integer]

    def upsampling(self, random_seed):
        if self.split == 'train':
            print("Upsampling the training set again.")
            loaded_tuple = load_wavpath_text_speaker_sex_emotion_lang(
                self.hparams, self.split, self.speaker, self.emotion, random_seed)
            self.wavpath_text_speaker_sex_emotion_lang = loaded_tuple[0]

    def __getitem__(self, index):
        mel_text_etc_tuple = self.get_mel_text_etc_tuple(self.wavpath_text_speaker_sex_emotion_lang[index])
        return mel_text_etc_tuple

    def __len__(self):
        return len(self.wavpath_text_speaker_sex_emotion_lang)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        batch: [text_normalized, mel_normalized, speaker, sex, emotion_vec, lang]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))

        # incldue speakers, sex, emotion vectors, and language.
        speakers = torch.LongTensor(len(batch))
        sex = torch.LongTensor(len(batch))
        emotion_input_vector_dim = batch[0][4].size(0)
        emotion_input_vectors = torch.FloatTensor(len(batch), emotion_input_vector_dim)
        emotion_target_vector_dim = batch[0][5].size(0)
        emotion_target_vectors = torch.FloatTensor(len(batch), emotion_target_vector_dim)
        lang = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

            speakers[i] = batch[ids_sorted_decreasing[i]][2]
            sex[i] = batch[ids_sorted_decreasing[i]][3]
            emotion_input_vectors[i,:] = batch[ids_sorted_decreasing[i]][4]
            emotion_target_vectors[i,:] = batch[ids_sorted_decreasing[i]][5]
            print("emotion_target_vectors[i,:]", emotion_target_vectors[i,:])
            lang = batch[ids_sorted_decreasing[i]][6]

        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths, \
            speakers, sex, emotion_input_vectors, emotion_target_vectors, lang
