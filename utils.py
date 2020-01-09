import numpy as np
from scipy.io.wavfile import read
import torch
import pandas as pd

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    #mask = (ids < lengths.unsqueeze(1)).byte()
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def one_hot_encoding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].

    From: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/26
    """
    y = torch.eye(num_classes)
    return y[labels]


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def load_wavpath_text_speaker_sex_emotion_lang(csv_path, split):
    '''
    split in {'train', 'val', 'test'}
    '''
    df = pd.read_csv(csv_path)
    df = df[df.split == split]
    columns = ['wav_path', 'text', 'speaker', 'sex', 'emotion', 'lang']
    df = df[columns]
    df_list = df.values.tolist()
    speaker_list = df.speaker.unique().tolist()
    sex_list = df.sex.unique().tolist()
    emotion_list = df.emotion.unique().tolist()
    lang_list = df.lang.unique().tolist()

    return df_list, speaker_list, sex_list, emotion_list, lang_list


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
