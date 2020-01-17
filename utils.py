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


def load_wavpath_text_speaker_sex_emotion_lang(hparams, split):
    '''
    split in {'train', 'val', 'test'}
    '''
    columns = ['wav_path', 'text', 'speaker', 'sex', 'emotion', 'lang']
    df_list = list()
    all_csv_paths = hparams.csv_data_paths.values()
    for csv_path in all_csv_paths:
        df = pd.read_csv(csv_path)
        df = df[df.split == split]
        df = df[columns]
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)

    speaker_list = df.speaker.unique().tolist()
    sex_list = df.sex.unique().tolist()
    emotion_list = df.emotion.unique().tolist()
    lang_list = df.lang.unique().tolist()

    # all_dbs != selected_dbs
    all_dbs = hparams.csv_data_paths.keys()
    selected_dbs = hparams.dbs
    if sorted(list(all_dbs)) != sorted(list(selected_dbs)):
        df_list = list()
        for db in selected_dbs:
            csv_path = hparams.csv_data_paths[db]
            df = pd.read_csv(csv_path)
            df = df[df.split == split]
            df = df[columns]
            df_list.append(df)
        df = pd.concat(df_list, ignore_index=True)

    row_list = df.values.tolist()

    return row_list, speaker_list, sex_list, emotion_list, lang_list


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
