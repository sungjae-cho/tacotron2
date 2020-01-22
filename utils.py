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


def load_wavpath_text_speaker_sex_emotion_lang(hparams, split, speaker, emotion):
    '''
    split in {'train', 'val', 'test'}
    speaker: str.
    emotion: str. {'amused', 'angry', 'neutral', 'disgusted', 'sleepy'}
    '''
    columns = ['wav_path', 'text', 'speaker', 'sex', 'emotion', 'lang']

    # Import all pontential DBs
    df_list = list()
    all_csv_paths = hparams.csv_data_paths.values()
    for csv_path in all_csv_paths:
        df = pd.read_csv(csv_path)
        df = df[df.split == split]
        df = df[columns]
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)

    # Create lists containing all pontential values.
    speaker_list = sorted(df.speaker.unique().tolist())
    sex_list = sorted(df.sex.unique().tolist())
    emotion_list = sorted(df.emotion.unique().tolist())
    lang_list = sorted(df.lang.unique().tolist())

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

    # Import particular emotions unless all emotions are used.
    if sorted(emotion_list) != sorted(hparams.emotions):
        df = df[df.emotion.isin(hparams.emotions)]

    # Select a particular speaker to be contained in df.
    if speaker is not None:
        df = df[df.speaker == speaker]
    if emotion is not None:
        df = df[df.emotion == emotion]

    # Upsampling datasets that do not have as many samples as the largest
    # dataset has to the extent that the largest one has.
    if split == 'train':
        df_size = df.groupby(['speaker', 'emotion']).size().reset_index(name='size')
        max_size = df_size['size'].max()

        df_dataset_list = list()
        for _, row in df_size.iterrows():
            row_speaker = row['speaker']
            row_emotion = row['emotion']
            row_size = row['size']

            df_spk_emo = df[(df.speaker == row_speaker) & (df.emotion == row_emotion)]

            n_dup_dfs = max_size // row_size
            n_rest_samples = max_size % row_size

            dup_dfs = [df_spk_emo] * n_dup_dfs + [df[:n_rest_samples]]
            df_dataset = pd.concat(dup_dfs, ignore_index=True)
            df_dataset_list.append(df_dataset)

        df = pd.concat(df_dataset_list, ignore_index=True)

    # Make all rows as elements of a list.
    row_list = df.values.tolist()

    return row_list, speaker_list, sex_list, emotion_list, lang_list

def load_pretrained_model(finetune_model, pretrained_path, model_optim=False,
        resume=False, freeze_pretrained=False, except_for=None):
    '''
    Author: Tae-Ho Kim (ktho894[at]gmail.com)
        load pretrained model to finetun_model.
        finetune_model (nn.Module): model will be fine tuned.
        pretrained_path (str): path to pretrained model. state_dict should be indexed.
        freeze_pretrained (bool or list): freeze all pretrained weight or given list.
    '''
    checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    feed_weight = checkpoint['state_dict'].copy()

    if type(freeze_pretrained) == list:
        frozen_weights = freeze_pretrained
    elif freeze_pretrained:
        frozen_weights = list(feed_weight.keys())
        if except_for is not None:
            for except_key in except_for:
                frozen_weights = [xx for xx in frozen_weights if except_key not in xx]
    else:
        frozen_weights = []

    finetune_state_dict = finetune_model.state_dict()

    # If pretrained weights have different shape or non-exist, then compensate it.
    for k, v in checkpoint['state_dict'].items():
        if k in finetune_state_dict.keys():
            cp_tensor = checkpoint['state_dict'][k]
            ft_tensor = finetune_state_dict[k]
            if cp_tensor.shape != ft_tensor.shape:
                feed_weight[k] = finetune_state_dict[k]
                if len(cp_tensor.shape) == len(ft_tensor.shape):
                    ft_import_dim = list()
                    for i_dim in range(len(cp_tensor.shape)):
                        if cp_tensor.shape[i_dim] <= ft_tensor.shape[i_dim]:
                            ft_import_dim.append(cp_tensor.shape[i_dim])
                    if len(ft_import_dim) == len(cp_tensor.shape):
                        d = ft_import_dim
                        if len(d) == 1:
                            feed_weight[k][:d[0]] = cp_tensor
                        elif len(d) == 2:
                            feed_weight[k][:d[0],:d[1]] = cp_tensor
                        elif len(d) == 3:
                            feed_weight[k][:d[0],:d[1],:d[2]] = cp_tensor
                        else:
                            print("Implement more dimensions")
                            exit()
                        print("{} weights are partially imported.".format(k))
                if k in frozen_weights:
                    frozen_weights.remove(k)
                resume = False
                print('[{}] Weights in model-will-be-finetuned is not in pretrained model. Resume is not available'.format(k))
            else:
                # k is in finetune network and shape is same.
                pass
        else:
            del feed_weight[k]
            if k in frozen_weights:
                frozen_weights.remove(k)

    # If new weights in finetune network is not in pretrained weight,
    for k, v in finetune_state_dict.items():
        if k not in feed_weight.keys():
            feed_weight[k] = v
    finetune_model.load_state_dict(feed_weight)

    # freeze params
    for name, param in finetune_model.named_parameters():
        if name in frozen_weights:
            param.requires_grad = False
        print('{}\t{}\t{}'.format(name, param.shape, param.requires_grad))

    print('loaded checkpoint %s' % (pretrained_path))

    if resume:
        start_epoch = checkpoint['epoch']
        model_optim.load_state_dict(checkpoint['optimizer'])
        for state in model_optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        plot_losses = checkpoint['plot_losses']
    else:
        start_epoch = 0
        plot_losses = []

    return finetune_model, model_optim, start_epoch, plot_losses



def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)

def get_spk_adv_inputs(padded_encoder_outputs, input_lengths):
    '''
    Get a batch for the speaker adversarial training module.

    PARAMS
    -----
    padded_encoder_outputs
    - type: torch.cuda.FloatTensor
    - size: [batch_size, max_text_len(=variable), encoder_embedding_dim]
    input_lengths
    - type: torch.cuda.LongTensor
    - size: [batch_size]

    RETURNS
    -----
    spk_adv_inputs
    - type: torch.cuda.LongTensor
    - size: [sum_max_text_len, encoder_embedding_dim]
    '''
    input_lengths = input_lengths.tolist()
    batch_size = padded_encoder_outputs.size(0)
    text_dim = padded_encoder_outputs.size(2)
    encoder_output_list = list()
    for i in range(batch_size):
        input_length = input_lengths[i]
        padded_encoder_output = padded_encoder_outputs[i,:,:]
        encoder_output = padded_encoder_output[:input_length,:]
        encoder_output_per_step = encoder_output.view(-1, text_dim)
        encoder_output_list.append(encoder_output_per_step)
    spk_adv_inputs = torch.cat(encoder_output_list)

    return spk_adv_inputs

def get_spk_adv_targets(speaker_targets, input_lengths):
    '''
    Get a batch for the speaker adversarial training module.

    PARAMS
    -----
    speaker_targets
    - type: torch.cuda.LongTensor
    - size: [batch_size]
    input_lengths
    - type: torch.cuda.LongTensor
    - size: [batch_size]

    RETURNS
    -----
    spk_adv_targets
    - type: torch.cuda.LongTensor
    - size: [sum_max_text_len]
    '''
    batch_size = input_lengths.size(0)
    input_lengths = input_lengths.tolist()
    spk_adv_target_list = list()
    for i in range(batch_size):
        input_length = input_lengths[i]
        spk_adv_target_list.append(speaker_targets[i].expand(input_length))
    spk_adv_targets = torch.cat(spk_adv_target_list)

    return spk_adv_targets
