import numpy as np
import random
from scipy.io.wavfile import read
import torch
import pandas as pd
import torch.nn.functional as F
from os import listdir
from os.path import isfile, join
from yin import compute_yin

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
    if len(data.shape) == 2:
        data = data.mean(axis=1) # for multichannel audios.
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def load_wavpath_text_speaker_sex_emotion_lang(hparams, split, speaker, emotion, random_seed):
    '''
    split in {'train', 'val', 'test'}
    speaker: str.
    emotion: str. {'amused', 'angry', 'neutral', 'disgusted', 'sleepy'}
    '''
    columns = ['wav_path', 'text', 'speaker', 'sex', 'emotion', 'lang']

    # Import all pontential DBs
    df_list = list()
    all_csv_paths = list()
    for db in hparams.all_dbs:
        all_csv_paths.append(hparams.csv_data_paths[db])

    for csv_path in all_csv_paths:
        df = pd.read_csv(csv_path)
        df = df[df.split == split]
        df = df[columns]
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)

    # Have only specified speakers and emotions in hparams.
    # These speakers and emotions will have to be used in this project
    # even though they are not used in this run.
    df = df[df.speaker.isin(hparams.all_speakers)]
    df = df[df.emotion.isin(hparams.all_emotions)]

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

    # Import particular speakers unless all emotions are used while this run.
    if sorted(speaker_list) != sorted(hparams.speakers):
        df = df[df.speaker.isin(hparams.speakers)]

    # Import particular emotions unless all emotions are used while this run..
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
        # Shuffle every time upsampling.
        df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

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
    random.seed(random_seed)
    random.shuffle(row_list)

    return row_list, speaker_list, sex_list, emotion_list, lang_list

def load_pretrained_model(finetune_model, pretrained_path, model_optim=False,
        resume=False, freeze_pretrained=False, except_for=['nothing']):
    '''
    Author: Tae-Ho Kim (ktho894[at]gmail.com)
        load pretrained model to finetun_model.
        finetune_model (nn.Module): model will be fine tuned.
        pretrained_path (str): path to pretrained model. state_dict should be indexed.
        freeze_pretrained (bool or list): freeze all pretrained weight or given list.
    '''
    checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    feed_weight = checkpoint['state_dict'].copy()

    #if type(freeze_pretrained) == list:
    #    frozen_weights = freeze_pretrained
    if freeze_pretrained:
        frozen_weights = list(feed_weight.keys())
        if except_for[0] != 'nothing':
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


def get_adv_inputs(padded_encoder_outputs, input_lengths):
    '''
    Get a batch for the speaker and emotion adversarial training modules.

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
    adv_inputs
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
    adv_inputs = torch.cat(encoder_output_list)

    return adv_inputs

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
    spk_adv_inputs = get_adv_inputs(padded_encoder_outputs, input_lengths)

    return spk_adv_inputs

def get_emo_adv_inputs(padded_encoder_outputs, input_lengths):
    '''
    Get a batch for the emotion adversarial training module.

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
    emo_adv_inputs = get_adv_inputs(padded_encoder_outputs, input_lengths)

    return emo_adv_inputs

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

def get_emo_adv_targets(emotion_targets, input_lengths):
    '''
    Get a batch for the emotion adversarial training module.

    PARAMS
    -----
    emotion_targets
    - type: torch.cuda.LongTensor
    - size: [batch_size]
    input_lengths
    - type: torch.cuda.LongTensor
    - size: [batch_size]

    RETURNS
    -----
    emo_adv_targets
    - type: torch.cuda.LongTensor
    - size: [sum_max_text_len]
    '''
    batch_size = input_lengths.size(0)
    input_lengths = input_lengths.tolist()
    emo_adv_target_list = list()
    for i in range(batch_size):
        input_length = input_lengths[i]
        emo_adv_target_list.append(emotion_targets[i].expand(input_length))
    emo_adv_targets = torch.cat(emo_adv_target_list)

    return emo_adv_targets

def hard_clip(input):
    return F.hardtanh(input + 0.5 , min_val=0, max_val=1)

def soft_clip(input, p=10):
    return torch.log(1 + torch.exp(p * input)) / p - torch.log(1 + torch.exp(p * (input - 1))) / p

def get_clsf_report(confusion_matrix, target_names, all_target_names):
    report_dict = dict()
    cm = confusion_matrix

    # Compute accuracy
    accuracy = cm.diagonal().sum() / cm.sum()
    report_dict['accuracy'] = accuracy

    for str_target in  target_names:
        i_class = all_target_names.index(str_target)
        tp = cm[i_class, i_class]
        tn = cm[i_class,:].sum() - tp
        fp = cm[:,i_class].sum() - tp
        fn = cm.sum() - tp - tn - fp
        if (tp + fp) == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
        if (tp + tn) == 0:
            recall = 0.0
        else:
            recall = tp / (tp + tn)
        if precision == 0 or recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 / (1 / precision + 1 / recall)

        report_dict[str_target] = {
            "precision":precision,
            "recall":recall,
            "f1-score":f1_score
        }

    return report_dict

def get_KLD_weight(iteration, hparams):
    if hparams.KLD_weight_scheduling == 'fixed':
        return hparams.res_en_KLD_weight
    elif hparams.KLD_weight_scheduling == 'pulse':
        return get_pulse_KLD_weight(iteration, hparams)
    elif hparams.KLD_weight_scheduling == 'cycle_linear':
        return get_cycle_linear_KLD_weight(iteration, hparams)


def get_pulse_KLD_weight(iteration, hparams):
    '''
    Rference source: https://github.com/rishikksh20/gmvae_tacotron/blob/master/tacotron/utils/util.py#L24
    Adapt the refence source to built-in functions.
    '''
    warm_up_step = hparams.KLD_weight_warm_up_step
    init_KLD_weight = hparams.init_KLD_weight
    KLD_weight_cof = hparams.KLD_weight_cof

    # Get w1.
    w1 = 0.0
    if iteration <= warm_up_step:
        if (iteration % 100) < 1:
            w1 = init_KLD_weight + (iteration / 100  * KLD_weight_cof)

    # Get w2.
    w2 = 0.0
    if iteration > warm_up_step:
        if (iteration % 400) < 1:
            w2 = init_KLD_weight + ((iteration - warm_up_step) / 400 * KLD_weight_cof) + (warm_up_step / 100 * KLD_weight_cof)

    return np.max([w1, w2])

def get_cycle_linear_KLD_weight(iteration, hparams):
    period = hparams.cycle_KLDW_period
    ratio = hparams.cycle_KLDW_ratio
    start = hparams.cycle_KLDW_min
    stop = hparams.cycle_KLDW_max

    early_period = int(period * ratio)
    iter_in_period = iteration % period

    if iter_in_period < early_period:
        gradient = (stop - start) / early_period
        KLD_weight = iter_in_period * gradient
    else:
        KLD_weight = stop

    return KLD_weight

def get_files(dir_path):
    files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    return files

def get_checkpoint_iteration(checkpoint_name):
    cp_iter = checkpoint_name.split('-')[0].split('_')[1]
    return cp_iter

def get_checkpoint_iter2path(outdir, prj_name, run_name, cp_iter):
    cp_dir_path = join(outdir, prj_name, run_name)
    cp_files = get_files(cp_dir_path)
    for cp_file in cp_files:
        i = get_checkpoint_iteration(cp_file)
        if int(i) == int(cp_iter):
            cp_path = join(cp_dir_path, cp_file)
            break

    return cp_path

def discretize_att_w(attention_weights, discrete_bound=False):
    '''
    PARAMS
    -----
    attention_weights: Attention weights of one batch.
    - torch.Tensor. Size == (batches, max_encoding_steps).
    discrete_bound: list. float. length == 2.
    - discrete_bound[0]: whether to discretize nonmax values.
    - discrete_bound[1]: whether to discretize the max values.

    RETURNS
    -----
    discrete_att_w: Discretized attention weights of one batch.
    - torch.Tensor. Size == (batches, max_encoding_steps).
    '''
    max_expanded = torch.max(attention_weights, dim=1).values.unsqueeze(-1).expand(attention_weights.size())
    discrete_att_w = attention_weights
    if isinstance(discrete_bound, bool) and discrete_bound:
        discrete_att_w = discrete_att_w.masked_fill((attention_weights != max_expanded), 0)
        discrete_att_w = discrete_att_w.masked_fill((attention_weights == max_expanded), 1)
    if isinstance(discrete_bound, list):
        if discrete_bound[0]:
            discrete_att_w = discrete_att_w.masked_fill((attention_weights != max_expanded), 0)
        if discrete_bound[1]:
            discrete_att_w = discrete_att_w.masked_fill((attention_weights == max_expanded), 1)

    return discrete_att_w

def get_f0(wav, sampling_rate=22050, frame_length=1024, hop_length=256,
        f0_min=100, f0_max=300, harm_thresh=0.1):
    f0, harmonic_rates, argmins, times = compute_yin(
        wav, sampling_rate, frame_length, hop_length, f0_min, f0_max,
        harm_thresh)
    pad = int((frame_length / hop_length) / 2)
    f0 = [0.0] * pad + f0 + [0.0] * pad

    f0 = np.array(f0, dtype=np.float32)
    return f0

def get_text_durations(alignment):
    '''
    Params
    -----
    alignment: A stack of attention weights at every decoding step.
    - type: numpy.ndarray.
    - dtype: float.
    - shape: (mel_steps, text_steps)

    Returns
    -----
    text_durations: # Mel frames where a grapheme/phoneme is continuously spoken.
    - type: list.
    - dtype: int.
    x_chunks: A list of chunks containing mel steps where a grapheme/phoneme is continuously spoken.
    - type: list.
    - dtype: list. that contains integers.
    att_text_seq: A sequence of maximally attended text.
    - type: list.
    - dtype: int.
    '''
    (mel_steps, text_steps) = alignment.shape
    text_durations = list()
    x_chunks = list()
    # y cantains maximally attended text locations.
    x = list(range(mel_steps))
    att_text_seq = y = alignment.argmax(axis=1)
    prev_yi = -1
    for i in range(mel_steps):
        xi = x[i]
        yi = y[i]
        if i > 0:
            prev_yi = y[i-1]
        if i < mel_steps -1:
            next_yi = y[i+1]

        if (prev_yi != yi) or (i == 0):
            start = True
        else:
            start = False

        if (next_yi != yi) or (i == mel_steps - 1):
            end = True
        else:
            end = False

        if start:
            xi_list = list()

        xi_list.append(xi)

        if end:
            yi_list = y[xi_list[0]:xi_list[-1]+1]
            duration = len(xi_list)
            text_durations += [duration]*len(xi_list)
            x_chunks.append(xi_list)

    return text_durations, x_chunks, att_text_seq
