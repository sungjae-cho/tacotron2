import os
import time
import argparse
import math
import signal
import sys
import random
import numpy as np
import pandas as pd
import wandb
from numpy import finfo
from apex import amp
#from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix
from torch.nn import MSELoss, L1Loss

from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss, KLD_loss, TotalLoss
from logger import Tacotron2Logger
from hparams import create_hparams
from measures import forward_attention_ratio, attention_ratio, attention_range_ratio, multiple_attention_ratio
from measures import get_mel_length, get_mel_lengths
from measures import get_attention_quality
from utils import get_spk_adv_targets, get_emo_adv_targets, load_pretrained_model, get_clsf_report, \
    get_KLD_weight, get_checkpoint_iter2path
from adam_step import adam_step

def reduce_tensor(tensor, reduce_op='mean'):
    rt = tensor.cuda().detach().clone()
    if reduce_op == 'sum':
        dist.all_reduce(rt, op=dist.reduce_op.SUM)
    if reduce_op == 'mean':
        dist.all_reduce(rt, op=dist.reduce_op.SUM)
        rt /= dist.get_world_size()
    if reduce_op == 'max':
        dist.all_reduce(rt, op=dist.reduce_op.MAX)
    if reduce_op == 'min':
        dist.all_reduce(rt, op=dist.reduce_op.MIN)
    return rt

def reduce_scalar(scalar, reduce_op='mean'):
    tensor = torch.FloatTensor([scalar]).cuda()
    rs = reduce_tensor(tensor, reduce_op).item()
    return rs

def gather_all_tensor(tensor):
    tensor = tensor.cuda().detach().contiguous()
    tensor_list = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, tensor)
    one_tensor = torch.cat(tensor_list)
    return one_tensor

def init_distributed(hparams, n_gpus, rank, group_name, first_device):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device((rank + first_device) % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method="{}:{}".format(hparams.dist_url, hparams.dist_port),
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams, 'train')

    valsets = dict()
    all_valset = TextMelLoader(hparams, 'val')
    valsets[('all', 'all')] = all_valset
    for speaker in hparams.speakers:
        for emotion in hparams.emotions:
            valset = TextMelLoader(hparams, 'val', speaker, emotion)
            if len(valset) != 0:
                valsets[(speaker, emotion)] = valset

    collate_fn = TextMelCollate(hparams)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, trainset, valsets, collate_fn





def prepare_directories_and_logger(hparams, output_directory, log_directory, rank,
                                   run_name, prj_name, resume, model):
    if rank == 0:
        if not os.path.isdir(os.path.join(output_directory, prj_name, run_name)):
            os.makedirs(os.path.join(output_directory, prj_name, run_name))
            os.chmod(os.path.join(output_directory, prj_name, run_name), 0o775)
        logger = Tacotron2Logger(hparams, run_name, prj_name,
            os.path.join(log_directory, prj_name, run_name), model, resume)
    else:
        logger = None
    return logger


def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler, logger,
        rank, hparams):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model = load_pretrained_model(model, checkpoint_path,
        freeze_pretrained=hparams.freeze_pretrained,
        except_for=hparams.freeze_except_for)[0]
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    if 'amp_scaling_state' in checkpoint_dict.keys():
        optimizer.state = {} # Because of the following recommendation https://github.com/NVIDIA/apex/issues/480#issuecomment-587154020
        amp.load_state_dict(checkpoint_dict['amp_scaling_state'])
    '''if hparams.fp16_run:
        amp_scaler.load_state_dict(checkpoint_dict['amp_scaler'])'''
    learning_rate = checkpoint_dict['learning_rate']
    if hparams.use_saved_learning_rate:
        learning_rate = checkpoint_dict['learning_rate']
    else:
        learning_rate = hparams.learning_rate
    print("Loaded learning_rate=", learning_rate)
    if ('lr_scheduler' in checkpoint_dict.keys()) and hparams.use_saved_learning_rate:
        lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
    else:
        lr_scheduler.load_state_dict({'base_lrs':[learning_rate]})
    iteration = checkpoint_dict['iteration']
    float_epoch = checkpoint_dict['float_epoch']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    if rank == 0:
        dict_vars = checkpoint_dict['training_epoch_variables']
        logger.set_training_epoch_variables(dict_vars)
    return model, optimizer, learning_rate, iteration, float_epoch, checkpoint_dict


def save_checkpoint(hparams, model, optimizer, learning_rate, iteration, float_epoch, lr_scheduler,
        logger, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    checkpoint_dict = {
        'iteration': iteration,
        'float_epoch': float_epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'learning_rate': learning_rate,
        'lr_scheduler': lr_scheduler.state_dict(),
        'training_epoch_variables': logger.get_training_epoch_variables(),
    }
    if hparams.fp16_run:
        checkpoint_dict['amp_scaling_state'] = amp.state_dict()
        '''checkpoint_dict['amp_scaler'] = amp_scaler.state_dict()'''
    torch.save(checkpoint_dict, filepath)


def fill_synth_dict(hparams, synth_dict, idx, inputs, outputs,
        batch_attention_measures_tf, batch_attention_measures_fr,
        temp_prosody_hiddens):
    # Inputs
    (input_lengths, text_padded, speakers, emotion_input_vectors, \
        text_raw) = inputs
    text_length = input_lengths[idx].item()
    synth_dict['text_sequence'] = text_padded[idx,:text_length]
    synth_dict['text_raw'] = text_raw[idx]
    synth_dict['speaker_tensor'] = speakers[idx]
    synth_dict['emotion_input_tensor'] = emotion_input_vectors[idx]

    # Outputs
    (output_lengths, gate_outputs_fr, end_points_fr,
        mel_padded, mel_outputs_postnet, mel_outputs_postnet_fr,
        alignments, alignments_fr,
        prosody_tf, prosody_fr) = outputs
    mel_length = output_lengths[idx].item()
    mel_length_fr = get_mel_length(gate_outputs_fr[idx])
    if end_points_fr is not None:
        mel_length_fr = min(mel_length_fr, end_points_fr[idx])
    synth_dict['mel_true'] = mel_padded[idx,:,:mel_length]
    synth_dict['mel_output_tf'] = mel_outputs_postnet[idx,:,:mel_length]
    synth_dict['mel_output_fr'] = mel_outputs_postnet_fr[idx,:,:mel_length_fr]
    synth_dict['alignment_tf'] = alignments[idx,:mel_length,:text_length]
    synth_dict['alignment_fr'] = alignments_fr[idx,:mel_length_fr,:text_length]
    prosody_ref_tf, prosody_pred_tf = prosody_tf
    prosody_ref_fr, prosody_pred_fr = prosody_fr
    if hparams.reference_encoder:
        synth_dict['prosody_ref_tf'] = prosody_ref_tf[idx,:mel_length,:]
        synth_dict['prosody_ref_fr'] = prosody_ref_fr[idx,:mel_length_fr,:]
    if hparams.prosody_predictor:
        synth_dict['prosody_pred_tf'] = prosody_pred_tf[idx,:mel_length,:]
        synth_dict['prosody_pred_fr'] = prosody_pred_fr[idx,:mel_length_fr,:]


    # Teacher forcing attention measures
    (batch_attention_quality,
        batch_ar, batch_letter_ar, batch_punct_ar, batch_blank_ar,
        batch_arr, batch_mar) = batch_attention_measures_tf
    synth_dict['attention_quality_tf'] = batch_attention_quality[idx].item()
    synth_dict['ar_tf'] = batch_ar[idx].item()
    synth_dict['letter_ar_tf'] = batch_letter_ar[idx].item()
    synth_dict['punct_ar_tf'] = batch_punct_ar[idx].item()
    synth_dict['blank_ar_tf'] = batch_blank_ar[idx].item()
    synth_dict['arr_tf'] = batch_arr[idx].item()
    synth_dict['mar_tf'] = batch_mar[idx].item()

    # Free running attention measures
    (batch_attention_quality_fr,
        batch_ar_fr, batch_letter_ar_fr, batch_punct_ar_fr, batch_blank_ar_fr,
        batch_arr_fr, batch_mar_fr) = batch_attention_measures_fr
    synth_dict['attention_quality_fr'] = batch_attention_quality_fr[idx].item()
    synth_dict['ar_fr'] = batch_ar_fr[idx].item()
    synth_dict['letter_ar_fr'] = batch_letter_ar_fr[idx].item()
    synth_dict['punct_ar_fr'] = batch_punct_ar_fr[idx].item()
    synth_dict['blank_ar_fr'] = batch_blank_ar_fr[idx].item()
    synth_dict['arr_fr'] = batch_arr_fr[idx].item()
    synth_dict['mar_fr'] = batch_mar_fr[idx].item()

    # temp_prosody_hiddens
    if hparams.reference_encoder == 'Glob2Temp':
        temp_prosody_hiddens_tf, temp_prosody_hiddens_fr = temp_prosody_hiddens
        synth_dict['temp_prosody_hiddens_tf'] = temp_prosody_hiddens_tf[idx,:mel_length,:]
        synth_dict['temp_prosody_hiddens_fr'] = temp_prosody_hiddens_tf[idx,:mel_length_fr,:]


def compute_alignments(pretrained_path, hparams):
    """Handles all the validation scoring and printing"""
    if hparams.distributed_run:
        return

    torch.cuda.empty_cache()
    model = load_model(hparams)
    model = load_pretrained_model(model, pretrained_path)[0]
    model.eval()

    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams, 'train')
    valset = TextMelLoader(hparams, 'val')
    testset = TextMelLoader(hparams, 'test')
    set_types = ['train', 'val', 'test']
    datasets = [trainset, valset, testset]
    collate_fn = TextMelCollate(hparams)

    # Logging lists
    wav_path_list = list()
    text_raw_list = list()
    far_list = list()
    ar_list = list()
    letter_ar_list = list()
    punct_ar_list = list()
    blank_ar_list = list()
    arr_list = list()
    mar_list = list()
    aq_list = list()


    for set_type, dataset in zip(set_types, datasets):
        print("{} set size: {}".format(set_type, len(dataset)))

    for set_type, dataset in zip(set_types, datasets):
        with torch.no_grad():
            dataset_loader = DataLoader(dataset, sampler=None, num_workers=1,
                                    shuffle=False, batch_size=hparams.batch_size,
                                    pin_memory=False, collate_fn=collate_fn)

            #######################
            # TEACHER FORCING #####
            #######################
            # forward_attention_ratio
            batch_far_list = list()
            # attention_ratio
            batch_ar_list = list()
            batch_letter_ar_list = list()
            batch_punct_ar_list = list()
            batch_blank_ar_list = list()
            # attention_range_ratio
            batch_arr_list = list()
            # multiple_attention_ratio
            batch_mar_list = list()
            # attention_quality
            batch_aq_list = list()

            print("Start to iterate {} set".format(set_type))
            for i, batch in tqdm(enumerate(dataset_loader), total=len(dataset)):
                # Parse inputs of each batch
                x, y, etc = model.parse_batch(batch)
                text_padded, input_lengths, mel_padded, max_len, output_lengths = x
                mel_padded, gate_padded = y
                speakers, sex, emotion_input_vectors, emotion_targets, lang, \
                    text_raw, wav_paths = etc

                ############################################################
                # TEACHER FORCING #####
                # Forward propagation by teacher forcing
                (y_pred, y_pred_speakers, y_pred_emotions, y_pred_res_en,
                    att_means) = model(
                        x, speakers, emotion_input_vectors,
                        zero_res_en=hparams.val_tf_zero_res_en)

                # Forward propagtion results
                mel_outputs, mel_outputs_postnet, gate_outputs, alignments, \
                    prosody_ref, prosody_pred, end_points = y_pred

                # [M1] forward_attention_ratio
                _, batch_far = forward_attention_ratio(alignments, input_lengths, output_lengths=output_lengths, mode_mel_length="ground_truth")
                # [M2] attention_ratio
                ar_pairs = attention_ratio(alignments, input_lengths, text_padded, output_lengths=output_lengths, mode_mel_length="ground_truth")
                batch_ar = ar_pairs[0][1]
                batch_letter_ar = ar_pairs[1][1]
                batch_punct_ar = ar_pairs[2][1]
                batch_blank_ar = ar_pairs[3][1]
                # [M3] attention_range_ratio
                _, batch_arr = attention_range_ratio(alignments, input_lengths, output_lengths=output_lengths, mode_mel_length="ground_truth")
                # [M4] multiple_attention_ratio
                _, batch_mar = multiple_attention_ratio(alignments, input_lengths, text_padded, output_lengths=output_lengths, mode_mel_length="ground_truth")
                # [M_total] Attention quality
                batch_attention_quality = get_attention_quality(batch_far, batch_mar, batch_letter_ar)

                # Append attention measures into lists.
                ## Attention measures: Teacher-forcing
                wav_path_list += wav_paths
                text_raw_list += text_raw
                far_list += batch_far.squeeze().cpu().tolist()
                ar_list += batch_ar.squeeze().cpu().tolist()
                letter_ar_list += batch_letter_ar.squeeze().cpu().tolist()
                punct_ar_list += batch_punct_ar.squeeze().cpu().tolist()
                blank_ar_list += batch_blank_ar.squeeze().cpu().tolist()
                arr_list += batch_arr.squeeze().cpu().tolist()
                mar_list += batch_mar.squeeze().cpu().tolist()
                aq_list += batch_attention_quality.squeeze().cpu().tolist()
                # End of the iteration
                ############################################################
            # End of the datset
            ####################################################################

    csv_path = 'metadata/bc2013_att_measures.csv'
    df = pd.DataFrame({
        'wav_path':wav_path_list,
        'text':text_raw_list,
        'aq':aq_list,
        'far':far_list,
        'ar':ar_list,
        'letter_ar':letter_ar_list,
        'punct_ar':punct_ar_list,
        'blank_ar':blank_ar_list,
        'arr':arr_list,
        'mar':mar_list,
    })
    df.to_csv(csv_path, index=False)
    print("Results are saved in {}".format(csv_path))


def validate(model, criterion, trainset, valsets, iteration, epoch, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank, hparams):
    """Handles all the validation scoring and printing"""
    for val_type, valset in valsets.items():
        #val_type: tuple. (str_speaker, str_emotion).
        '''if val_type == ('all', 'all'):
            continue'''
        torch.cuda.empty_cache()
        model.eval()
        with torch.no_grad():
            val_sampler = DistributedSampler(valset) if distributed_run else None
            val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                    shuffle=False, batch_size=batch_size,
                                    pin_memory=False, collate_fn=collate_fn)

            criterion = criterion
            val_loss_mel = 0.0
            val_loss_gate = 0.0
            val_loss_KLD = 0.0
            val_loss_ref_enc = 0.0
            val_loss_spk_adv = 0.0
            val_loss_emo_adv = 0.0
            val_loss_att_means = 0.0
            val_loss = 0.0

            sum_output_lengths = 0.0

            if hparams.residual_encoder:
                list_residual_encoding = list()
                list_mu = list()
                list_logvar = list()

            if hparams.prosody_predictor:
                np_sum_prosody_pred_dims = np.zeros(hparams.prosody_dim)

            if hparams.reference_encoder:
                np_sum_prosody_ref_dims = np.zeros(hparams.prosody_dim)

            if hparams.speaker_adversarial_training and val_type == ('all', 'all'):
                np_spk_cm_sum = np.zeros((len(trainset.speaker_list), len(trainset.speaker_list)))

            if hparams.emotion_adversarial_training and val_type == ('all', 'all'):
                np_emo_cm_sum = np.zeros((len(trainset.emotion_list), len(trainset.emotion_list)))

            #######################
            # TEACHER FORCING #####
            #######################
            # forward_attention_ratio
            batch_far_list = list()
            # attention_ratio
            batch_ar_list = list()
            batch_letter_ar_list = list()
            batch_punct_ar_list = list()
            batch_blank_ar_list = list()
            # attention_range_ratio
            batch_arr_list = list()
            # multiple_attention_ratio
            batch_mar_list = list()

            # sample to synthesize\
            i_valset = random.randint(0, len(valset) - 1)
            i_batch_rand, i_sample_rand = divmod(i_valset, hparams.batch_size)

            #sample_with_worst_attention_quality
            min_attention_quality_tf = 2
            synth_dict_min_aq_tf = dict()
            min_attention_quality_fr = 2
            synth_dict_min_aq_fr = dict()


            ####################
            # FREE RUNNING #####
            ####################
            # forward_attention_ratio
            batch_far_fr_list = list()
            # attention_ratio
            batch_ar_fr_list = list()
            batch_letter_ar_fr_list = list()
            batch_punct_ar_fr_list = list()
            batch_blank_ar_fr_list = list()
            # attention_range_ratio
            batch_arr_fr_list = list()
            # multiple_attention_ratio
            batch_mar_fr_list = list()

            for i, batch in enumerate(val_loader):
                # Parse inputs of each batch
                x, y, etc = model.parse_batch(batch)
                text_padded, input_lengths, mel_padded, max_len, output_lengths = x
                mel_padded, gate_padded = y
                speakers, sex, emotion_input_vectors, emotion_targets, lang, \
                    text_raw, wav_paths = etc

                ############################################################
                # TEACHER FORCING #####
                # Forward propagation by teacher forcing
                (y_pred, y_pred_speakers, y_pred_emotions, y_pred_res_en,
                    att_means) = model(
                        x, speakers, emotion_input_vectors,
                        stop_prediction2=hparams.val_tf_stop_pred2,
                        zero_res_en=hparams.val_tf_zero_res_en)

                # Forward propagtion results
                mel_outputs, mel_outputs_postnet, gate_outputs, alignments, \
                    prosody_ref, prosody_pred, end_points = y_pred
                logit_speakers, prob_speakers, int_pred_speakers = y_pred_speakers
                logit_emotions, prob_emotions, int_pred_emotions = y_pred_emotions
                residual_encoding, mu, logvar = y_pred_res_en
                prosody = prosody_ref, prosody_pred
                temp_prosody_hiddens_tf = model.get_temp_prosody_hiddens()

                # Compute stop gate accuracy
                np_output_lengths = output_lengths.cpu().numpy()
                mel_lengths = get_mel_lengths(gate_outputs)
                np_mel_lengths = mel_lengths.cpu().numpy()
                gate_accuracy = accuracy_score(np_output_lengths, np_mel_lengths)
                # Compute stop gate MAE(pred_lengths, true_lengths)
                gate_mae = mean_absolute_error(np_output_lengths, np_mel_lengths)
                if end_points is not None:
                    end_point_mae = mean_absolute_error(np.asarray(end_points), np_mel_lengths)
                else:
                    end_point_mae = 0


                if hparams.speaker_adversarial_training:
                    spk_adv_targets = get_spk_adv_targets(speakers, input_lengths)
                if hparams.emotion_adversarial_training:
                    emo_adv_targets = get_emo_adv_targets(emotion_targets, input_lengths)

                (loss, loss_taco2, loss_mel, loss_gate, loss_KLD, loss_ref_enc,
                    loss_spk_adv, loss_emo_adv, loss_att_means
                    ) = criterion(
                        mel_outputs, mel_outputs_postnet, mel_padded,
                        gate_outputs, gate_padded,
                        y_pred, y,
                        mu, logvar,
                        prosody_pred, prosody_ref,
                        logit_speakers, speakers,
                        logit_emotions, emotion_targets,
                        att_means, input_lengths,
                        iteration
                    )

                # [M1] forward_attention_ratio
                _, batch_far = forward_attention_ratio(alignments, input_lengths, output_lengths=output_lengths, mode_mel_length="ground_truth")
                # [M2] attention_ratio
                ar_pairs = attention_ratio(alignments, input_lengths, text_padded, output_lengths=output_lengths, mode_mel_length="ground_truth")
                batch_ar = ar_pairs[0][1]
                batch_letter_ar = ar_pairs[1][1]
                batch_punct_ar = ar_pairs[2][1]
                batch_blank_ar = ar_pairs[3][1]
                # [M3] attention_range_ratio
                _, batch_arr = attention_range_ratio(alignments, input_lengths, output_lengths=output_lengths, mode_mel_length="ground_truth")
                # [M4] multiple_attention_ratio
                _, batch_mar = multiple_attention_ratio(alignments, input_lengths, text_padded, output_lengths=output_lengths, mode_mel_length="ground_truth")
                # [M_total] Attention quality
                batch_attention_quality = get_attention_quality(batch_far, batch_mar, batch_letter_ar)

                ############################################################
                # FREE RUNNING #####
                # Forward propagation by free running, i.e., feeding previous outputs to the current inputs.
                if hparams.reference_encoder in hparams.reference_encoders_taking_mels_at_inference:
                    inputs_fr = x
                else:
                    inputs_fr = text_padded, input_lengths

                _, mel_outputs_postnet_fr, gate_outputs_fr, alignments_fr, prosody_ref_fr, prosody_pred_fr, end_points_fr = model(inputs_fr, speakers, emotion_input_vectors, teacher_forcing=False, stop_prediction2=hparams.val_fr_stop_pred2)
                prosody_fr = prosody_ref_fr, prosody_pred_fr
                temp_prosody_hiddens_fr = model.get_temp_prosody_hiddens()

                # Computing attention measures.
                # [M1] forward_attention_ratio
                _, batch_far_fr = forward_attention_ratio(alignments_fr, input_lengths, gate_outputs=gate_outputs_fr, pred_end_points=end_points_fr, mode_mel_length="stop_token")
                #_, batch_far_fr = forward_attention_ratio(alignments_fr, input_lengths, output_lengths=torch.IntTensor(end_points_fr), mode_mel_length="ground_truth")
                # [M2] attention_ratio
                ar_fr_pairs = attention_ratio(alignments_fr, input_lengths, text_padded, gate_outputs=gate_outputs_fr, pred_end_points=end_points_fr, mode_mel_length="stop_token")
                #ar_fr_pairs = attention_ratio(alignments_fr, input_lengths, text_padded, output_lengths=torch.IntTensor(end_points_fr), mode_mel_length="ground_truth")
                batch_ar_fr = ar_fr_pairs[0][1]
                batch_letter_ar_fr = ar_fr_pairs[1][1]
                batch_punct_ar_fr = ar_fr_pairs[2][1]
                batch_blank_ar_fr = ar_fr_pairs[3][1]
                # [M3] attention_range_ratio
                _, batch_arr_fr = attention_range_ratio(alignments_fr, input_lengths, gate_outputs=gate_outputs_fr, pred_end_points=end_points_fr, mode_mel_length="stop_token")
                #_, batch_arr_fr = attention_range_ratio(alignments_fr, input_lengths, output_lengths=torch.IntTensor(end_points_fr), mode_mel_length="ground_truth")
                # [M4] multiple_attention_ratio
                _, batch_mar_fr = multiple_attention_ratio(alignments_fr, input_lengths, text_padded, gate_outputs=gate_outputs_fr, pred_end_points=end_points_fr, mode_mel_length="stop_token")
                #_, batch_mar_fr = multiple_attention_ratio(alignments_fr, input_lengths, text_padded, output_lengths=torch.IntTensor(end_points_fr), mode_mel_length="ground_truth")
                # [M_total] Attention quality
                batch_attention_quality_fr = get_attention_quality(batch_far_fr, batch_mar_fr, batch_letter_ar_fr)

                if hparams.speaker_adversarial_training:
                    np_target_speakers = spk_adv_targets.cpu().numpy()
                    np_pred_speakers = int_pred_speakers.cpu().numpy()
                    np_speaker_array = np.arange(len(trainset.speaker_list))
                    np_spk_cm = confusion_matrix(
                        np.append(np_target_speakers, np_speaker_array),
                        np.append(np_pred_speakers, np_speaker_array),
                        labels=list(np_speaker_array))
                    np_spk_cm -= np.eye(np_speaker_array.shape[0], dtype=np_spk_cm.dtype)
                    tensor_spk_cm = torch.IntTensor(np_spk_cm).cuda()

                if hparams.emotion_adversarial_training:
                    np_target_emotions = emo_adv_targets.cpu().numpy()
                    np_pred_emotions = int_pred_emotions.cpu().numpy()
                    np_emotion_array = np.arange(len(trainset.emotion_list))
                    np_emo_cm = confusion_matrix(
                        np.append(np_target_emotions, np_emotion_array),
                        np.append(np_pred_emotions, np_emotion_array),
                        labels=list(np_emotion_array))
                    np_emo_cm -= np.eye(np_emotion_array.shape[0], dtype=np_emo_cm.dtype)
                    tensor_emo_cm = torch.IntTensor(np_emo_cm).cuda()

                if hparams.reference_encoder:
                    sum_prosody_ref_dims = prosody_ref.sum(dim=(0,1))

                if hparams.prosody_predictor:
                    sum_prosody_pred_dims = prosody_pred.sum(dim=(0,1))

                if distributed_run:
                    # Losses
                    reduced_val_loss_mel = reduce_tensor(loss_mel).item()
                    reduced_val_loss_gate = reduce_tensor(loss_gate).item()
                    reduced_loss_KLD = reduce_tensor(loss_KLD).item()
                    reduced_loss_ref_enc = reduce_tensor(loss_ref_enc).item()
                    reduced_val_loss_spk_adv = reduce_tensor(loss_spk_adv).item()
                    reduced_val_loss_emo_adv = reduce_tensor(loss_emo_adv).item()
                    reduced_val_loss_att_means = reduce_tensor(loss_att_means).item()
                    reduced_val_loss = reduce_tensor(loss).item()
                    # Inputs
                    input_lengths = gather_all_tensor(input_lengths)
                    if False:
                        # Lines in this block cause communication failure.
                        text_padded = gather_all_tensor(text_padded)
                    speakers = gather_all_tensor(speakers)
                    emotion_input_vectors = gather_all_tensor(emotion_input_vectors)
                    # Outputs
                    output_lengths = gather_all_tensor(output_lengths)
                    gate_outputs_fr = gather_all_tensor(gate_outputs_fr)
                    if False:
                        # Lines in this block cause communication failure.
                        mel_padded = gather_all_tensor(mel_padded)
                        mel_outputs_postnet = gather_all_tensor(mel_outputs_postnet)
                        mel_outputs_postnet_fr = gather_all_tensor(mel_outputs_postnet_fr)
                        alignments = gather_all_tensor(alignments)
                        alignments_fr = gather_all_tensor(alignments_fr)
                    # Attention measures: Teacher-forcing
                    batch_far = gather_all_tensor(batch_far)
                    batch_ar = gather_all_tensor(batch_ar)
                    batch_letter_ar = gather_all_tensor(batch_letter_ar)
                    batch_punct_ar = gather_all_tensor(batch_punct_ar)
                    batch_blank_ar = gather_all_tensor(batch_blank_ar)
                    batch_arr = gather_all_tensor(batch_arr)
                    batch_mar = gather_all_tensor(batch_mar)
                    batch_attention_quality = gather_all_tensor(batch_attention_quality)
                    # Attention measures: Free-running
                    batch_far_fr = gather_all_tensor(batch_far_fr)
                    batch_ar_fr = gather_all_tensor(batch_ar_fr)
                    batch_letter_ar_fr = gather_all_tensor(batch_letter_ar_fr)
                    batch_punct_ar_fr = gather_all_tensor(batch_punct_ar_fr)
                    batch_blank_ar_fr = gather_all_tensor(batch_blank_ar_fr)
                    batch_arr_fr = gather_all_tensor(batch_arr_fr)
                    batch_mar_fr = gather_all_tensor(batch_mar_fr)
                    batch_attention_quality_fr = gather_all_tensor(batch_attention_quality_fr)
                    # Things concerning the residual encoder
                    if hparams.residual_encoder:
                        residual_encoding = gather_all_tensor(residual_encoding)
                        mu = gather_all_tensor(mu)
                        logvar = gather_all_tensor(logvar)
                    if hparams.prosody_predictor:
                        sum_prosody_pred_dims = reduce_tensor(sum_prosody_pred_dims, 'sum')
                    if hparams.reference_encoder:
                        sum_prosody_ref_dims = reduce_tensor(sum_prosody_ref_dims, 'sum')
                    # Things concerning speaker adversarial training
                    if hparams.speaker_adversarial_training:
                        tensor_spk_cm = reduce_tensor(tensor_spk_cm, 'sum')
                    # Things concerning emotion adversarial training
                    if hparams.emotion_adversarial_training:
                        tensor_emo_cm = reduce_tensor(tensor_emo_cm, 'sum')
                else:
                    reduced_val_loss_mel = loss_mel.item()
                    reduced_val_loss_gate = loss_gate.item()
                    reduced_loss_KLD = loss_KLD.item()
                    reduced_loss_ref_enc = loss_ref_enc.item()
                    reduced_val_loss_spk_adv = loss_spk_adv.item()
                    reduced_val_loss_emo_adv = loss_emo_adv.item()
                    reduced_val_loss_att_means = loss_att_means.item()
                    reduced_val_loss = loss.item()

                # Accumulate losses.
                val_loss_mel += reduced_val_loss_mel
                val_loss_gate += reduced_val_loss_gate
                val_loss_KLD += reduced_loss_KLD
                val_loss_ref_enc += reduced_loss_ref_enc
                val_loss_spk_adv += reduced_val_loss_spk_adv
                val_loss_emo_adv += reduced_val_loss_emo_adv
                val_loss_att_means += reduced_val_loss_att_means
                val_loss += reduced_val_loss

                # Append attention measures into lists.
                ## Attention measures: Teacher-forcing
                batch_far_list.append(batch_far)
                batch_ar_list.append(batch_ar)
                batch_letter_ar_list.append(batch_letter_ar)
                batch_punct_ar_list.append(batch_punct_ar)
                batch_blank_ar_list.append(batch_blank_ar)
                batch_arr_list.append(batch_arr)
                batch_mar_list.append(batch_mar)
                ## Attention measures: Free-running
                batch_far_fr_list.append(batch_far_fr)
                batch_ar_fr_list.append(batch_ar_fr)
                batch_letter_ar_fr_list.append(batch_letter_ar_fr)
                batch_punct_ar_fr_list.append(batch_punct_ar_fr)
                batch_blank_ar_fr_list.append(batch_blank_ar_fr)
                batch_arr_fr_list.append(batch_arr_fr)
                batch_mar_fr_list.append(batch_mar_fr)

                # Append outputs of the residual encoding.
                if hparams.residual_encoder:
                    list_residual_encoding.append(residual_encoding)
                    list_mu.append(mu)
                    list_logvar.append(logvar)

                # Accumulate outputs of the speaker adversarial training module.
                if hparams.speaker_adversarial_training and val_type == ('all', 'all'):
                    np_spk_cm = tensor_spk_cm.cpu().numpy()
                    np_spk_cm_sum += np_spk_cm

                # Accumulate outputs of the emotion adversarial training module.
                if hparams.emotion_adversarial_training and val_type == ('all', 'all'):
                    np_emo_cm = tensor_emo_cm.cpu().numpy()
                    np_emo_cm_sum += np_emo_cm

                if rank == 0:
                    # Wrap up data of audios to be logged.
                    inputs = (input_lengths, text_padded, speakers, emotion_input_vectors, text_raw)
                    outputs = (output_lengths, gate_outputs_fr, end_points_fr, mel_padded, mel_outputs_postnet, mel_outputs_postnet_fr, alignments, alignments_fr, prosody, prosody_fr)
                    batch_attention_measures_tf = (batch_attention_quality, batch_ar, batch_letter_ar, batch_punct_ar, batch_blank_ar, batch_arr, batch_mar)
                    batch_attention_measures_fr = (batch_attention_quality_fr, batch_ar_fr, batch_letter_ar_fr, batch_punct_ar_fr, batch_blank_ar_fr, batch_arr_fr, batch_mar_fr)
                    temp_prosody_hiddens = temp_prosody_hiddens_tf, temp_prosody_hiddens_fr

                    sum_output_lengths += output_lengths.sum().cpu().item()

                    # [SynthDict 1] A random sample.
                    if i == i_batch_rand:
                        i_rand = i_sample_rand
                        # (i, i_rand) == (0, 0) is a random sample
                        # b/c the validation data reshuffled at every epoch
                        synth_dict_rand = dict()
                        fill_synth_dict(hparams, synth_dict_rand, i_rand, inputs, outputs,
                                batch_attention_measures_tf, batch_attention_measures_fr,
                                temp_prosody_hiddens)

                    # [SynthDict 2] A teacher-forcing sample that has the minimum attention quality.
                    if min_attention_quality_tf > batch_attention_quality.min().item():
                        min_attention_quality_tf = batch_attention_quality.min().item()
                        i_min = batch_attention_quality.argmin().item()
                        fill_synth_dict(hparams, synth_dict_min_aq_tf, i_min, inputs, outputs,
                                batch_attention_measures_tf, batch_attention_measures_fr,
                                temp_prosody_hiddens)

                    # [SynthDict 3] A free-running sample that has the minimum attention quality.
                    if min_attention_quality_fr > batch_attention_quality_fr.min().item():
                        min_attention_quality_fr = batch_attention_quality_fr.min().item()
                        i_min = batch_attention_quality_fr.argmin().item()
                        fill_synth_dict(hparams, synth_dict_min_aq_fr, i_min, inputs, outputs,
                                batch_attention_measures_tf, batch_attention_measures_fr,
                                temp_prosody_hiddens)

                    if hparams.prosody_predictor:
                        np_sum_prosody_pred_dims += sum_prosody_pred_dims.detach().cpu().numpy()

                    if hparams.reference_encoder:
                        np_sum_prosody_ref_dims += sum_prosody_ref_dims.detach().cpu().numpy()

                ############################################################

            if rank == 0:
                ############################################################
                # TEACHER FORCING #####
                n_val_batches = (i + 1)
                val_loss_mel = val_loss_mel / n_val_batches
                val_loss_gate = val_loss_gate / n_val_batches
                val_loss_KLD = val_loss_KLD / n_val_batches
                val_loss_ref_enc = val_loss_ref_enc / n_val_batches
                val_loss_spk_adv = val_loss_spk_adv / n_val_batches
                val_loss_emo_adv = val_loss_emo_adv / n_val_batches
                val_loss_att_means = val_loss_att_means / n_val_batches
                val_loss = val_loss / n_val_batches

                if hparams.residual_encoder:
                    residual_encoding = torch.cat(list_residual_encoding)
                    mu = torch.cat(list_mu)
                    logvar = torch.cat(list_logvar)

                if hparams.prosody_predictor:
                    mean_prosody_pred_dim = np_sum_prosody_pred_dims / sum_output_lengths

                if hparams.reference_encoder:
                    mean_prosody_ref_dim = np_sum_prosody_ref_dims / sum_output_lengths

                # [M1] forward_attention_ratio
                val_batch_far = torch.cat(batch_far_list)
                val_mean_far = val_batch_far.mean().item()
                far_pair = (val_mean_far, val_batch_far)
                # [M2] attention_ratio
                val_batch_ar = torch.cat(batch_ar_list)
                val_mean_ar = val_batch_ar.mean().item()
                val_batch_letter_ar = torch.cat(batch_letter_ar_list)
                val_mean_letter_ar = val_batch_letter_ar.mean().item()
                val_batch_punct_ar = torch.cat(batch_punct_ar_list)
                val_mean_punct_ar = val_batch_punct_ar.mean().item()
                val_batch_blank_ar = torch.cat(batch_blank_ar_list)
                val_mean_blank_ar = val_batch_blank_ar.mean().item()

                # [M3] attention_range_ratio
                val_batch_arr = torch.cat(batch_arr_list)
                val_mean_arr = val_batch_arr.mean().item()
                arr_pair = (val_mean_arr, val_batch_arr)
                # [M4] multiple_attention_ratio
                val_batch_mar = torch.cat(batch_mar_list)
                val_mean_mar = val_batch_mar.mean().item()
                mar_pair = (val_mean_mar, val_batch_mar)

                ############################################################
                # FREE RUNNING #####
                # [M1] forward_attention_ratio
                val_batch_far_fr = torch.cat(batch_far_fr_list)
                val_mean_far_fr = val_batch_far_fr.mean().item()
                far_fr_pair = (val_mean_far_fr, val_batch_far_fr)
                # [M2] attention_ratio
                val_batch_ar_fr = torch.cat(batch_ar_fr_list)
                val_mean_ar_fr = val_batch_ar_fr.mean().item()
                val_batch_letter_ar_fr = torch.cat(batch_letter_ar_fr_list)
                val_mean_letter_ar_fr = val_batch_letter_ar_fr.mean().item()
                val_batch_punct_ar_fr = torch.cat(batch_punct_ar_fr_list)
                val_mean_punct_ar_fr = val_batch_punct_ar_fr.mean().item()
                val_batch_blank_ar_fr = torch.cat(batch_blank_ar_fr_list)
                val_mean_blank_ar_fr = val_batch_blank_ar_fr.mean().item()

                # [M3] attention_range_ratio
                val_batch_arr_fr = torch.cat(batch_arr_fr_list)
                val_mean_arr_fr = val_batch_arr_fr.mean().item()
                arr_fr_pair = (val_mean_arr_fr, val_batch_arr_fr)
                # [M4] multiple_attention_ratio
                val_batch_mar_fr = torch.cat(batch_mar_fr_list)
                val_mean_mar_fr = val_batch_mar_fr.mean().item()
                mar_fr_pair = (val_mean_mar_fr, val_batch_mar_fr)


        if rank == 0:
            print("Validation loss {} {}: {:9f}  ".format(str(val_type), iteration, val_loss))
            losses = (val_loss, val_loss_mel, val_loss_gate, val_loss_KLD, val_loss_ref_enc, val_loss_spk_adv, val_loss_emo_adv, val_loss_att_means)
            ar_pairs = ((val_mean_ar, val_batch_ar),
                        (val_mean_letter_ar, val_batch_letter_ar),
                        (val_mean_punct_ar, val_batch_punct_ar),
                        (val_mean_blank_ar, val_batch_blank_ar))
            ar_fr_pairs = ((val_mean_ar_fr, val_batch_ar_fr),
                           (val_mean_letter_ar_fr, val_batch_letter_ar_fr),
                           (val_mean_punct_ar_fr, val_batch_punct_ar_fr),
                           (val_mean_blank_ar_fr, val_batch_blank_ar_fr))
            attention_measures = far_pair, ar_pairs, arr_pair, mar_pair
            attention_measures_fr = far_fr_pair, ar_fr_pairs, arr_fr_pair, mar_fr_pair


            # ToDeleete
            ###########
            dict_log_values = {
                'iteration':iteration,
                'epoch':epoch,
                'model':model,
                'x':x,
                'etc':etc,
                'y':y,
                'y_pred':y_pred,
                'int_pred_speakers':int_pred_speakers,
                'losses':losses,
                'attention_measures':attention_measures,
                'attention_measures_fr':attention_measures_fr,
                'gate_accuracy':gate_accuracy,
                'gate_mae':gate_mae,
                'end_point_mae':end_point_mae,
                'synth_dict_rand':synth_dict_rand,
                'synth_dict_min_aq_tf':synth_dict_min_aq_tf,
                'synth_dict_min_aq_fr':synth_dict_min_aq_fr,
            }
            if hparams.residual_encoder:
                dict_log_values['residual_encoding'] = residual_encoding
                dict_log_values['mu'] = mu
                dict_log_values['logvar'] = logvar
            if hparams.prosody_predictor:
                dict_log_values['mean_prosody_pred_dim'] = mean_prosody_pred_dim
            if hparams.reference_encoder:
                dict_log_values['mean_prosody_ref_dim'] = mean_prosody_ref_dim
            if hparams.speaker_adversarial_training:
                if val_type == ('all', 'all'):
                    # Compute the classification measures of the speaker adversarial classifier.
                    dict_log_values['speaker_clsf_report'] = get_clsf_report(np_spk_cm_sum, hparams.speakers, trainset.speaker_list)
            if hparams.emotion_adversarial_training:
                if val_type == ('all', 'all'):
                    # Compute the classification measures of the emotion adversarial classifier.
                    dict_log_values['emotion_clsf_report'] = get_clsf_report(np_emo_cm_sum, hparams.emotions, trainset.emotion_list)

            logger.log_validation(trainset, valset, val_type, hparams, dict_log_values)
    model.train()


def train(output_directory, log_directory, checkpoint_path, pretrained_path,
          warm_start, n_gpus,
          rank, group_name, hparams, run_name, prj_name, resume, first_device):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name, first_device)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 betas=hparams.adam_batas,
                                 eps=hparams.adam_eps,
                                 weight_decay=hparams.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        0.5**(1 / (12500 * (64 / (hparams.batch_size * n_gpus)))),
        -1
    )


    if hparams.fp16_run:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=hparams.fp16_opt_level) # default: opt_level='O2'
    '''if hparams.fp16_run:
        # Creates a GradScaler once at the beginning of training.
        amp_scaler = GradScaler()
    else:
        amp_scaler = None'''

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = TotalLoss(hparams)

    logger = prepare_directories_and_logger(
        hparams,
        output_directory, log_directory, rank, run_name, prj_name, resume, model)

    train_loader, trainset, valsets, collate_fn = prepare_dataloaders(hparams)
    with open(os.path.join(output_directory, prj_name, run_name, 'model.txt'), 'w') as f:
        f.write(str(model))
    with open(os.path.join(output_directory, prj_name, run_name, 'hparams.txt'), 'w') as f:
        f.write(str(hparams.to_json(indent=4)))

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, learning_rate, iteration, float_epoch, cp_dict = load_checkpoint(
                checkpoint_path, model, optimizer, lr_scheduler, logger, rank,
                hparams)
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(float_epoch))

    if pretrained_path is not None:
        model = load_pretrained_model(model, pretrained_path,
            freeze_pretrained=hparams.freeze_pretrained,
            except_for=hparams.freeze_except_for)[0]

    model.train()
    is_overflow = False
    is_first_iteration = True
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = hparams.learning_rate
        param_group['lr'] = learning_rate

    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Start Epoch {}:".format(epoch+1))
        if hparams.speaker_adversarial_training:
            np_spk_cm_sum = np.zeros((len(trainset.speaker_list), len(trainset.speaker_list)))
        if hparams.emotion_adversarial_training:
            np_emo_cm_sum = np.zeros((len(trainset.emotion_list), len(trainset.emotion_list)))

        for i, batch in enumerate(train_loader):
            batches_per_epoch = len(train_loader)
            float_epoch = epoch + i / len(train_loader)
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_scheduler.get_lr()[0]

            # Clears the gradients of all optimized torch.Tensor s.
            model.zero_grad()

            # Parse the current batch.
            x, y, etc = model.parse_batch(batch)
            text_padded, input_lengths, mel_padded, max_len, output_lengths = x
            mel_padded, gate_padded = y
            speakers, sex, emotion_input_vectors, emotion_targets, lang, \
                text_raw, wav_paths = etc

            # Forward propagtion
            start_foward = time.perf_counter()
            (y_pred, y_pred_speakers, y_pred_emotions,
                y_pred_res_en, att_means
            ) = model(x, speakers, emotion_input_vectors)
            '''with autocast(enabled=hparams.fp16_run):
                (y_pred, y_pred_speakers, y_pred_emotions,
                    y_pred_res_en, att_means
                ) = model(x, speakers, emotion_input_vectors)'''
            forward_duration = time.perf_counter() - start_foward
            temp_prosody = model.get_temp_prosody()
            temp_prosody_hiddens = model.get_temp_prosody_hiddens()

            # Forward propagtion results
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments, \
                prosody_ref, prosody_pred, end_points = y_pred
            logit_speakers, prob_speakers, int_pred_speakers = y_pred_speakers
            logit_emotions, prob_emotions, int_pred_emotions = y_pred_emotions
            residual_encoding, mu, logvar = y_pred_res_en
            prosody = prosody_ref, prosody_pred

            start_loss = time.perf_counter()
            (loss, loss_taco2, loss_mel, loss_gate, loss_KLD, loss_ref_enc,
                loss_spk_adv, loss_emo_adv, loss_att_means
                ) = criterion(
                    mel_outputs, mel_outputs_postnet, mel_padded,
                    gate_outputs, gate_padded,
                    y_pred, y,
                    mu, logvar,
                    prosody_pred, prosody_ref,
                    logit_speakers, speakers,
                    logit_emotions, emotion_targets,
                    att_means, input_lengths,
                    iteration
                )
            '''with autocast(enabled=hparams.fp16_run):
                (loss, loss_taco2, loss_mel, loss_gate, loss_KLD, loss_ref_enc,
                    loss_spk_adv, loss_emo_adv, loss_att_means
                    ) = criterion(
                        mel_outputs, mel_outputs_postnet, mel_padded,
                        gate_outputs, gate_padded,
                        y_pred, y,
                        mu, logvar,
                        prosody_pred, prosody_ref,
                        logit_speakers, speakers,
                        logit_emotions, emotion_targets,
                        att_means, input_lengths,
                        iteration
                    )'''
            loss_duration = time.perf_counter() - start_loss

            start_backward = time.perf_counter()
            # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True.
            # These are accumulated into x.grad for every parameter x
            if hparams.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                '''amp_scaler.scale(loss).backward()'''
            else:
                loss.backward()

            if hparams.fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), hparams.grad_clip_thresh)
                '''# Unscales the gradients of optimizer's assigned params in-place
                amp_scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)'''
                is_overflow = math.isnan(grad_norm)
                if is_overflow:
                    pass
                else:
                    clip_coef = hparams.grad_clip_thresh / (grad_norm + 1e-6)
                    if clip_coef < 1:
                        clipped_grad_norm = grad_norm * clip_coef
                    else:
                        clipped_grad_norm = grad_norm
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)
                clip_coef = hparams.grad_clip_thresh / (grad_norm + 1e-6)
                if clip_coef < 1:
                    clipped_grad_norm = grad_norm * clip_coef
                else:
                    clipped_grad_norm = grad_norm

            learning_rate = lr_scheduler.get_lr()[0]
            # optimizer.step is performs a parameter update based on the current
            # gradient (stored in .grad attribute of a parameter) and the update rule.
            w_steps_abs_mean, adam_steps_abs_mean, adam_step_numers_abs_mean, \
                adam_step_denoms_abs_mean, grads_abs_mean = adam_step(optimizer)
            optimizer.step()
            '''if hparams.fp16_run:
                amp_scaler.step(optimizer)
                # Updates the scale for next iteration.
                amp_scaler.update()
            else:
                optimizer.step()'''
            backward_duration = time.perf_counter() - start_backward

            # Because of the following recommendation
            # https://github.com/NVIDIA/apex/issues/480#issuecomment-587154020
            if (checkpoint_path is not None) and hparams.fp16_run and \
                    (hparams.fp16_opt_level == 'O2') and is_first_iteration:
                optimizer.load_state_dict(cp_dict['optimizer'])

            # Compute stop gate performance.
            np_output_lengths = output_lengths.cpu().numpy()
            mel_lengths = get_mel_lengths(gate_outputs)
            np_mel_lengths = mel_lengths.cpu().numpy()
            # Compute stop gate accuracy
            gate_accuracy = accuracy_score(np_output_lengths, np_mel_lengths)
            # Compute stop gate MAE(pred_lengths, true_lengths)
            gate_mae = mean_absolute_error(np_output_lengths, np_mel_lengths)

            # Compute forward_attention_ratio.
            mean_far, batch_far = forward_attention_ratio(alignments, input_lengths, output_lengths=output_lengths, mode_mel_length="ground_truth")
            ar_pairs = attention_ratio(alignments, input_lengths, text_padded, output_lengths=output_lengths, mode_mel_length="ground_truth")
            mean_ar, batch_ar = ar_pairs[0]
            mean_letter_ar, batch_letter_ar = ar_pairs[1]
            best_letter_ar = batch_letter_ar.max().item()
            worst_letter_ar = batch_letter_ar.min().item()
            mean_punct_ar, batch_punct_ar = ar_pairs[2]
            mean_blank_ar, batch_blank_ar = ar_pairs[3]

            mean_arr, batch_arr = attention_range_ratio(alignments, input_lengths, output_lengths=output_lengths, mode_mel_length="ground_truth")
            mean_mar, batch_mar = multiple_attention_ratio(alignments, input_lengths, text_padded, output_lengths=output_lengths, mode_mel_length="ground_truth")
            mean_attention_quality = get_attention_quality(mean_far, mean_mar, mean_letter_ar)
            batch_attention_quality = get_attention_quality(batch_far, batch_mar, batch_letter_ar)
            best_attention_quality = batch_attention_quality.max().item()
            worst_attention_quality = batch_attention_quality.min().item()

            if hparams.speaker_adversarial_training:
                np_target_speakers = spk_adv_targets.cpu().numpy()
                np_pred_speakers = int_pred_speakers.cpu().numpy()
                np_speaker_array = np.arange(len(trainset.speaker_list))
                np_spk_cm = confusion_matrix(
                    np.append(np_target_speakers, np_speaker_array),
                    np.append(np_pred_speakers, np_speaker_array),
                    labels=list(np_speaker_array))
                np_spk_cm -= np.eye(np_speaker_array.shape[0], dtype=np_spk_cm.dtype)
                tensor_spk_cm = torch.IntTensor(np_spk_cm).cuda()

            if hparams.emotion_adversarial_training:
                np_target_emotions = emo_adv_targets.cpu().numpy()
                np_pred_emotions = int_pred_emotions.cpu().numpy()
                np_emotion_array = np.arange(len(trainset.emotion_list))
                np_emo_cm = confusion_matrix(
                    np.append(np_target_emotions, np_emotion_array),
                    np.append(np_pred_emotions, np_emotion_array),
                    labels=list(np_emotion_array))
                np_emo_cm -= np.eye(np_emotion_array.shape[0], dtype=np_emo_cm.dtype)
                tensor_emo_cm = torch.IntTensor(np_emo_cm).cuda()

            sum_output_lengths = np_output_lengths.sum()

            if hparams.reference_encoder:
                sum_prosody_ref_dims = prosody_ref.sum(dim=(0,1))

            if hparams.prosody_predictor:
                sum_prosody_pred_dims = prosody_pred.sum(dim=(0,1))

            if hparams.distributed_run:
                reduced_loss_mel = reduce_tensor(loss_mel).item()
                reduced_loss_gate = reduce_tensor(loss_gate).item()
                reduced_loss_KLD = reduce_tensor(loss_KLD).item()
                reduced_loss_ref_enc = reduce_tensor(loss_ref_enc).item()
                reduced_loss_spk_adv = reduce_tensor(loss_spk_adv).item()
                reduced_loss_emo_adv = reduce_tensor(loss_emo_adv).item()
                reduced_loss_att_means = reduce_tensor(loss_att_means).item()
                reduced_loss = reduce_tensor(loss).item()
                gate_accuracy = reduce_scalar(gate_accuracy)
                gate_mae = reduce_scalar(gate_mae)
                mean_far = reduce_scalar(mean_far)
                mean_ar = reduce_scalar(mean_ar)
                mean_letter_ar = reduce_scalar(mean_letter_ar)
                best_letter_ar = reduce_scalar(best_letter_ar, 'max')
                worst_letter_ar = reduce_scalar(worst_letter_ar, 'min')
                mean_punct_ar = reduce_scalar(mean_punct_ar)
                mean_blank_ar = reduce_scalar(mean_blank_ar)
                mean_arr = reduce_scalar(mean_arr)
                mean_mar = reduce_scalar(mean_mar)
                mean_attention_quality = reduce_scalar(mean_attention_quality)
                batch_far = gather_all_tensor(batch_far)
                batch_ar = gather_all_tensor(batch_ar)
                batch_letter_ar = gather_all_tensor(batch_letter_ar)
                batch_punct_ar = gather_all_tensor(batch_punct_ar)
                batch_blank_ar = gather_all_tensor(batch_blank_ar)
                batch_arr = gather_all_tensor(batch_arr)
                batch_mar = gather_all_tensor(batch_mar)
                batch_attention_quality = gather_all_tensor(batch_attention_quality)
                best_attention_quality = reduce_scalar(best_attention_quality, 'max')
                worst_attention_quality = reduce_scalar(worst_attention_quality, 'min')
                sum_output_lengths = reduce_scalar(sum_output_lengths, 'sum')
                if hparams.residual_encoder:
                    residual_encoding = gather_all_tensor(residual_encoding)
                    mu = gather_all_tensor(mu)
                    logvar = gather_all_tensor(logvar)
                if hparams.prosody_predictor:
                    sum_prosody_pred_dims = reduce_tensor(sum_prosody_pred_dims, 'sum')
                if hparams.reference_encoder:
                    sum_prosody_ref_dims = reduce_tensor(sum_prosody_ref_dims, 'sum')
                if hparams.speaker_adversarial_training:
                    tensor_spk_cm = reduce_tensor(tensor_spk_cm, 'sum')
                if hparams.emotion_adversarial_training:
                    tensor_emo_cm = reduce_tensor(tensor_emo_cm, 'sum')
            else:
                reduced_loss_mel = loss_mel.item()
                reduced_loss_gate = loss_gate.item()
                reduced_loss_KLD = loss_KLD.item()
                reduced_loss_ref_enc = loss_ref_enc.item()
                reduced_loss_spk_adv = loss_spk_adv.item()
                reduced_loss_emo_adv = loss_emo_adv.item()
                reduced_loss_att_means = loss_att_means.item()
                reduced_loss = loss.item()

            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                n_enc_steps = max_len
                n_dec_steps = np_output_lengths.sum()
                print("Epoch {} Float Epoch {:4f} Iteration {} Learning rate {} Train total loss {:.6f} Mel loss {:.6f} Gate loss {:.6f} KLD loss {:.6f} RefEn loss {:.6f} SpkAdv loss {:.6f} EmoAdv loss {:.6f} MonoAtt MSE loss {:.6f} Grad Norm {:.6f} Clipped Grad Norm {:.6f} EncSteps {} DecSteps {} EncDecSteps {} {:.1f}EncDecSteps/s ForwardDuration {:.2f}s LossDuration {:.2f}s BackwardDuration {:.2f}s IterDuration {:.2f}s/it ".format(
                    epoch, float_epoch, iteration, learning_rate, reduced_loss, reduced_loss_mel, reduced_loss_gate, reduced_loss_KLD, reduced_loss_ref_enc, reduced_loss_spk_adv, reduced_loss_emo_adv, reduced_loss_att_means, grad_norm, clipped_grad_norm, n_enc_steps, n_dec_steps, (n_enc_steps + n_dec_steps), ((n_enc_steps + n_dec_steps) / duration), forward_duration, loss_duration, backward_duration, duration))
                reduced_losses = (reduced_loss, reduced_loss_mel, reduced_loss_gate, reduced_loss_KLD, reduced_loss_ref_enc, reduced_loss_spk_adv, reduced_loss_emo_adv, reduced_loss_att_means)
                att_measures = (
                    (mean_far, batch_far),
                    (mean_ar, batch_ar),
                    (mean_letter_ar, batch_letter_ar, best_letter_ar, worst_letter_ar),
                    (mean_punct_ar, batch_punct_ar),
                    (mean_blank_ar, batch_blank_ar),
                    (mean_arr, batch_arr),
                    (mean_mar, batch_mar),
                    (mean_attention_quality, batch_attention_quality, best_attention_quality, worst_attention_quality)
                )
                # Add logging objects into dict_log_values. ====================
                dict_log_values = {
                    'iteration':iteration,
                    'epoch':float_epoch,
                    'losses':reduced_losses,
                    'grad_norm':grad_norm,
                    'clipped_grad_norm':clipped_grad_norm,
                    'learning_rate':learning_rate,
                    'KLD_weight':hparams.res_en_KLD_weight,
                    'duration':duration,
                    'n_encoder_steps':n_enc_steps,
                    'n_decoder_steps':n_dec_steps,
                    'x':x,
                    'etc':etc,
                    'y_pred':y_pred,
                    'int_pred_speakers':int_pred_speakers,
                    'gate_accuracy':gate_accuracy,
                    'gate_mae':gate_mae,
                    'att_measures':att_measures,
                    'w_steps_abs_mean':w_steps_abs_mean,
                    'adam_steps_abs_mean':adam_steps_abs_mean,
                    'adam_step_numers_abs_mean':adam_step_numers_abs_mean,
                    'adam_step_denoms_abs_mean':adam_step_denoms_abs_mean,
                    'grads_abs_mean':grads_abs_mean,
                }

                if hparams.residual_encoder:
                    dict_log_values['residual_encoding'] = residual_encoding
                    dict_log_values['mu'] = mu
                    dict_log_values['logvar'] = logvar

                if hparams.prosody_predictor:
                    np_sum_prosody_pred_dims = sum_prosody_pred_dims.detach().cpu().numpy()
                    mean_prosody_pred_dim = np_sum_prosody_pred_dims / sum_output_lengths
                    dict_log_values['mean_prosody_pred_dim'] = mean_prosody_pred_dim

                if hparams.reference_encoder:
                    np_sum_prosody_ref_dims = sum_prosody_ref_dims.detach().cpu().numpy()
                    mean_prosody_ref_dim = np_sum_prosody_ref_dims / sum_output_lengths
                    dict_log_values['mean_prosody_ref_dim'] = mean_prosody_ref_dim

                if hparams.speaker_adversarial_training:
                    # Compute the accuracy of the speaker adversarial classifier.
                    np_spk_cm = tensor_spk_cm.cpu().numpy()
                    speaker_clsf_report = get_clsf_report(np_spk_cm, hparams.speakers, trainset.speaker_list)
                    spk_adv_accuracy = speaker_clsf_report['accuracy']

                    # Save objects to log into dict_log_values.
                    dict_log_values['spk_adv_accuracy'] = spk_adv_accuracy
                    dict_log_values['speaker_clsf_report'] = speaker_clsf_report

                    # Logging for the performance of train_epoch.
                    np_spk_cm_sum += np_spk_cm
                    if (i+1 == batches_per_epoch):
                        # If it is the last batch, ...
                        speaker_clsf_report_train_epoch = get_clsf_report(np_spk_cm_sum, hparams.speakers, trainset.speaker_list)
                        dict_log_values['speaker_clsf_report_train_epoch'] = speaker_clsf_report_train_epoch

                if hparams.emotion_adversarial_training:
                    # Compute the accuracy of the emotion adversarial classifier.
                    np_emo_cm = tensor_emo_cm.cpu().numpy()
                    emotion_clsf_report = get_clsf_report(np_emo_cm, hparams.emotions, trainset.emotion_list)
                    emo_adv_accuracy = emotion_clsf_report['accuracy']

                    # Save objects to log into dict_log_values.
                    dict_log_values['emo_adv_accuracy'] = emo_adv_accuracy
                    dict_log_values['emotion_clsf_report'] = emotion_clsf_report

                    # Logging for the performance of train_epoch.
                    np_emo_cm_sum += np_emo_cm
                    if (i+1 == batches_per_epoch):
                        # If it is the last batch, ...
                        emotion_clsf_report_train_epoch = get_clsf_report(np_emo_cm_sum, hparams.emotions, trainset.emotion_list)
                        dict_log_values['emotion_clsf_report_train_epoch'] = emotion_clsf_report_train_epoch

                # Pass training logging objects to logger.
                logger.log_training(trainset, hparams, dict_log_values, batches_per_epoch)

            if hparams.log_validation and (not is_overflow and ((iteration % hparams.iters_per_checkpoint == 0) or (i+1 == batches_per_epoch) or is_first_iteration)):
                if rank == 0 and hparams.reference_encoder != '':
                    logger.log_training_prosody(hparams, iteration,
                        input_lengths, text_padded, text_raw,
                        output_lengths, mel_outputs_postnet,
                        alignments,
                        temp_prosody, temp_prosody_hiddens)
                validate(model, criterion, trainset, valsets, iteration, float_epoch,
                         hparams.batch_size, n_gpus, collate_fn, logger,
                         False, rank, hparams)
                         #hparams.distributed_run, rank, hparams)
                if rank == 0 and (iteration % hparams.iters_per_checkpoint == 0):
                    checkpoint_path = os.path.join(
                        os.path.join(output_directory, prj_name, run_name), "checkpoint_{}-epoch_{:.4}".format(iteration, float_epoch))
                    save_checkpoint(hparams, model, optimizer, learning_rate, iteration, float_epoch,
                                    lr_scheduler, logger, checkpoint_path)
                if rank == 0 and (i+1 == batches_per_epoch):
                    checkpoint_path = os.path.join(
                        os.path.join(output_directory, prj_name, run_name), "checkpoint_{}-epoch_{:.4}_end-epoch_{}".format(iteration, float_epoch, epoch+1))
                    save_checkpoint(hparams, model, optimizer, learning_rate, iteration, epoch+1,
                                    lr_scheduler, logger, checkpoint_path)

            tmp_iteration = iteration
            tmp_learning_rate = learning_rate

            def signal_handler(sig, frame):
                print('You pressed Ctrl+C!')
                if rank == 0:
                    checkpoint_path = os.path.join(
                        os.path.join(output_directory, prj_name, run_name), "checkpoint_{}-epoch_{:.4}".format(iteration, float_epoch))
                    save_checkpoint(hparams, model, optimizer, tmp_learning_rate, tmp_iteration, float_epoch,
                                    lr_scheduler, logger, checkpoint_path)
                sys.exit(0)

            signal.signal(signal.SIGINT, signal_handler)

            lr_scheduling_start_iter = round(hparams.lr_scheduling_start_iter * (64 / (hparams.batch_size * n_gpus)))
            if iteration > lr_scheduling_start_iter \
                    and learning_rate >= hparams.lr_min \
                    and hparams.lr_scheduling:
                lr_scheduler.step()

            iteration += 1
            is_first_iteration = False

        # End of the current epoch
        # Upsampling again
        if hparams.resampling_trainset_at_each_epoch:
            trainset.upsampling(epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        default='outdir',
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        default='logdir',
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        required=False, help='pretrained model path')
    parser.add_argument('-r', '--resume', type=str, default="",
                        required=False, help='whether to resume logging')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--first_device', type=int, default=0,
                        required=False, help='GPU device number for rank 0 process')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--visible_gpus', type=str, default="0",
                        required=False, help='CUDA visible GPUs')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    parser.add_argument('--run_name', type=str,
                        help='give a distinct name for this running')
    parser.add_argument('--prj_name', type=str, default='tts-tacotron2',
                        help='give a project name for this running')


    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled

    # Following lines to control randomness
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
    torch.backends.cudnn.deterministic = hparams.cudnn_deterministic
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    torch.cuda.manual_seed_all(hparams.seed) # if use multi-GPU
    np.random.seed(hparams.seed)
    random.seed(hparams.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    if args.rank == 0:
        if args.resume == "":
            wandb.init(name=args.run_name, project=args.prj_name, resume=args.resume)
        else:
            wandb.init(project=args.prj_name, resume=args.resume)

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)
    print("cuDNN Deterministic:", hparams.cudnn_deterministic)
    print("Random Seed:", hparams.seed)
    print("Visible GPU IDs:", args.visible_gpus)

    if hparams.compute_alignments:
        if args.pretrained_path is not None:
            compute_alignments(args.pretrained_path, hparams)
        else:
            print("pretrained_path must be specified!")
        exit()

    # If checkpoint_path is given as a number, then the number is considered
    # as an iteration number. Thus, convert that number to the checkpoint path.
    if args.checkpoint_path is not None:
        if args.checkpoint_path.isnumeric():
            args.checkpoint_path = get_checkpoint_iter2path(
                args.output_directory, args.prj_name, args.run_name,
                args.checkpoint_path)
            print("Import the checkpoint from {}".format(args.checkpoint_path))

    train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.pretrained_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams,
          args.run_name, args.prj_name, args.resume, args.first_device)
