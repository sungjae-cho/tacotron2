import os
import time
import argparse
import math
import signal
import sys
import random
from numpy import finfo
from apex import amp

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, mean_absolute_error
from torch.nn import MSELoss

from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss, forward_attention_loss
from logger import Tacotron2Logger
from hparams import create_hparams
from measures import forward_attention_ratio, attention_ratio, attention_range_ratio, multiple_attention_ratio
from measures import get_mel_length, get_mel_lengths
from measures import get_attention_quality
from utils import get_spk_adv_targets, load_pretrained_model

def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

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
    for speaker in all_valset.speaker_list:
        for emotion in all_valset.emotion_list:
            valset = TextMelLoader(hparams, 'val', speaker, emotion)
            if len(valset) != 0:
                valsets[(speaker, emotion)] = valset

    collate_fn = TextMelCollate(hparams.n_frames_per_step)

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
                                   run_name, prj_name, resume):
    if rank == 0:
        if not os.path.isdir(os.path.join(output_directory, prj_name, run_name)):
            os.makedirs(os.path.join(output_directory, prj_name, run_name))
            os.chmod(os.path.join(output_directory, prj_name, run_name), 0o775)
        logger = Tacotron2Logger(hparams, run_name, prj_name,
            os.path.join(log_directory, prj_name, run_name), resume)
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


def load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler, logger):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    print("Loaded learning_rate=", learning_rate)
    if 'lr_scheduler' in checkpoint_dict.keys():
        lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
    else:
        lr_scheduler.load_state_dict({'base_lrs':[learning_rate]})
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    dict_vars = checkpoint_dict['training_epoch_variables']
    logger.set_training_epoch_variables(dict_vars)
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, lr_scheduler,
        logger, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate,
                'lr_scheduler': lr_scheduler.state_dict(),
                'training_epoch_variables': logger.get_training_epoch_variables()
                }, filepath)


def fill_synth_dict(synth_dict, idx, inputs, outputs,
        batch_attention_measures_tf, batch_attention_measures_fr):
    # Inputs
    (input_lengths, text_padded, speakers, emotion_vectors) = inputs
    text_length = input_lengths[idx].item()
    synth_dict['text_sequence'] = text_padded[idx,:text_length]
    synth_dict['speaker_tensor'] = speakers[idx]
    synth_dict['emotion_tensor'] = emotion_vectors[idx]

    # Outputs
    (output_lengths, gate_outputs_fr,
        mel_padded, mel_outputs_postnet, mel_outputs_postnet_fr,
        alignments, alignments_fr) = outputs
    mel_length = output_lengths[idx].item()
    mel_length_fr = get_mel_length(gate_outputs_fr[idx])
    synth_dict['mel_true'] = mel_padded[idx,:,:mel_length]
    synth_dict['mel_output_tf'] = mel_outputs_postnet[idx,:,:mel_length]
    synth_dict['mel_output_fr'] = mel_outputs_postnet_fr[idx,:,:mel_length_fr]
    synth_dict['alignment_tf'] = alignments[idx,:mel_length,:text_length]
    synth_dict['alignment_fr'] = alignments_fr[idx,:mel_length_fr,:text_length]

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


def validate(model, criterions, valsets, iteration, epoch, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank, hparams):
    """Handles all the validation scoring and printing"""
    for val_type, valset in valsets.items():
        #val_type: tuple. (str_speaker, str_emotion).
        model.eval()
        with torch.no_grad():
            val_sampler = DistributedSampler(valset) if distributed_run else None
            val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                    shuffle=False, batch_size=batch_size,
                                    pin_memory=False, collate_fn=collate_fn)

            criterion, criterion_dom = criterions
            val_loss_mel = 0.0
            val_loss_gate = 0.0
            val_loss_spk_adv = 0.0
            val_loss_att_means = 0.0
            val_loss = 0.0

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
                speakers, sex, emotion_vectors, lang = etc

                ############################################################
                # TEACHER FORCING #####
                # Forward propagation by teacher forcing
                y_pred, y_pred_speakers, att_means = model(x, speakers, emotion_vectors)

                # Forward propagtion results
                mel_outputs, mel_outputs_postnet, gate_outputs, alignments = y_pred
                spk_logit_outputs, prob_speakers, pred_speakers = y_pred_speakers

                # Compute stop gate accuracy
                np_output_lengths = output_lengths.cpu().numpy()
                mel_lengths = get_mel_lengths(gate_outputs)
                np_mel_lengths = mel_lengths.cpu().numpy()
                gate_accuracy = accuracy_score(np_output_lengths, np_mel_lengths)
                # Compute stop gate MAE(pred_lengths, true_lengths)
                gate_mae = mean_absolute_error(np_output_lengths, np_mel_lengths)

                # Caculate Tacotron2 losses.
                loss_taco2, loss_mel, loss_gate = criterion(y_pred, y)

                # Forward the speaker adversarial module.
                if hparams.speaker_adversarial_training:
                    spk_adv_targets = get_spk_adv_targets(speakers, input_lengths)
                    # Compute the speaker adversarial loss
                    loss_spk_adv = criterion_dom(spk_logit_outputs, spk_adv_targets)
                    # Compute the accuracy of the adversarial speaker classifier.
                    np_speakers = spk_adv_targets.cpu().numpy()
                    np_pred_speakers = pred_speakers.cpu().numpy()
                    spk_adv_accuracy = accuracy_score(np_speakers, np_pred_speakers)
                else:
                    loss_spk_adv = torch.zeros(1).cuda()

                if hparams.monotonic_attention:
                    input_lengths = x[1]
                    loss_att_means = MSELoss()(att_means, input_lengths.float())
                else:
                    loss_att_means = torch.zeros(1).cuda()

                loss = loss_taco2 + hparams.speaker_adv_weight * loss_spk_adv + \
                    hparams.loss_att_means_weight * loss_att_means

                if distributed_run:
                    reduced_val_loss_mel = reduce_tensor(loss_mel.data, n_gpus).item()
                    reduced_val_loss_gate = reduce_tensor(loss_gate.data, n_gpus).item()
                    reduced_val_loss_spk_adv = reduce_tensor(loss_spk_adv.data, n_gpus).item()
                    reduced_val_loss_att_means = reduce_tensor(loss_att_means.data, n_gpus).item()
                    reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
                else:
                    reduced_val_loss_mel = loss_mel.item()
                    reduced_val_loss_gate = loss_gate.item()
                    reduced_val_loss_spk_adv = loss_spk_adv.item()
                    reduced_val_loss_att_means = loss_att_means.item()
                    reduced_val_loss = loss.item()

                val_loss_mel += reduced_val_loss_mel
                val_loss_gate += reduced_val_loss_gate
                val_loss_spk_adv += reduced_val_loss_spk_adv
                val_loss_att_means += reduced_val_loss_att_means
                val_loss += reduced_val_loss

                # [M1] forward_attention_ratio
                _, batch_far = forward_attention_ratio(alignments, input_lengths, output_lengths=output_lengths, mode_mel_length="ground_truth")
                batch_far_list.append(batch_far)
                # [M2] attention_ratio
                ar_pairs = attention_ratio(alignments, input_lengths, text_padded, output_lengths=output_lengths, mode_mel_length="ground_truth")
                batch_ar = ar_pairs[0][1]
                batch_letter_ar = ar_pairs[1][1]
                batch_punct_ar = ar_pairs[2][1]
                batch_blank_ar = ar_pairs[3][1]
                batch_ar_list.append(batch_ar)
                batch_letter_ar_list.append(batch_letter_ar)
                batch_punct_ar_list.append(batch_punct_ar)
                batch_blank_ar_list.append(batch_blank_ar)
                # [M3] attention_range_ratio
                _, batch_arr = attention_range_ratio(alignments, input_lengths, output_lengths=output_lengths, mode_mel_length="ground_truth")
                batch_arr_list.append(batch_arr)
                # [M4] multiple_attention_ratio
                _, batch_mar = multiple_attention_ratio(alignments, input_lengths, output_lengths=output_lengths, mode_mel_length="ground_truth")
                batch_mar_list.append(batch_mar)

                # [M_total] Attention quality
                batch_attention_quality = get_attention_quality(batch_far, batch_ar, batch_arr, batch_mar)

                ############################################################
                # FREE RUNNING #####
                # Forward propagation by free running, i.e., feeding previous outputs to the current inputs.
                _, mel_outputs_postnet_fr, gate_outputs_fr, alignments_fr = model.free_running(text_padded, input_lengths, speakers, emotion_vectors)

                # Computing attention measures.
                # [M1] forward_attention_ratio
                _, batch_far_fr = forward_attention_ratio(alignments_fr, input_lengths, gate_outputs=gate_outputs_fr, mode_mel_length="stop_token")
                batch_far_fr_list.append(batch_far_fr)
                # [M2] attention_ratio
                ar_fr_pairs = attention_ratio(alignments_fr, input_lengths, text_padded, gate_outputs=gate_outputs_fr, mode_mel_length="stop_token")
                batch_ar_fr = ar_fr_pairs[0][1]
                batch_letter_ar_fr = ar_fr_pairs[1][1]
                batch_punct_ar_fr = ar_fr_pairs[2][1]
                batch_blank_ar_fr = ar_fr_pairs[3][1]
                batch_ar_fr_list.append(batch_ar_fr)
                batch_letter_ar_fr_list.append(batch_letter_ar_fr)
                batch_punct_ar_fr_list.append(batch_punct_ar_fr)
                batch_blank_ar_fr_list.append(batch_blank_ar_fr)
                # [M3] attention_range_ratio
                _, batch_arr_fr = attention_range_ratio(alignments_fr, input_lengths, gate_outputs=gate_outputs_fr, mode_mel_length="stop_token")
                batch_arr_fr_list.append(batch_arr_fr)
                # [M4] multiple_attention_ratio
                _, batch_mar_fr = multiple_attention_ratio(alignments_fr, input_lengths, gate_outputs=gate_outputs_fr, mode_mel_length="stop_token")
                batch_mar_fr_list.append(batch_mar_fr)

                # [M_total] Attention quality
                batch_attention_quality_fr = get_attention_quality(batch_far_fr, batch_ar_fr, batch_arr_fr, batch_mar_fr)

                # Wrap up data of audios to be logged.
                inputs = (input_lengths, text_padded, speakers, emotion_vectors)
                outputs = (output_lengths, gate_outputs_fr, mel_padded, mel_outputs_postnet, mel_outputs_postnet_fr, alignments, alignments_fr)
                batch_attention_measures_tf = (batch_attention_quality, batch_ar, batch_letter_ar, batch_punct_ar, batch_blank_ar, batch_arr, batch_mar)
                batch_attention_measures_fr = (batch_attention_quality_fr, batch_ar_fr, batch_letter_ar_fr, batch_punct_ar_fr, batch_blank_ar_fr, batch_arr_fr, batch_mar_fr)
                # [SynthDict 1] A random sample.
                if i == i_batch_rand:
                    i_rand = i_sample_rand
                    synth_dict_rand = dict()
                    fill_synth_dict(synth_dict_rand, i_rand, inputs, outputs,
                            batch_attention_measures_tf, batch_attention_measures_fr)

                # [SynthDict 2] A teacher-forcing sample that has the minimum attention quality.
                if min_attention_quality_tf > batch_attention_quality.min().item():
                    min_attention_quality_tf = batch_attention_quality.min().item()
                    i_min = batch_attention_quality.argmin().item()
                    fill_synth_dict(synth_dict_min_aq_tf, i_min, inputs, outputs,
                            batch_attention_measures_tf, batch_attention_measures_fr)

                # [SynthDict 3] A free-running sample that has the minimum attention quality.
                if min_attention_quality_fr > batch_attention_quality_fr.min().item():
                    min_attention_quality_fr = batch_attention_quality_fr.min().item()
                    i_min = batch_attention_quality_fr.argmin().item()
                    fill_synth_dict(synth_dict_min_aq_fr, i_min, inputs, outputs,
                            batch_attention_measures_tf, batch_attention_measures_fr)

                ############################################################

            ############################################################
            # TEACHER FORCING #####
            n_val_batches = (i + 1)
            val_loss_mel = val_loss_mel / n_val_batches
            val_loss_gate = val_loss_gate / n_val_batches
            val_loss_spk_adv = val_loss_spk_adv / n_val_batches
            val_loss_att_means = val_loss_att_means / n_val_batches
            val_loss = val_loss / n_val_batches

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
            ar_pairs = ((val_mean_ar, val_batch_ar),
                        (val_mean_letter_ar, val_batch_letter_ar),
                        (val_mean_punct_ar, val_batch_punct_ar),
                        (val_mean_blank_ar, val_batch_blank_ar))
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
            ar_fr_pairs = ((val_mean_ar_fr, val_batch_ar_fr),
                           (val_mean_letter_ar_fr, val_batch_letter_ar_fr),
                           (val_mean_punct_ar_fr, val_batch_punct_ar_fr),
                           (val_mean_blank_ar_fr, val_batch_blank_ar_fr))
            # [M3] attention_range_ratio
            val_batch_arr_fr = torch.cat(batch_arr_fr_list)
            val_mean_arr_fr = val_batch_arr_fr.mean().item()
            arr_fr_pair = (val_mean_arr_fr, val_batch_arr_fr)
            # [M4] multiple_attention_ratio
            val_batch_mar_fr = torch.cat(batch_mar_fr_list)
            val_mean_mar_fr = val_batch_mar_fr.mean().item()
            mar_fr_pair = (val_mean_mar_fr, val_batch_mar_fr)


        model.train()
        if rank == 0:
            print("Validation loss {} {}: {:9f}  ".format(str(val_type), iteration, val_loss))
            losses = (val_loss, val_loss_mel, val_loss_gate, val_loss_spk_adv, val_loss_att_means)
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
                'pred_speakers':pred_speakers,
                'losses':losses,
                'attention_measures':attention_measures,
                'attention_measures_fr':attention_measures_fr,
                'gate_accuracy':gate_accuracy,
                'gate_mae':gate_mae,
                'synth_dict_rand':synth_dict_rand,
                'synth_dict_min_aq_tf':synth_dict_min_aq_tf,
                'synth_dict_min_aq_fr':synth_dict_min_aq_fr,
            }
            if hparams.speaker_adversarial_training:
                dict_log_values['spk_adv_accuracy'] = spk_adv_accuracy

            logger.log_validation(valset, val_type, hparams, dict_log_values)


def train(output_directory, log_directory, checkpoint_path, pretrained_path,
          warm_start, n_gpus,
          rank, group_name, hparams, run_name, prj_name, resume):
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
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        0.5**(1 / (125000 * (64 / hparams.batch_size))),
        -1
    )

    if hparams.fp16_run:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O1') # default: opt_level='O2'

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = Tacotron2Loss()
    criterion_dom = torch.nn.CrossEntropyLoss()

    logger = prepare_directories_and_logger(
        hparams,
        output_directory, log_directory, rank, run_name, prj_name, resume)

    train_loader, trainset, valsets, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer, lr_scheduler, logger)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    if pretrained_path is not None:
        model = load_pretrained_model(model, pretrained_path,
            freeze_pretrained=hparams.freeze_pretrained,
            except_for=hparams.freeze_except_for)[0]

    model.train()
    is_overflow = False
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = hparams.learning_rate
        param_group['lr'] = learning_rate

    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Start Epoch {}:".format(epoch+1))
        for i, batch in enumerate(train_loader):
            batches_per_epoch = len(train_loader)
            float_epoch = iteration / batches_per_epoch
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_scheduler.get_lr()[0]

            model.zero_grad()

            # Parse the current batch.
            x, y, etc = model.parse_batch(batch)
            text_padded, input_lengths, mel_padded, max_len, output_lengths = x
            mel_padded, gate_padded = y
            speakers, sex, emotion_vectors, lang = etc

            # Forward propagtion
            y_pred, y_pred_speakers, att_means = model(x, speakers, emotion_vectors)

            # Forward propagtion results
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = y_pred
            spk_logit_outputs, prob_speakers, pred_speakers = y_pred_speakers

            # Compute stop gate accuracy
            np_output_lengths = output_lengths.cpu().numpy()
            mel_lengths = get_mel_lengths(gate_outputs)
            np_mel_lengths = mel_lengths.cpu().numpy()
            gate_accuracy = accuracy_score(np_output_lengths, np_mel_lengths)
            # Compute stop gate MAE(pred_lengths, true_lengths)
            gate_mae = mean_absolute_error(np_output_lengths, np_mel_lengths)

            # Caculate Tacotron2 losses.
            loss_taco2, loss_mel, loss_gate = criterion(y_pred, y)

            # Forward the speaker adversarial module.
            if hparams.speaker_adversarial_training:
                spk_adv_targets = get_spk_adv_targets(speakers, input_lengths)
                # Compute the speaker adversarial loss.
                loss_spk_adv = criterion_dom(spk_logit_outputs, spk_adv_targets)
                # Compute the accuracy of the adversarial speaker classifier.
                np_speakers = spk_adv_targets.cpu().numpy()
                np_pred_speakers = pred_speakers.cpu().numpy()
                spk_adv_accuracy = accuracy_score(np_speakers, np_pred_speakers)
            else:
                loss_spk_adv = torch.zeros(1).cuda()

            if hparams.monotonic_attention:
                input_lengths = x[1]
                loss_att_means = MSELoss()(att_means, input_lengths.float())
            else:
                loss_att_means = torch.zeros(1).cuda()

            loss = loss_taco2 + hparams.speaker_adv_weight * loss_spk_adv + \
                hparams.loss_att_means_weight * loss_att_means

            if prj_name == "forward_attention_loss":
                input_lengths = x[1]
                gate_outputs = y_pred[2]
                alignments = y_pred[3]
                mean_far, _ = forward_attention_ratio(alignments, input_lengths, output_lengths=output_lengths, mode_mel_length="ground_truth")
                if mean_far > 0.95:
                    fa_loss = forward_attention_loss(alignments, input_lengths, output_lengths=output_lengths, mode_mel_length="ground_truth")
                    loss += fa_loss
                    float_fa_loss = fa_loss.item()
                else:
                    float_fa_loss = None
            else:
                float_fa_loss = None

            if hparams.distributed_run:
                reduced_loss_mel = reduce_tensor(loss_mel.data, n_gpus).item()
                reduced_loss_gate = reduce_tensor(loss_gate.data, n_gpus).item()
                reduced_loss_spk_adv = reduce_tensor(loss_spk_adv.data, n_gpus).item()
                reduced_loss_att_means = reduce_tensor(reduced_loss_att_means.data, n_gpus).item()
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss_mel = loss_mel.item()
                reduced_loss_gate = loss_gate.item()
                reduced_loss_spk_adv = loss_spk_adv.item()
                reduced_loss_att_means = loss_att_means.item()
                reduced_loss = loss.item()
            if hparams.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if hparams.fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), hparams.grad_clip_thresh)
                is_overflow = math.isnan(grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)

            learning_rate = lr_scheduler.get_lr()[0]
            optimizer.step()

            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                print("Iteration {} Learning rate {} Train total loss {:.6f} Mel loss {:.6f} Gate loss {:.6f} SpkAdv loss {:.6f} MonoAtt MSE loss {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration, learning_rate, reduced_loss, reduced_loss_mel, reduced_loss_gate, reduced_loss_spk_adv, reduced_loss_att_means, grad_norm, duration))
                reduced_losses = (reduced_loss, reduced_loss_mel, reduced_loss_gate, reduced_loss_spk_adv, reduced_loss_att_means)

                dict_log_values = {
                    'iteration':iteration,
                    'epoch':float_epoch,
                    'losses':reduced_losses,
                    'grad_norm':grad_norm,
                    'learning_rate':learning_rate,
                    'duration':duration,
                    'x':x,
                    'etc':etc,
                    'y_pred':y_pred,
                    'pred_speakers':pred_speakers,
                    'gate_accuracy':gate_accuracy,
                    'gate_mae':gate_mae,
                }
                if hparams.speaker_adversarial_training:
                    dict_log_values['spk_adv_accuracy'] = spk_adv_accuracy

                logger.log_training(hparams, dict_log_values, batches_per_epoch)

            if not is_overflow and ((iteration % hparams.iters_per_checkpoint == 0) or (i+1 == batches_per_epoch)):
                criterions = (criterion, criterion_dom)
                validate(model, criterions, valsets, iteration, float_epoch,
                         hparams.batch_size, n_gpus, collate_fn, logger,
                         hparams.distributed_run, rank, hparams)
                if rank == 0 and (iteration % hparams.iters_per_checkpoint == 0):
                    checkpoint_path = os.path.join(
                        os.path.join(output_directory, prj_name, run_name), "checkpoint_{}-epoch_{:.4}".format(iteration, float_epoch))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    lr_scheduler, logger, checkpoint_path)
                if rank == 0 and (i+1 == batches_per_epoch):
                    checkpoint_path = os.path.join(
                        os.path.join(output_directory, prj_name, run_name), "checkpoint_{}-epoch_{:.4}_end-epoch_{}".format(iteration, float_epoch, epoch+1))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    lr_scheduler, logger, checkpoint_path)

            tmp_iteration = iteration
            tmp_learning_rate = learning_rate

            def signal_handler(sig, frame):
                print('You pressed Ctrl+C!')
                if rank == 0:
                    checkpoint_path = os.path.join(
                        os.path.join(output_directory, prj_name, run_name), "checkpoint_{}-epoch_{:.4}".format(iteration, float_epoch))
                    save_checkpoint(model, optimizer, tmp_learning_rate, tmp_iteration,
                                    lr_scheduler, logger, checkpoint_path)
                sys.exit(0)

            signal.signal(signal.SIGINT, signal_handler)

            if iteration > round(50000 * (64 / hparams.batch_size)):
                lr_scheduler.step()

            iteration += 1

        # End of the current epoch
        # Upsampling again
        trainset.upsampling(iteration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
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
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)
    print("Visible GPU IDs:", args.visible_gpus)

    train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.pretrained_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams,
          args.run_name, args.prj_name, args.resume)
