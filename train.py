import os
import time
import argparse
import math
from numpy import finfo

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss, forward_attention_loss
from logger import Tacotron2Logger
from hparams import create_hparams
from measures import forward_attention_ratio
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
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams, 'train')
    valset = TextMelLoader(hparams, 'val')
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
    return train_loader, valset, collate_fn


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


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, criterions, valset, iteration, epoch, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank, hparams):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        criterion, criterion_dom = criterions
        val_loss = 0.0
        mean_far_sum = 0.0
        batch_far_list = list()
        for i, batch in enumerate(val_loader):
            x, y, etc = model.parse_batch(batch)
            input_lengths = x[1]
            speakers, sex, emotion_vectors, lang = etc
            y_pred, y_pred_speakers = model(x, speakers, emotion_vectors)
            alignments = y_pred[3]
            (spk_logit_outputs, prob_speakers, pred_speakers) = y_pred_speakers

            loss_mel = criterion(y_pred, y)
            if hparams.speaker_adversarial_training:
                spk_adv_targets = get_spk_adv_targets(speakers, input_lengths)
                loss_spk_adv = criterion_dom(spk_logit_outputs, spk_adv_targets)
            else:
                loss_spk_adv = torch.zeros(1)
            loss = loss_mel + hparams.speaker_adv_weight * loss_spk_adv

            if distributed_run:
                reduced_val_loss_mel = reduce_tensor(loss_mel.data, n_gpus).item()
                reduced_val_loss_spk_adv = reduce_tensor(loss_spk_adv.data, n_gpus).item()
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss_mel = loss_mel.item()
                reduced_val_loss_spk_adv = loss_spk_adv.item()
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
            mean_far, batch_far = forward_attention_ratio(alignments, input_lengths)
            mean_far_sum += mean_far
            batch_far_list.append(batch_far)
        val_loss = val_loss / (i + 1)
        val_mean_far = mean_far_sum / (i + 1)
        val_batch_far = torch.cat(batch_far_list)
        far_pair = (val_mean_far, val_batch_far)

    model.train()
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, reduced_val_loss))
        reduced_val_losses = (reduced_val_loss, reduced_val_loss_mel, reduced_val_loss_spk_adv)
        logger.log_validation(valset,
            reduced_val_losses, far_pair,
            model, x, y, etc, y_pred, pred_speakers,
            iteration, epoch, hparams)


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

    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = Tacotron2Loss()
    criterion_dom = torch.nn.CrossEntropyLoss()

    logger = prepare_directories_and_logger(
        hparams,
        output_directory, log_directory, rank, run_name, prj_name, resume)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    if pretrained_path is not None:
        model = load_pretrained_model(model, pretrained_path)[0]

    model.train()
    is_overflow = False
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = learning_rate
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        0.5**(1 / (125000 * (64 / hparams.batch_size))),
        last_epoch=-1
    )
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            batches_per_epoch = len(train_loader)
            float_epoch = iteration / batches_per_epoch
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group['lr'] = scheduler.get_lr()[0]

            model.zero_grad()

            # Parse the current batch.
            x, y, etc = model.parse_batch(batch)
            speakers, sex, emotion_vectors, lang = etc
            y_pred, y_pred_speakers = model(x, speakers, emotion_vectors)
            (spk_logit_outputs, prob_speakers, pred_speakers) = y_pred_speakers

            # Caculate losses.
            loss_mel = criterion(y_pred, y)
            if hparams.speaker_adversarial_training:
                input_lengths = x[1]
                spk_adv_targets = get_spk_adv_targets(speakers, input_lengths)
                loss_spk_adv = criterion_dom(spk_logit_outputs, spk_adv_targets)
            else:
                loss_spk_adv = torch.zeros(1).cuda()
            loss = loss_mel + hparams.speaker_adv_weight * loss_spk_adv

            if prj_name == "forward_attention_loss":
                input_lengths = x[1]
                alignments = y_pred[3]
                mean_far, _ = forward_attention_ratio(alignments, input_lengths)
                if mean_far > 0.95:
                    fa_loss = forward_attention_loss(alignments, input_lengths)
                    loss += fa_loss
                    float_fa_loss = fa_loss.item()
                else:
                    float_fa_loss = None
            else:
                float_fa_loss = None

            if hparams.distributed_run:
                reduced_loss_mel = reduce_tensor(loss_mel.data, n_gpus).item()
                reduced_loss_spk_adv = reduce_tensor(loss_spk_adv.data, n_gpus).item()
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss_mel = loss_mel.item()
                reduced_loss_spk_adv = loss_spk_adv.item()
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

            learning_rate = scheduler.get_lr()[0]
            print("learning_rate:", learning_rate)
            optimizer.step()

            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                print("Iteration {} Train total loss {:.6f} Mel loss {:.6f} SpkAdv loss {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration, reduced_loss, reduced_loss_mel, reduced_loss_spk_adv, grad_norm, duration))
                reduced_losses = (reduced_loss, reduced_loss_mel, reduced_loss_spk_adv)
                logger.log_training(
                    reduced_losses, grad_norm, learning_rate, duration, x, etc,
                    y_pred, pred_speakers,
                    iteration, float_epoch, float_fa_loss)

            if not is_overflow and ((iteration % hparams.iters_per_checkpoint == 0) or (i+1 == batches_per_epoch)):
                criterions = (criterion, criterion_dom)
                validate(model, criterions, valset, iteration, float_epoch,
                         hparams.batch_size, n_gpus, collate_fn, logger,
                         hparams.distributed_run, rank, hparams)
                if rank == 0 and (iteration % hparams.iters_per_checkpoint == 0):
                    checkpoint_path = os.path.join(
                        os.path.join(output_directory, prj_name, run_name), "checkpoint_{}-epoch_{:.4}".format(iteration, float_epoch))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)
                if rank == 0 and (i+1 == batches_per_epoch):
                    checkpoint_path = os.path.join(
                        os.path.join(output_directory, prj_name, run_name), "checkpoint_{}-epoch_{:.4}_end-epoch_{}".format(iteration, float_epoch, epoch))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)

            if iteration > round(50000 * (64 / hparams.batch_size)):
                scheduler.step()

            iteration += 1


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
    parser.add_argument('-r', '--resume', type=bool, default=False,
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
