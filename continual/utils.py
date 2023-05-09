# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import sys
import time
from collections import defaultdict, deque
from typing import Dict, Any, Union
from argparse import Namespace
import datetime
import warnings

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy

import torch
import numpy as np
from torch import nn
import torch.distributed as dist


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def update_dict(self, d):
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if len(iterable) > 0:
            print('{} Total time: {} ({:.4f} s / it)'.format(
                header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def progress_bar(i: int, max_iter: int, epoch: Union[int, str],
                 task_number: int, loss: float) -> None:
    """
    Prints out the progress bar on the stderr file.
    :param i: the current iteration
    :param max_iter: the maximum number of iteration
    :param epoch: the epoch
    :param task_number: the task index
    :param loss: the current value of the loss function
    """
    if not (i + 1) % 10 or (i + 1) == max_iter:
        progress = min(float((i + 1) / max_iter), 1)
        progress_bar = ('█' * int(50 * progress)) + ('┈' * (50 - int(50 * progress)))
        print('\r[ {} ] Task {} | epoch {}: |{}| loss: {}'.format(
            datetime.datetime.now().strftime("%m-%d | %H:%M"),
            task_number + 1 if isinstance(task_number, int) else task_number,
            epoch,
            progress_bar,
            round(loss, 8)
        ), file=sys.stderr, end='', flush=True)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def load_first_task_model(model_without_ddp, loss_scaler, task_id, args):
    strict = False

    if args.resume.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.resume, map_location='cpu', check_hash=True)
    elif os.path.isdir(args.resume):
        path = os.path.join(args.resume, f"checkpoint_{task_id}.pth")
        checkpoint = torch.load(path, map_location='cpu')
    else:
        checkpoint = torch.load(args.resume, map_location='cpu')

    model_ckpt = checkpoint['model']

    if not strict:
        for i in range(1, 6):
            k = f"head.fcs.{i}.weight"
            if k in model_ckpt: del model_ckpt[k]
            k = f"head.fcs.{i}.bias"
            if k in model_ckpt: del model_ckpt[k]
    model_without_ddp.load_state_dict(model_ckpt, strict=strict)
    if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        if 'scaler' in checkpoint:
            try:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            except:
                warnings.warn("Could not reload loss scaler, probably because of amp/noamp mismatch")


def change_pos_embed_size(pos_embed, new_size=32, patch_size=16, old_size=224):
    nb_patches = (new_size // patch_size) ** 2
    new_pos_embed = torch.randn(1, nb_patches + 1, pos_embed.shape[2])
    new_pos_embed[0, 0] = pos_embed[0, 0]

    lo_idx = 1
    for i in range(nb_patches):
        hi_idx = lo_idx + old_size // nb_patches
        new_pos_embed[0, i] = pos_embed[0, lo_idx:hi_idx].mean(dim=0)
        lo_idx = hi_idx

    return torch.nn.Parameter(new_pos_embed)


def freeze_parameters(m, requires_grad=False):
    if m is None:
        return

    if isinstance(m, nn.Parameter):
        m.requires_grad = requires_grad
    else:
        for p in m.parameters():
            p.requires_grad = requires_grad


def fetch_fname(args):
    if args.representation_replay:
        return "representation_replay"
    else:
        return "default_settings"


class TensorboardLogger:
    def __init__(self, args: Namespace, setting: str,
                 stash: Dict[Any, str]=None) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.settings = [setting]
        # if setting == 'class-il':
        #     self.settings.append('task-il')
        self.loggers = {}
        self.name = stash['model_name']
        # important arguments
        imp_args = ['log_dir', 'seed', 'attn_weight', 'attn_version', 'no_distillation', 'distill_version',
                    'distill_weight', 'sep_memory', 'cos_weight', 'dynamic_tokens', 'representation_replay',
                    'replay_from', 'finetuning']
        for a_setting in self.settings:
            self.loggers[a_setting] = SummaryWriter(
                os.path.join(args.output_basedir, args.log_path, 'tensorboard_runs', a_setting, self.name),
                purge_step=stash['task_idx'] * args.base_epochs + stash['epoch_idx']+1)
        config_text = ', '.join(
            ["%s=%s" % (name, getattr(args, name)) for name in args.__dir__()
             if not name.startswith('_') if name in imp_args])
        for a_logger in self.loggers.values():
            a_logger.add_text('config', config_text)

    def get_name(self) -> str:
        """
        :return: the name of the model
        """
        return self.name

    def get_log_dir(self):
        if 'class-il' in self.loggers.keys():
            return self.loggers['class-il'].log_dir
        elif 'domain-il' in self.loggers.keys():
            return self.loggers['domain-il'].log_dir
        else:
            return self.loggers['general-continual'].log_dir

    def log_accuracy(self, all_accs: np.ndarray, all_mean_accs: np.ndarray,
                     args: Namespace, task_number: int) -> None:
        """
        Logs the current accuracy value for each task.
        :param all_accs: the accuracies (class-il, task-il) for each task
        :param all_mean_accs: the mean accuracies for (class-il, task-il)
        :param args: the arguments of the run
        :param task_number: the task index
        """
        mean_acc_common = all_mean_accs
        for setting, a_logger in self.loggers.items():
            mean_acc = mean_acc_common
            # index = 1 if setting == 'task-il' else 0
            accs = [all_accs[kk] for kk in range(len(all_accs))]
            for kk, acc in enumerate(accs):
                a_logger.add_scalar('acc_task%02d' % (kk + 1), acc,
                                    task_number * args.base_epochs)
            a_logger.add_scalar('acc_mean', mean_acc, task_number * args.base_epochs)

    def log_accuracy5(self, all_accs: np.ndarray, all_mean_accs: np.ndarray,
                     args: Namespace, task_number: int) -> None:
        """
        Logs the current top-5 accuracy value for each task.
        :param all_accs: the accuracies (class-il, task-il) for each task
        :param all_mean_accs: the mean accuracies for (class-il, task-il)
        :param args: the arguments of the run
        :param task_number: the task index
        """
        mean_acc_common = all_mean_accs
        for setting, a_logger in self.loggers.items():
            mean_acc = mean_acc_common
            # index = 1 if setting == 'task-il' else 0
            accs = [all_accs[kk] for kk in range(len(all_accs))]
            for kk, acc in enumerate(accs):
                a_logger.add_scalar('acc_task%02d' % (kk + 1), acc,
                                    task_number * args.base_epochs)
            a_logger.add_scalar('acc_mean5', mean_acc, task_number * args.base_epochs)

    def log_loss(self, loss: float, n_epochs: int, epoch: int,
                 task_number: int) -> None:
        """
        Logs the loss value at each iteration.
        :param loss: the loss value
        :param args: the arguments of the run
        :param epoch: the epoch index
        :param task_number: the task index
        :param iteration: the current iteration
        """
        for a_logger in self.loggers.values():
            a_logger.add_scalar('loss', loss, task_number * n_epochs + epoch)

    def log_lr(self, lr: float, n_epochs: int, epoch: int,
                 task_number: int) -> None:
        """
        Logs the loss value at each iteration.
        :param lr: the learning rate value
        :param args: the arguments of the run
        :param epoch: the epoch index
        :param task_number: the task index
        :param iteration: the current iteration
        """
        for a_logger in self.loggers.values():
            a_logger.add_scalar('lr', lr, task_number * n_epochs + epoch)

    def log_ssl_loss(self, loss: float, n_epochs: int,
                    epoch: int, task_number: int) -> None:
        """
        Logs the loss value at each iteration.
        :param loss: the loss value
        :param args: the arguments of the run
        :param epoch: the epoch index
        :param task_number: the task index
        :param iteration: the current iteration
        """
        for a_logger in self.loggers.values():
            a_logger.add_scalar('ssl_loss', loss, task_number * n_epochs + epoch)

    def log_loss_rotation(self, loss: float, n_epochs: int,
                    epoch: int, task_number: int) -> None:
        """
        Logs the loss value at each iteration.
        :param loss: the loss value
        :param args: the arguments of the run
        :param epoch: the epoch index
        :param task_number: the task index
        :param iteration: the current iteration
        """
        for a_logger in self.loggers.values():
            a_logger.add_scalar('second_loss', loss, task_number * n_epochs + epoch)

    def log_div_loss(self, loss: float, n_epochs: int,
                    epoch: int, task_number: int) -> None:
        """
        Logs the loss value at each iteration.
        :param loss: the loss value
        :param args: the arguments of the run
        :param epoch: the epoch index
        :param task_number: the task index
        :param iteration: the current iteration
        """
        for a_logger in self.loggers.values():
            a_logger.add_scalar('Divergence loss', loss, task_number * n_epochs + epoch)

    def log_kd_loss(self, loss: float, n_epochs: int,
                    epoch: int, task_number: int) -> None:
        """
        Logs the loss value at each iteration.
        :param loss: the loss value
        :param args: the arguments of the run
        :param epoch: the epoch index
        :param task_number: the task index
        :param iteration: the current iteration
        """
        for a_logger in self.loggers.values():
            a_logger.add_scalar('KD_loss', loss, task_number * n_epochs + epoch)

    def log_attn_loss(self, loss: float, n_epochs: int,
                    epoch: int, task_number: int) -> None:
        """
        Logs the loss value at each iteration.
        :param loss: the loss value
        :param args: the arguments of the run
        :param epoch: the epoch index
        :param task_number: the task index
        :param iteration: the current iteration
        """
        for a_logger in self.loggers.values():
            a_logger.add_scalar('Attention_loss', loss, task_number * n_epochs + epoch)

    def log_cos_loss(self, loss: float, n_epochs: int,
                    epoch: int, task_number: int) -> None:
        """
        Logs the loss value at each iteration.
        :param loss: the loss value
        :param args: the arguments of the run
        :param epoch: the epoch index
        :param task_number: the task index
        :param iteration: the current iteration
        """
        for a_logger in self.loggers.values():
            a_logger.add_scalar('Cosine_loss', loss, task_number * n_epochs + epoch)

    def log_loss_gcl(self, loss: float, iteration: int) -> None:
        """
        Logs the loss value at each iteration.
        :param loss: the loss value
        :param iteration: the current iteration
        """
        for a_logger in self.loggers.values():
            a_logger.add_scalar('loss', loss, iteration)

    def close(self) -> None:
        """
        At the end of the execution, closes the logger.
        """
        for a_logger in self.loggers.values():
            a_logger.close()


def plot_task_performance(task_perf_path, num_tasks, tb_logger, task_wise_acc, teacher_plot=False):
    np_perf = np.loadtxt(task_perf_path)

    n_rows, n_cols = 1, 1
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(5, 5), sharey=True)

    x_labels = [f"Task {ci}" for ci in range(1, num_tasks + 1)]
    y_labels = [f"After Task {ci}" for ci in range(1, num_tasks + 1)]

    im = ax.imshow(task_wise_acc, cmap='YlOrRd')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            text = ax.text(j, i, np_perf[i, j], ha="center", va="center", color="darkgray", fontsize=8)
    ax.set_title('Task performance of Model', fontsize=17)

    fig.tight_layout()
    # plt.show()
    if teacher_plot:
        plot_path = os.path.join(tb_logger.loggers['class-il'].log_dir, 'teacher_task_performance.png')
    else:
        plot_path = os.path.join(tb_logger.loggers['class-il'].log_dir, 'task_performance.png')

    fig.savefig(plot_path, bbox_inches='tight')
    fig.savefig(os.path.join(tb_logger.loggers['class-il'].log_dir, 'task_performance.pdf'), bbox_inches='tight')


def mixup_batch_representations(x, y, alpha=1.0, use_cuda=True, aug_prob=0.):
    '''Returns mixed inputs, pairs of targets, and lambda
    source: https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
    '''
    batch_size = x.size()[0]

    aug_indexes = torch.rand(batch_size) < aug_prob

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    if use_cuda:
        random_index = torch.randperm(batch_size).cuda()
    else:
        random_index = torch.randperm(batch_size)

    x[torch.where(aug_indexes)] = lam*x[torch.where(aug_indexes)] + (1 - lam)*x[random_index[torch.where(aug_indexes)]]

    y_a = y
    y_b = deepcopy(y)
    y_b[torch.where(aug_indexes)] = y[random_index[torch.where(aug_indexes)]]
    return x, y_a, y_b, lam, aug_indexes


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_batch_representations(x, y, alpha=1.0, use_cuda=True, aug_prob=0.):
    '''Returns cut_mixed inputs, pairs of targets, and lambda
    source: https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
    '''
    batch_size = x.size()[0]

    aug_indexes = torch.rand(batch_size) < aug_prob

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    if use_cuda:
        random_index = torch.randperm(batch_size).cuda()
    else:
        random_index = torch.randperm(batch_size)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.shape, lam)
    x[aug_indexes, :, bbx1:bbx2, bby1:bby2] = x[random_index[aug_indexes], :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.shape[-1] * x.shape[-2]))

    y_a = y
    y_b = deepcopy(y)
    y_b[torch.where(aug_indexes)] = y[random_index[torch.where(aug_indexes)]]
    return x, y_a, y_b, lam, aug_indexes


def mixup_representations(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda
    source: https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
    '''
    batch_size = x.size()[0]

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def get_random_labels(orig_targets, start_index, end_index, noise_rate):

    y_targeted = orig_targets

    noise_mask = (
            torch.FloatTensor(size=orig_targets.shape).uniform_(0, 1)
            < noise_rate
    ).to(orig_targets.device)
    rand_labels = torch.randint(low=start_index, high=end_index, size=orig_targets.shape).to(orig_targets.device)
    # rand_labels = torch.fmod(rand_labels + orig_targets, end_index-start_index)
    y_targeted = torch.where(noise_mask, rand_labels, orig_targets)

    return y_targeted