# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

# Modified for DyTox by Arthur Douillard
import argparse
import copy
import datetime
import json
import os
import statistics
import time
import warnings
from pathlib import Path
import yaml
import random
import csv
import pickle
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from continuum.metrics import Logger
from continuum.tasks import split_train_val
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from continual.mixup import Mixup
import continual.utils as utils
from continual import factory, scaler
from continual.classifier import Classifier
from continual.rehearsal import Memory, get_finetuning_dataset, get_separate_finetuning_dataset
from continual.sam import SAM
from continual.datasets import build_dataset
from continual.engine import eval_and_log, train_one_epoch, evaluate_teacher
from continual.losses import bce_with_logits

warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser('DyTox training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--incremental-batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--base-epochs', default=500, type=int,
                        help='Number of epochs for base task')
    parser.add_argument('--no-amp', default=False, action='store_true',
                        help='Disable mixed precision')

    # Model parameters
    parser.add_argument('--model', default='')
    parser.add_argument('--input-size', default=32, type=int, help='images input size')
    parser.add_argument('--patch-size', default=16, type=int)
    parser.add_argument('--embed-dim', default=768, type=int)
    parser.add_argument('--depth', default=12, type=int)
    parser.add_argument('--num-heads', default=12, type=int)
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--norm', default='layer', choices=['layer', 'scale'],
                        help='Normalization layer type')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--incremental-lr", default=None, type=float,
                        help="LR to use for incremental task (t > 0)")
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--incremental-warmup-lr', type=float, default=None, metavar='LR',
                        help='warmup learning rate (default: 1e-6) for task T > 0')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem", "old"')

    # Distillation parameters
    parser.add_argument('--auto-kd', default=False, action='store_true',
                        help='Balance kd factor as WA https://arxiv.org/abs/1911.07053')
    parser.add_argument('--distillation-tau', default=1.0, type=float,
                        help='Temperature for the KD')

    # Dataset parameters
    parser.add_argument('--data-path', default='', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--output-dir', default='',
                        help='Dont use that')
    parser.add_argument('--output-basedir', default='./checkponts/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # Continual Learning parameters
    parser.add_argument("--initial-increment", default=50, type=int,
                        help="Base number of classes")
    parser.add_argument("--increment", default=10, type=int,
                        help="Number of new classes per incremental task")
    parser.add_argument('--class-order', default=None, type=int, nargs='+',
                        help='Class ordering, a list of class ids.')

    parser.add_argument("--eval-every", default=50, type=int,
                        help="Eval model every X epochs, if None only eval at the task end")
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Only do one batch per epoch')
    parser.add_argument('--max-task', default=None, type=int,
                        help='Max task id to train on')
    parser.add_argument('--name', default='birt', help='Name to display for screen')
    parser.add_argument('--options', default=[], nargs='*')
    # Adding number of tasks parameter for rotated mnist
    parser.add_argument("--number_of_tasks", default=5, type=int,
                        help="Base number of tasks for Rotated MNIST")

    # DyTox related
    parser.add_argument('--dytox', action='store_true', default=False,
                        help='Enable super DyTox god mode.')
    parser.add_argument('--ind-clf', default='', choices=['1-1', '1-n', 'n-n', 'n-1'],
                        help='Independent classifier per task but predicting all seen classes')
    parser.add_argument('--joint-tokens', default=False, action='store_true',
                        help='Forward w/ all task tokens alltogether [Faster but not working as well, not sure why')

    # Diversity
    parser.add_argument('--head-div', default=0., type=float,
                        help='Use a divergent head to predict among new classes + 1 using last token')
    parser.add_argument('--head-div-mode', default=['tr', 'ft'], nargs='+', type=str,
                        help='Only do divergence during training (tr) and/or finetuning (ft).')

    # SAM-related parameters
    # SAM fails with Mixed Precision, so use --no-amp
    parser.add_argument('--sam-rho', default=0., type=float,
                        help='Rho parameters for Sharpness-Aware Minimization. Disabled if == 0.')
    parser.add_argument('--sam-adaptive', default=False, action='store_true',
                        help='Adaptive version of SAM (more robust to rho)')
    parser.add_argument('--sam-first', default='main', choices=['main', 'memory'],
                        help='Apply SAM first step on main or memory loader (need --sep-memory for the latter)')
    parser.add_argument('--sam-second', default='main', choices=['main', 'memory'],
                        help='Apply SAM second step on main or memory loader (need --sep-memory for the latter)')
    parser.add_argument('--sam-skip-first', default=False, action='store_true',
                        help='Dont use SAM for first task')
    parser.add_argument('--sam-final', default=None, type=float,
                        help='Final value of rho is it is changed linearly per task.')
    parser.add_argument('--sam-div', default='', type=str,
                        choices=['old_no_upd'],
                        help='SAM for diversity')
    parser.add_argument('--sam-mode', default=['tr', 'ft'], nargs='+', type=str,
                        help='Only do SAM during training (tr) and/or finetuning (ft).')
    parser.add_argument('--look-sam-k', default=0, type=int,
                        help='Apply look sam every K updates (see under review ICLR22)')
    parser.add_argument('--look-sam-alpha', default=0.7, type=float,
                        help='Alpha factor of look sam to weight gradient reuse, 0 < alpha <= 1')

    # Rehearsal memory
    parser.add_argument('--memory-size', default=2000, type=int,
                        help='Total memory size in number of stored (image, label).')
    parser.add_argument('--fixed-memory', default=False, action='store_true',
                        help='Dont fully use memory when no all classes are seen as in Hou et al. 2019')
    parser.add_argument('--rehearsal', default="random",
                        choices=[
                            'random',
                            'closest_token', 'closest_all',
                            'icarl_token', 'icarl_all',
                            'furthest_token', 'furthest_all'
                        ],
                        help='Method to herd sample for rehearsal.')
    parser.add_argument('--replay-memory', default=0, type=int,
                        help='Replay memory according to Guido rule [NEED DOC]')

    # Finetuning
    parser.add_argument('--finetuning', default='', choices=['balanced'],
                        help='Whether to do a finetuning after each incremental task. Backbone are frozen.')
    parser.add_argument('--finetuning-epochs', default=30, type=int,
                        help='Number of epochs to spend in finetuning.')
    parser.add_argument('--finetuning-lr', default=5e-5, type=float,
                        help='LR during finetuning, will be kept constant.')
    parser.add_argument('--finetuning-teacher', default=False, action='store_true',
                        help='Use teacher/old model during finetuning for all kd related.')
    parser.add_argument('--finetuning-resetclf', default=False, action='store_true',
                        help='Reset classifier before finetuning phase (similar to GDumb/DER).')
    parser.add_argument('--only-ft', default=False, action='store_true',
                        help='Only train on FT data')

    # What to freeze
    parser.add_argument('--freeze-task', default=[], nargs="*", type=str,
                        help='What to freeze before every incremental task (t > 0).')
    parser.add_argument('--freeze-ft', default=[], nargs="*", type=str,
                        help='What to freeze before every finetuning (t > 0).')
    parser.add_argument('--freeze-eval', default=False, action='store_true',
                        help='Frozen layers are put in eval. Important for stoch depth')

    # Convit - CaiT
    parser.add_argument('--local-up-to-layer', default=10, type=int,
                        help='number of GPSA layers')
    parser.add_argument('--locality-strength', default=1., type=float,
                        help='Determines how focused each head is around its attention center')
    parser.add_argument('--class-attention', default=False, action='store_true',
                        help='Freeeze and Process the class token as done in CaiT')

    # Logs
    parser.add_argument('--log-path', default="logs")
    parser.add_argument('--log-category', default="misc")

    # Classification
    parser.add_argument('--bce-loss', default=False, action='store_true')

    # distributed training parameters
    parser.add_argument('--local_rank', default=None, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Resuming
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-task', default=0, type=int, help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, help='resume from checkpoint')
    parser.add_argument('--save-every-epoch', default=None, type=int)

    parser.add_argument('--validation', default=0.0, type=float,
                        help='Use % of the training set as val, replacing the test.')

    # Additional Losses
    parser.add_argument('--attn_weight', default=0.0, type=float,
                        help='Weight for mse loss between teacher model and current model attention map.')
    parser.add_argument('--previous_attention', default=False, action='store_true',
                        help='Whether to include the Convit layer attention or not.')
    parser.add_argument('--no_distillation', default=False, action='store_true',
                        help='Switch off distillation from teacher model.')
    parser.add_argument('--sep_memory', default=False, action='store_true',
                        help='Useful for different forward passes for buffered memory samples.')
    parser.add_argument('--cos_weight', default=0.0, type=float,
                        help='Weight for cosine similarity loss between task tokens.')
    parser.add_argument('--dynamic_tokens', default=1.0, type=float,
                        help='Dynamically backpropagate gradients to only topk percent tokens.')
    parser.add_argument('--representation_replay', default=False, action='store_true',
                        help='Replay representations rather than input image from second task, \
                              sep memory MUST BE true for this to work')
    parser.add_argument('--replay_from', default=1, type=int,
                        help='From which layer to extract features, layers below will be frozen.')
    parser.add_argument('--attn_version', default='v1', type=str,
                        help='Different attention consistency losses to try.')
    parser.add_argument('--distill_version', default='kl', type=str,
                        help='Different approaches to try distillation from the teacher model. ["kl", "mse", "linf", "l2"]')
    parser.add_argument('--tensorboard', default=False, action='store_true',
                        help='Log the accuracies and losses in tensorboard')
    parser.add_argument('--distill_weight', default=1., type=float,
                        help='Weights for distillation loss from teacher model')
    parser.add_argument('--distill_weight_buffer', default=1., type=float,
                        help='Weights for distillation loss from teacher model for buffer samples')
    parser.add_argument('--ema_alpha', default=0., type=float,
                        help='Alpha value to update the EMA model')
    parser.add_argument('--evaluate_teacher', default=False, action='store_true',
                        help='Whether to evaluate the teacher model at the end of task 2')
    parser.add_argument('--single_head', default=False, type=bool,
                        help='Train Dytox without expanding task tokens and classifiers')
    parser.add_argument('--multi_token_setup', default=False, type=bool,
                        help='Train Dytox with expanding only task tokens and not classifiers')
    parser.add_argument('--use_repeatedaug_single', default=False, action='store_true',
                        help='Whether to use repeated augmentation in single GPU setting as well')
    parser.add_argument('--window_size', default=7, type=int,
                        help='Window size to use in Swin Transformer.')
    parser.add_argument('--rep_mixup_alpha', default=0.1, type=float,
                        help='Augment the intermediate features during replay if rep_mixup_alpha>0.')
    parser.add_argument('--rep_noise_weight', default=0., type=float,
                        help='Augment the intermediate features with gaussian noise if rep_noise_weight>0.')
    parser.add_argument('--mixup_prob', default=0.5, type=float,
                        help='Probability for augmenting the representations using mixup')
    parser.add_argument('--rep_cutmix_alpha', default=0., type=float,
                        help='Augment the intermediate features during replay if rep_cutmix alpha>0.')
    parser.add_argument('--cutmix_prob', default=0.5, type=float,
                        help='Probability for augmenting the representations using cutmix')
    parser.add_argument('--shot_prob', default=0.5, type=float,
                        help='Probability for augmenting the representations using shotnoise')
    parser.add_argument('--tensor_prob', default=0.5, type=float,
                        help='Probability for augmenting the representations using gaussian noise with tensor std')
    parser.add_argument('--repnoise_prob', default=0.5, type=float,
                        help='Probability for augmenting the representations using gaussian noise')
    parser.add_argument('--shotnoise_strength', default=0, type=int,
                        help='Strength of shot noise to add to the representations')
    parser.add_argument('--tensor_noise_weight', default=0., type=float,
                        help='Weight for input dependent noise added to representations')
    parser.add_argument('--logit_noise_dev', default=0.01, type=float,
                        help='Standand deviation to sample noise for logit replay')
    parser.add_argument('--ext_lambda', default=0.5, type=float,
                        help='Lambda for extrapolating between feature samples')
    parser.add_argument('--scale_prob', default=0., type=float,
                        help='Scale for representations')
    parser.add_argument('--ema_frequency', default=0., type=float,
                        help='Frequency in terms of probability to do EMA update of teacher')
    parser.add_argument('--taskwise_kd', default=False, action='store_true',
                        help='Distill logits with respect to each task separately')
    parser.add_argument('--separate_softmax', default=False, action='store_true',
                        help='Apply CE loss to past and current samples separately')
    parser.add_argument('--csv_filename', default='', type=str,
                        help='Filename of the csv file to log the final numbers')
    parser.add_argument('--batch_mixup', default=False, action='store_true',
                        help='Do mixup within a batch of samples')
    parser.add_argument('--batch_logitnoise', default=False, action='store_true',
                        help='Add noise to logits within batch')
    parser.add_argument('--test_run', default=False, action='store_true',
                        help='Flag to switch off plotting and saving extra metrics')
    parser.add_argument('--global_index', default=0, type=int,
                        help='Global index for ema teacher.')
    parser.add_argument('--finetune_weight', default=1., type=float,
                        help='Weights for finetuning with representations')
    parser.add_argument('--random_label_corruption', default=0.5, type=float,
                        help='Percentage of labels to corrupt while learning on new task')

    # save checkpoints after every task
    parser.add_argument('--task_checkpoints', default=False, action='store_true',
                        help='Save checkpoints of the working model after every task')
    parser.add_argument('--teacher_task_checkpoints', default=False, action='store_true',
                        help='Save checkpoints of the teacher model after every task and calculate heatmap')
    return parser


def main(args):
    print(args)
    logger = Logger(list_subsets=['train', 'test'])

    use_distillation = args.auto_kd
    device = torch.device(args.device)

    # log accuracies and losses
    now = datetime.datetime.now()
    sub_folder = utils.fetch_fname(args)
    if args.tensorboard:
        name_parts = [args.options[0].split("/")[-1].split(".")[0], args.name,
                      'buf_' + str(args.memory_size), sub_folder, now.strftime("%Y%m%d_%H%M%S_%f")]
        model_stash = {
            'task_idx': 0,
            'epoch_idx': 0,
            'batch_idx': 0,
            'model_name': '/'.join(name_parts),
            'mean_accs': [],
            'args': args
        }
        tb_logger = utils.TensorboardLogger(args, 'class-il', model_stash)
        model_stash['tensorboard_name'] = tb_logger.get_name()

    # set new path for log dir
    args.log_dir = tb_logger.loggers['class-il'].log_dir

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    # cudnn.benchmark = True

    scenario_train, args.nb_classes = build_dataset(is_train=True, args=args)
    scenario_val, _ = build_dataset(is_train=False, args=args)

    log_lr = args.incremental_lr

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None

    model = factory.get_backbone(args)
    model.head = Classifier(
        model.embed_dim, args.nb_classes, args.initial_increment,
        args.increment, len(scenario_train) # len(scenario_train) = nb_tasks = 10
    )
    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)

    # Start the logging process on disk ----------------------------------------
    if args.name:
        log_path = os.path.join(args.log_dir, f"logs_{args.trial_id}.json")
        long_log_path = os.path.join(args.log_dir, f"long_logs_{args.trial_id}.json")

        if utils.is_main_process():
            os.system("echo '\ek{}\e\\'".format(args.name))
            os.makedirs(args.log_dir, exist_ok=True)
            with open(os.path.join(args.log_dir, f"config_{args.trial_id}.json"), 'w+') as f:
                config = vars(args)
                config["nb_parameters"] = n_parameters
                json.dump(config, f, indent=2)
            with open(log_path, 'w+') as f:
                pass  # touch
            with open(long_log_path, 'w+') as f:
                pass  # touch
        log_store = {'results': {}}

        args.output_dir = os.path.join(args.output_basedir, f"{datetime.datetime.now().strftime('%y-%m-%d')}_{args.name}_{args.trial_id}")
    else:
        log_store = None
        log_path = long_log_path = None
    if args.output_dir and utils.is_main_process():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.distributed:
        torch.distributed.barrier()

    print('number of params:', n_parameters)

    loss_scaler = scaler.ContinualScaler(args.no_amp)

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0. or args.cutmix > 0.:
        criterion = SoftTargetCrossEntropy()
    elif args.bce_loss:
        criterion = bce_with_logits
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None

    output_dir = Path(args.output_dir)

    memory = None
    if args.memory_size > 0:
        memory = Memory(
            args.memory_size, scenario_train.nb_classes, args.rehearsal, args.representation_replay, args.fixed_memory
        )

    nb_classes = args.initial_increment # 10
    base_lr = args.lr # 5e-4
    accuracy_list = []
    accuracy_list5 = []
    start_time = time.time()

    if args.debug:
        args.base_epochs = 1
        args.epochs = 1

    args.increment_per_task = [args.initial_increment] + [args.increment for _ in range(len(scenario_train) - 1)]

    # --------------------------------------------------------------------------
    #
    # Begin of the task loop
    #
    # --------------------------------------------------------------------------
    dataset_true_val = None

    task_wise_acc = np.zeros((scenario_val.nb_tasks, scenario_val.nb_tasks), dtype=np.float32)
    teacher_accs = np.zeros((scenario_val.nb_tasks, scenario_val.nb_tasks), dtype=np.float32)

    args.incremental_lr = args.lr

    for task_id, dataset_train in enumerate(scenario_train):
        if args.max_task == task_id:
            print(f"Stop training because of max task")
            break
        print(f"Starting task id {task_id}/{len(scenario_train) - 1}")

        # ----------------------------------------------------------------------
        # Data
        dataset_val = scenario_val[:task_id + 1]
        if args.validation > 0.:  # use validation split instead of test
            if task_id == 0:
                dataset_train, dataset_val = split_train_val(dataset_train, args.validation)
                dataset_true_val = dataset_val
            else:
                dataset_train, dataset_val = split_train_val(dataset_train, args.validation)
                dataset_true_val.concat(dataset_val)
            dataset_val = dataset_true_val

        if not args.data_set.lower() == 'rotmnist':
            for i in range(3):  # Quick check to ensure same preprocessing between train/test
                assert abs(dataset_train.trsf.transforms[-1].mean[i] - dataset_val.trsf.transforms[-1].mean[i]) < 0.0001
                assert abs(dataset_train.trsf.transforms[-1].std[i] - dataset_val.trsf.transforms[-1].std[i]) < 0.0001

        loader_memory = None
        loader_buffer = None
        if task_id > 0 and memory is not None: # done from second task onward
            dataset_memory = memory.get_dataset(dataset_train)
            loader_buffer = factory.get_train_loaders(
                dataset_memory, args,
                args.replay_memory if args.replay_memory > 0 else args.batch_size, drop_last=False
            )
            loader_memory = factory.InfiniteLoader(loader_buffer)
            if not args.sep_memory:
                previous_size = len(dataset_train)
                dataset_train.add_samples(*memory.get())
                print(f"{len(dataset_train) - previous_size} samples added from memory.")

            if args.only_ft: # false
                dataset_train = get_finetuning_dataset(dataset_train, memory, 'balanced')
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # Ensembling
        if args.dytox:
            model_without_ddp = factory.update_birt(model_without_ddp, task_id, args)
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # Adding new parameters to handle the new classes
        print("Adding new parameters")
        if task_id > 0 and not args.dytox:
            model_without_ddp.head.add_classes()

        if task_id > 0: # note this! They are freezing old task tokens and old heads
            model_without_ddp.freeze(args.freeze_task)

        # Freezing first layer of the current model for representation replay
        if task_id > 0 and args.representation_replay:
            model_without_ddp.freeze(["partial_sab"])
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # Data
        loader_train, loader_val = factory.get_loaders(dataset_train, dataset_val, args)
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # Learning rate and optimizer
        if task_id > 0 and args.incremental_batch_size:
            args.batch_size = args.incremental_batch_size

        if args.incremental_lr is not None and task_id > 0:
            linear_scaled_lr = args.incremental_lr * args.batch_size * utils.get_world_size() / 512.0
        else:
            linear_scaled_lr = base_lr * args.batch_size * utils.get_world_size() / 512.0

        args.lr = linear_scaled_lr
        optimizer = create_optimizer(args, model_without_ddp)
        lr_scheduler, _ = create_scheduler(args, optimizer)
        # ----------------------------------------------------------------------

        if mixup_active:
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing,
                num_classes=nb_classes,
                loader_memory=loader_memory
            )

        skipped_task = False
        initial_epoch = epoch = 0
        if args.resume and args.start_task > task_id:
            utils.load_first_task_model(model_without_ddp, loss_scaler, task_id, args)
            print("Skipping first task")
            epochs = 0
            train_stats = {"task_skipped": str(task_id)}
            skipped_task = True
        elif args.base_epochs is not None and task_id == 0:
            epochs = args.base_epochs
        else:
            epochs = args.epochs

        if args.distributed:
            del model
            model = torch.nn.parallel.DistributedDataParallel(model_without_ddp, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model = model_without_ddp

        model_without_ddp.nb_epochs = epochs # 500
        model_without_ddp.nb_batch_per_epoch = len(loader_train) # 39

        # Init SAM, for DyTox++ (see appendix) ---------------------------------
        sam = None
        # This loop does not happen
        if args.sam_rho > 0. and 'tr' in args.sam_mode and ((task_id > 0 and args.sam_skip_first) or not args.sam_skip_first):
            if args.sam_final is not None:
                sam_step = (args.sam_final - args.sam_rho) / scenario_train.nb_tasks
                sam_rho = args.sam_rho + task_id * sam_step
            else:
                sam_rho = args.sam_rho

            print(f'Initialize SAM with rho={sam_rho}')
            sam = SAM(
                optimizer, model_without_ddp,
                rho=sam_rho, adaptive=args.sam_adaptive,
                div=args.sam_div,
                use_look_sam=args.look_sam_k > 0, look_sam_alpha=args.look_sam_alpha
            )
        # ----------------------------------------------------------------------

        print(f"Start training for {epochs-initial_epoch} epochs")
        max_accuracy = 0.0
        for epoch in range(initial_epoch, epochs):
            if args.distributed:
                loader_train.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                model, criterion, loader_train,
                optimizer, device, epoch, task_id, loss_scaler,
                args.clip_grad, mixup_fn,
                debug=args.debug,
                args=args,
                teacher_model=teacher_model,
                model_without_ddp=model_without_ddp,
                sam=sam,
                loader_memory=loader_memory,
                loader_buffer=loader_buffer
            )

            lr_scheduler.step(epoch)

            # log values for tensorboard
            if args.tensorboard:
                tb_logger.log_lr(train_stats[1]['lr'], args.base_epochs, epoch, task_id)
                tb_logger.log_loss(train_stats[1]['loss'], args.base_epochs, epoch, task_id)
                tb_logger.log_div_loss(train_stats[1]['div'], args.base_epochs, epoch, task_id)
                tb_logger.log_kd_loss(train_stats[1]['kd'], args.base_epochs, epoch, task_id)
                tb_logger.log_attn_loss(train_stats[1]['att'], args.base_epochs, epoch, task_id)
                tb_logger.log_cos_loss(train_stats[1]['cos'], args.base_epochs, epoch, task_id)

                model_stash['epoch_idx'] = epoch + 1


            if args.save_every_epoch is not None and epoch % args.save_every_epoch == 0:
                if os.path.isdir(args.resume):
                    with open(os.path.join(args.resume, 'save_log.txt'), 'w+') as f:
                        f.write(f'task={task_id}, epoch={epoch}\n')

                    checkpoint_paths = [os.path.join(args.resume, f'checkpoint_{task_id}.pth')]
                    for checkpoint_path in checkpoint_paths:
                        if (task_id < args.start_task and args.start_task > 0) and os.path.isdir(args.resume) and os.path.exists(checkpoint_path):
                            continue

                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'task_id': task_id,
                            'scaler': loss_scaler.state_dict(),
                            'args': args,
                        }, checkpoint_path)

            if args.eval_every and (epoch % args.eval_every == 0 or (args.finetuning and epoch == epochs - 1)):
                eval_and_log(
                    args, output_dir, model, model_without_ddp, optimizer, lr_scheduler,
                    epoch, task_id, loss_scaler, max_accuracy,
                    [], n_parameters, device, loader_val, train_stats[0], None, long_log_path,
                    logger, model_without_ddp.epoch_log(), acc5=[]
                )
                logger.end_epoch()

        if memory is not None:
            task_memory_path = os.path.join(args.resume, f'memory_{task_id}.npz')
            if os.path.isdir(args.resume) and os.path.exists(task_memory_path):
                # Resuming this task step, thus reloading saved memory samples
                # without needing to re-compute them
                memory.load(task_memory_path)
            else:
                memory.add(scenario_train[task_id], model, args.initial_increment if task_id == 0 else args.increment)

                if args.resume != '':
                    memory.save(task_memory_path)
                else:
                    memory.save(os.path.join(args.output_dir, f'memory_{task_id}.npz'))

            assert len(memory) <= args.memory_size

        # ----------------------------------------------------------------------
        # FINETUNING
        # ----------------------------------------------------------------------

        # Init SAM, for DyTox++ (see appendix) ---------------------------------
        sam = None
        if args.sam_rho > 0. and 'ft' in args.sam_mode and ((task_id > 0 and args.sam_skip_first) or not args.sam_skip_first):
            if args.sam_final is not None:
                sam_step = (args.sam_final - args.sam_rho) / scenario_train.nb_tasks
                sam_rho = args.sam_rho + task_id * sam_step
            else:
                sam_rho = args.sam_rho

            print(f'Initialize SAM with rho={sam_rho}')
            sam = SAM(
                optimizer, model_without_ddp,
                rho=sam_rho, adaptive=args.sam_adaptive,
                div=args.sam_div,
                use_look_sam=args.look_sam_k > 0, look_sam_alpha=args.look_sam_alpha
            )
        # ----------------------------------------------------------------------

        if args.finetuning and memory and (task_id > 0 or scenario_train.nb_classes == args.initial_increment) and \
                not skipped_task and args.log_category != 'joint':
            dataset_finetune = get_finetuning_dataset(dataset_train, memory, args.finetuning, args.representation_replay)
            print(
                f'Finetuning phase of type {args.finetuning} with {len(dataset_finetune)} samples.'
            )

            dataset_finetune, loader_val = factory.get_loaders(dataset_finetune, dataset_val, args, drop_last=False)

            if args.finetuning_resetclf:
                model_without_ddp.reset_classifier()

            model_without_ddp.freeze(args.freeze_ft)

            if args.distributed:
                del model
                model = torch.nn.parallel.DistributedDataParallel(model_without_ddp, device_ids=[args.gpu],
                                                                  find_unused_parameters=True)
            else:
                model = model_without_ddp

            model_without_ddp.begin_finetuning()

            args.lr = args.finetuning_lr * args.batch_size * utils.get_world_size() / 512.0
            optimizer = create_optimizer(args, model_without_ddp)
            for epoch in range(args.finetuning_epochs):
                if args.distributed:
                    dataset_finetune.sampler.set_epoch(epoch)

                train_stats = train_one_epoch(
                    model, criterion, dataset_finetune,
                    optimizer, device, epoch, task_id, loss_scaler,
                    args.clip_grad, mixup_fn,
                    debug=args.debug,
                    args=args,
                    teacher_model=None,
                    model_without_ddp=model_without_ddp,
                    loader_buffer=None,
                    finetuning=True,
                )

                if epoch % 10 == 0 or epoch == args.finetuning_epochs - 1:
                    eval_and_log(
                        args, output_dir, model, model_without_ddp, optimizer, lr_scheduler,
                        epoch, task_id, loss_scaler, max_accuracy,
                        [], n_parameters, device, loader_val, train_stats[0], None, long_log_path,
                        logger, model_without_ddp.epoch_log(), acc5=[]
                    )
                    logger.end_epoch()

            model_without_ddp.end_finetuning()

        eval_and_log(
            args, output_dir, model, model_without_ddp, optimizer, lr_scheduler,
            epoch, task_id, loss_scaler, max_accuracy,
            accuracy_list, n_parameters, device, loader_val, train_stats[0], log_store, log_path,
            logger, model_without_ddp.epoch_log(), skipped_task, acc5=accuracy_list5
        )

        # ----------------------------------------------------------------------
        # Initializing teacher model from previous task
        if use_distillation:
            if not teacher_model:
                teacher_model = copy.deepcopy(model_without_ddp)
            elif teacher_model and args.ema_alpha > 0.:
                old_teacher_model = copy.deepcopy(teacher_model)
                teacher_model = copy.deepcopy(model_without_ddp)
                sd = teacher_model.state_dict()
                for name, param in old_teacher_model.named_parameters():
                    sd[name] = args.ema_alpha * param.data + \
                               (1 - args.ema_alpha) * model_without_ddp.state_dict()[name].data
                teacher_model.load_state_dict(sd)
            else:
                teacher_model = copy.deepcopy(model_without_ddp)
            teacher_model.freeze(['all'])
            teacher_model.eval()
        # ----------------------------------------------------------------------

        # Save checkpoints after every task
        if args.task_checkpoints:
            task_checkpoint_path = os.path.join(tb_logger.loggers['class-il'].log_dir,
                                                'task{}_checkpoint.pth'.format(task_id))
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'args': args,
            }, task_checkpoint_path)

        if args.teacher_task_checkpoints:
            task_checkpoint_path = os.path.join(tb_logger.loggers['class-il'].log_dir,
                                                'teacher_task{}_checkpoint.pth'.format(task_id))
            utils.save_on_master({
                'model': teacher_model.state_dict(),
                'args': args,
            }, task_checkpoint_path)

        # Fill in accuracies for task wise validation set
        for cur_task_id in range(task_id+1):
            cur_loader_val = factory.get_train_loaders(scenario_val[cur_task_id], args)
            cur_teacher_acc = evaluate_teacher(cur_loader_val, teacher_model, device, args)["acc1"]
            teacher_accs[task_id, cur_task_id] = cur_teacher_acc
            cur_acc = evaluate_teacher(cur_loader_val, model, device, args)["acc1"]
            task_wise_acc[task_id, cur_task_id] = cur_acc

        logger.end_task()

        nb_classes += args.increment

        # reset tensorboard fields
        if args.tensorboard:
            model_stash['task_idx'] = task_id + 1
            model_stash['epoch_idx'] = 0

            model_stash['mean_accs'].append(accuracy_list[-1])
            tb_logger.log_accuracy(accuracy_list, accuracy_list[-1], args, task_id)
            tb_logger.log_accuracy5(accuracy_list5, accuracy_list5[-1], args, task_id)

    # save task wise accuracy in the text and plot format
    task_perf_path = os.path.join(tb_logger.loggers['class-il'].log_dir, 'task_performance.txt')
    if not args.test_run:
        np.savetxt(task_perf_path, task_wise_acc, fmt='%.2f')

    # save task wise accuracy in the text and plot format for teacher model
    teacher_task_perf_path = os.path.join(tb_logger.loggers['class-il'].log_dir, 'teacher_task_performance.txt')
    if not args.test_run:
        np.savetxt(teacher_task_perf_path, teacher_accs, fmt='%.2f')

    if args.log_category != 'joint' and not args.test_run:
        utils.plot_task_performance(task_perf_path=task_perf_path, num_tasks=scenario_val.nb_tasks,
                                tb_logger=tb_logger, task_wise_acc=task_wise_acc)
        utils.plot_task_performance(task_perf_path=teacher_task_perf_path, num_tasks=scenario_val.nb_tasks,
                                tb_logger=tb_logger, task_wise_acc=teacher_accs, teacher_plot=True)

    # finish logging the tensorboard
    if args.tensorboard:
        tb_logger.close()

    final_checkpoint_path = os.path.join(tb_logger.loggers['class-il'].log_dir, 'final_checkpoint.pth')
    if not args.test_run:
        utils.save_on_master({
            'model': model_without_ddp.state_dict(),
            'args': args,
        }, final_checkpoint_path)

    teacher_checkpoint_path = os.path.join(tb_logger.loggers['class-il'].log_dir, 'teacher_checkpoint.pth')
    if not args.test_run:
        utils.save_on_master({
            'model': teacher_model.state_dict(),
            'args': args,
        }, teacher_checkpoint_path)

    task_accuracy_list = [np.mean(task_wise_acc[i,:i+1]) for i in range(scenario_train.nb_tasks)]
    teacher_accuracy_list = [np.mean(teacher_accs[i,:i+1]) for i in range(scenario_train.nb_tasks)]

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print(f'Setting {args.data_set} with {args.initial_increment}-{args.increment}')
    print(f"All accuracies: {task_accuracy_list}")
    print(f"Average Incremental Accuracy: {statistics.mean(task_accuracy_list)}")

    # log results in a csv file
    col_names = ['dataset', 'model', 'seed', 'attn_weight', 'attn_version', 'no_distillation', 'distill_version',
                 'distill_weight', 'distill_weight_buffer', 'ema_alpha', 'sep_memory', 'cos_weight', 'dynamic_tokens', 'representation_replay',
                 'replay_from', 'buffer_size']
    acc_col_names = [f"acc{i + 1}" for i in reversed(range(scenario_val.nb_tasks))]
    teacher_acc_col_names = [f"t_acc{i + 1}" for i in reversed(range(scenario_val.nb_tasks))]
    col_names = col_names + acc_col_names + ['mean_acc', 'lr'] + teacher_acc_col_names + \
                ['mixup_alpha', 'cutmix_alpha', 'gaussian_noise', 'tensor_noise', 'shotnoise', 'ext_lambda',
                 'mixup_prob', 'cutmix_prob', 'shot_prob', 'tensor_prob', 'repnoise_prob', 'scale_prob', 'ema_prob',
                 'label_noise_prob', 'flr', 'logit_noise', 'finetune_weight', 'machine', 'epochs']

    arg_list = [args.data_set, args.model, args.seed, args.attn_weight, args.attn_version, args.no_distillation,
                args.distill_version, args.distill_weight, args.distill_weight_buffer, args.ema_alpha, args.sep_memory,
                args.cos_weight, args.dynamic_tokens, args.representation_replay, args.replay_from, args.memory_size]

    if args.csv_filename:
        csv_file = os.path.join(args.output_basedir, args.csv_filename)
    else:
        csv_file = os.path.join(args.output_basedir, "Other_results.csv")

    # from the last log, read other metrics
    with open(log_path) as f:
        last_results = json.loads(f.readlines()[-1])
    col_names += list(last_results.keys())

    if not os.path.exists(csv_file) and utils.is_main_process():
        with open(csv_file, 'a') as ff:
            wr = csv.writer(ff, quoting=csv.QUOTE_ALL)
            wr.writerow(col_names)

    representation_aug = ['a'+str(args.rep_mixup_alpha), 'cm'+str(args.rep_cutmix_alpha),
                          'g'+str(args.rep_noise_weight), 't'+str(args.tensor_noise_weight),
                          's'+str(args.shotnoise_strength), 'ex'+str(args.ext_lambda), 'mp'+str(args.mixup_prob),
                          'cmp'+str(args.cutmix_prob), 'sp'+str(args.shot_prob), 'tp'+str(args.tensor_prob),
                          'rp'+str(args.repnoise_prob), 'scalep'+str(args.scale_prob)]

    if utils.is_main_process():
        with open(csv_file, 'a') as ff:
            wr = csv.writer(ff, quoting=csv.QUOTE_ALL)
            wr.writerow(arg_list + task_accuracy_list[::-1] + [np.mean(task_accuracy_list)]+[log_lr] +
                        teacher_accuracy_list[::-1] + representation_aug +
                        ["ef"+str(args.ema_frequency)]+[str(args.random_label_corruption)]+["flr"+str(args.finetuning_lr)] +
                        ["logn"+str(args.logit_noise_dev), str(args.finetune_weight), os.uname()[1], str(args.epochs)] +
                        list(last_results.values()))

    if args.name:
        print(f"Experiment name: {args.name}")
        log_store['summary'] = {"avg": str(statistics.mean(task_accuracy_list))}
        if log_path is not None and utils.is_main_process():
            with open(log_path, 'a+') as f:
                f.write(json.dumps(log_store['summary']) + '\n')


def load_options(args, options):
    varargs = vars(args)

    name = []
    for o in options:
        with open(o) as f:
            new_opts = yaml.safe_load(f)

        for k, v in new_opts.items():
            if k not in varargs:
                raise ValueError(f'Option {k}={v} doesnt exist!')
        varargs.update(new_opts)
        name.append(o.split("/")[-1].replace('.yaml', ''))

    return '_'.join(name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BiRT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    utils.init_distributed_mode(args)

    if args.options:
        name = load_options(args, args.options)
        if not args.name:
            args.name = name

    args.log_dir = os.path.join(
        args.data_path, args.data_set.lower(), args.log_category,
        datetime.datetime.now().strftime('%y-%m'),
        f"week-{int(datetime.datetime.now().strftime('%d')) // 7 + 1}",
        f"{int(datetime.datetime.now().strftime('%d'))}_{args.name}"
    )

    if isinstance(args.class_order, list) and isinstance(args.class_order[0], list):
        print(f'Running {len(args.class_order)} different class orders.')
        class_orders = copy.deepcopy(args.class_order)

        for i, order in enumerate(class_orders, start=1):
            print(f'Running class ordering {i}/{len(class_orders)}.')
            args.trial_id = i
            args.class_order = order
            main(args)
    else:
        args.trial_id = 1
        main(args)
