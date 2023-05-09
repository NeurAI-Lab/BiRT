# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import json
import os
import math
from typing import Iterable, Optional
import pickle
import copy

import torch
from timm.data import Mixup
from timm.utils import accuracy
from timm.loss import SoftTargetCrossEntropy
from torch.nn import functional as F
import torch.nn as nn

import continual.utils as utils
from continual.losses import DistillationLoss


CE = SoftTargetCrossEntropy()


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, task_id: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, debug=False, args=None,
                    teacher_model: torch.nn.Module = None,
                    model_without_ddp: torch.nn.Module = None,
                    sam: torch.optim.Optimizer = None,
                    loader_memory=None, loader_buffer=None, finetuning=False):
    """Code is a bit ugly to handle SAM, sorry! :upside_down_face:"""
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Task: [{}] Epoch: [{}]'.format(task_id, epoch)
    print_freq = 10

    for batch_index, (samples, targets, batch_tasks) in enumerate(metric_logger.log_every(data_loader, print_freq,
                                                                                          header)):
        if batch_index == 0:
            print(f'Image size is {samples.shape}.')

        args.global_index += 1

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True) # in second task, contains 0,1,2,3
        optimizer.zero_grad()

        lam = None
        if mixup_fn is not None:
            samples, targets, lam = mixup_fn(samples, targets)

        if sam is not None and (args.sam_first == 'memory' and task_id > 0):
            # If you want to do the first step of SAM only on memory samples.
            x, y, _ = loader_memory.get()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=not args.no_amp):
                loss_tuple = forward(x, y, model, teacher_model, criterion, lam, args)
        else:
            with torch.cuda.amp.autocast(enabled=not args.no_amp):
                loss_tuple = forward(samples, targets, model, teacher_model, criterion, lam, args,
                                     batch_tasks=batch_tasks, epoch=epoch, finetuning=finetuning)

        loss = sum(filter(lambda x: x is not None, loss_tuple))
        internal_losses = model_without_ddp.get_internal_losses(loss) # empty for now, can be used for EWC
        for internal_loss_value in internal_losses.values():
            loss += internal_loss_value

        check_loss(loss)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        if sam is not None and args.look_sam_k > 0:
            # Look-sam only apply the costly sam estimation every k step.
            look_sam_update = False
            if batch_index % args.look_sam_k == 0:
                loss_scaler.pre_step(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
                loss_scaler.update()
                sam.first_step()  # modify weights to worse neighbor
                optimizer.zero_grad()

                look_sam_update = True

                with torch.cuda.amp.autocast(enabled=not args.no_amp):
                    loss_tuple = forward(samples, targets, model, teacher_model, criterion, lam, args)
                loss = sum(filter(lambda x: x is not None, loss_tuple))
                internal_losses = model_without_ddp.get_internal_losses(loss)
                for internal_loss_value in internal_losses.values():
                    loss += internal_loss_value

            check_loss(loss)
            loss_scaler.pre_step(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
            sam.second_step(look_sam_update=look_sam_update)
            loss_scaler.post_step(optimizer, model_without_ddp)
        elif sam is not None:
            loss_scaler.pre_step(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
            loss_scaler.update()
            sam.first_step()  # modify weights to worse neighbor
            optimizer.zero_grad()

            if args.sam_second == 'memory' and task_id > 0:
                x, y, _ = loader_memory.get()
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=not args.no_amp):
                    loss_tuple = forward(x, y, model, teacher_model, criterion, lam, args)
            else:
                with torch.cuda.amp.autocast(enabled=not args.no_amp):
                    loss_tuple = forward(samples, targets, model, teacher_model, criterion, lam, args)

            loss = sum(filter(lambda x: x is not None, loss_tuple))
            internal_losses = model_without_ddp.get_internal_losses(loss)
            for internal_loss_value in internal_losses.values():
                loss += internal_loss_value

            check_loss(loss)
            loss_scaler.pre_step(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
            sam.second_step()
            loss_scaler.post_step(optimizer, model_without_ddp)
        else:
            loss_scaler(loss, optimizer, model_without_ddp, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update_dict(internal_losses)
        metric_logger.update(loss=loss_tuple[0])
        metric_logger.update(kd=loss_tuple[1])
        metric_logger.update(div=loss_tuple[2])
        metric_logger.update(att=loss_tuple[3])
        metric_logger.update(cos=loss_tuple[4])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if debug:
            print('Debug, only doing one epoch!')
            break

        # ----------------------------------------------------------------------
        # Updating teacher model according to EMA frequency
        if torch.rand(1) < args.ema_frequency and not finetuning:
            if args.auto_kd and task_id > 0 and teacher_model:
                alpha = min(1 - 1 / (args.global_index + 1), args.ema_alpha)
                # print(f"=========Updating EMA of Teacher model at iter {batch_index} with alpha {alpha}.========")
                old_teacher_model = copy.deepcopy(teacher_model)
                teacher_model = pickle.loads(pickle.dumps(model_without_ddp))
                sd = teacher_model.state_dict()
                for name, param in old_teacher_model.named_parameters():
                    sd[name] = alpha * param.data + \
                               (1 - alpha) * model_without_ddp.state_dict()[name].data
                teacher_model.load_state_dict(sd)
                teacher_model.freeze(['all'])
                teacher_model.eval()
        # ----------------------------------------------------------------------

    if (args.sep_memory and loader_buffer is not None) or (finetuning and loader_buffer is not None):
        print("Iterating for buffered samples, finetuning: {}".format(finetuning))
        for batch_index, (samples, targets, batch_tasks) in enumerate(metric_logger.log_every(loader_buffer,
                                                                                              print_freq, header)):

            args.global_index += 1

            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True) # in second task, contains 0,1,2,3
            optimizer.zero_grad()

            # randomly corrupt certain labels with noise
            if args.random_label_corruption > 0:
                targets = utils.get_random_labels(targets, task_id * args.increment,
                                                  task_id * args.increment + args.increment,
                                                  args.random_label_corruption)

            lam = None
            with torch.cuda.amp.autocast(enabled=not args.no_amp):
                if args.multi_token_setup:
                    loss_tuple = forward(samples, targets, model, teacher_model, criterion, lam, args,
                                     buffer_iteration=True, batch_tasks=batch_tasks)
                else:
                    loss_tuple = forward(samples, targets, model, teacher_model, criterion, lam, args,
                                         buffer_iteration=True, finetuning=finetuning, epoch=epoch)
            loss = sum(filter(lambda x: x is not None, loss_tuple))
            internal_losses = model_without_ddp.get_internal_losses(loss) # empty for now, can be used for EWC
            for internal_loss_value in internal_losses.values():
                loss += internal_loss_value

            check_loss(loss)

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

            loss_scaler(loss, optimizer, model_without_ddp, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)

            torch.cuda.synchronize()

            metric_logger.update_dict(internal_losses)
            metric_logger.update(loss=loss_tuple[0])
            metric_logger.update(kd=loss_tuple[1])
            metric_logger.update(div=loss_tuple[2])
            metric_logger.update(att=loss_tuple[3])
            metric_logger.update(cos=loss_tuple[4])
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            # ----------------------------------------------------------------------
            # Updating teacher model according to EMA frequency
            if torch.rand(1) < args.ema_frequency and not finetuning:
                if args.auto_kd and task_id > 0 and teacher_model:
                    alpha = min(1 - 1 / (args.global_index + 1), args.ema_alpha)
                    # print(f"=========Updating EMA of Teacher model at iter {batch_index} with alpha {alpha}.========")
                    old_teacher_model = copy.deepcopy(teacher_model)
                    teacher_model = pickle.loads(pickle.dumps(model_without_ddp))
                    sd = teacher_model.state_dict()
                    for name, param in old_teacher_model.named_parameters():
                        sd[name] = alpha * param.data + \
                                   (1 - alpha) * model_without_ddp.state_dict()[name].data
                    teacher_model.load_state_dict(sd)
                    teacher_model.freeze(['all'])
                    teacher_model.eval()
            # ----------------------------------------------------------------------

    if hasattr(model_without_ddp, 'hook_after_epoch'):
        model_without_ddp.hook_after_epoch() # pass, nothing happens

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    track_keys = ['lr', 'loss', 'kd', 'div', 'att', 'cos']
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, \
           {k: metric_logger.meters[k].avg for k in track_keys}


def check_loss(loss):
    if not math.isfinite(loss.item()):
        raise Exception('Loss is {}, stopping training'.format(loss.item()))


def forward(samples, targets, model, teacher_model, criterion, lam, args, buffer_iteration=False, batch_tasks=None,
            finetuning=False, epoch=None):
    main_output, div_output = None, None

    representation_tensors = None
    mixup_indexes = None
    # Representation replay for buffered samples from second task onward
    if teacher_model is not None and args.representation_replay and (buffer_iteration or finetuning):
        if args.distributed:
            representation_tensors = samples

            rep_lam = None
            mixup_indexes = None
            if args.rep_cutmix_alpha > 0.:
                representation_tensors, y_a, y_b, rep_lam, mixup_indexes = utils.cutmix_batch_representations(
                    representation_tensors,
                    targets, args.rep_cutmix_alpha,
                    aug_prob=args.cutmix_prob)
            if args.rep_mixup_alpha > 0. and args.batch_mixup:
                representation_tensors, y_a, y_b, rep_lam, mixup_indexes = utils.mixup_batch_representations(
                    representation_tensors,
                    targets, args.rep_mixup_alpha,
                    aug_prob=args.mixup_prob)
            if args.rep_mixup_alpha > 0. and torch.rand(1) < args.mixup_prob and not args.batch_mixup:
                representation_tensors, y_a, y_b, rep_lam = utils.mixup_representations(representation_tensors, targets,
                                                                                        args.rep_mixup_alpha)
            if args.ext_lambda > 0.:
                freq_classes = torch.topk(torch.bincount(targets), 5)[1]
                for c in freq_classes:
                    indices = torch.where(targets == c)[0]
                    if len(indices) > 1:
                        representation_tensors[indices[0]] = representation_tensors[indices[0]] + args.ext_lambda * \
                                                             (representation_tensors[indices[0]] -
                                                              representation_tensors[indices[-1]])

            rep_shape = representation_tensors.shape
            representation_tensors = torch.permute(representation_tensors.view((rep_shape[0], rep_shape[1], -1)),
                                                   (0, 2, 1))
            outputs = model(representation_tensors, latter=True, args=args)
        else:
            representation_tensors = samples

            rep_lam = None
            mixup_indexes = None
            if args.rep_cutmix_alpha > 0.:
                representation_tensors, y_a, y_b, rep_lam, mixup_indexes = utils.cutmix_batch_representations(representation_tensors,
                                                                                       targets, args.rep_cutmix_alpha,
                                                                                       aug_prob=args.cutmix_prob)
            if args.rep_mixup_alpha > 0. and args.batch_mixup:
                representation_tensors, y_a, y_b, rep_lam, mixup_indexes = utils.mixup_batch_representations(representation_tensors,
                                                                                       targets, args.rep_mixup_alpha,
                                                                                       aug_prob=args.mixup_prob)
            if args.rep_mixup_alpha > 0. and torch.rand(1) < args.mixup_prob and not args.batch_mixup:
                representation_tensors, y_a, y_b, rep_lam = utils.mixup_representations(representation_tensors, targets,
                                                                              args.rep_mixup_alpha)

            if args.ext_lambda > 0.:
                freq_classes = torch.topk(torch.bincount(targets), 5)[1]
                for c in freq_classes:
                    indices = torch.where(targets == c)[0]
                    if len(indices) > 1:
                        representation_tensors[indices[0]] = representation_tensors[indices[0]] + args.ext_lambda * \
                                                             (representation_tensors[indices[0]] -
                                                              representation_tensors[indices[-1]])

            rep_shape = representation_tensors.shape
            representation_tensors = torch.permute(representation_tensors.view((rep_shape[0], rep_shape[1], -1)), (0,2,1))
            outputs = model.forward_latter(representation_tensors, args=args)
    elif args.representation_replay and finetuning:
        rep_shape = samples.shape
        representation_tensors = torch.permute(samples.view((rep_shape[0], rep_shape[1], -1)), (0, 2, 1))
        if args.distributed:
            outputs = model(representation_tensors, latter=True, args=args)
        else:
            outputs = model.forward_latter(representation_tensors)
    else:
        outputs = model(samples, batch_tasks=batch_tasks) # 128, 10

    if isinstance(outputs, dict):
        main_output = outputs['logits']
        div_output = outputs['div']
        main_attention = outputs['attention']
    else:
        main_output = outputs

    if teacher_model is not None and args.representation_replay and (buffer_iteration or finetuning) and rep_lam:
        loss = rep_lam * criterion(main_output, y_a) + (1-rep_lam) * criterion(main_output, y_b)
    else:
        if args.separate_softmax:
            if not buffer_iteration:
                loss = F.cross_entropy(main_output[:, -args.increment:], targets%args.increment)
            else:
                loss = F.cross_entropy(main_output[:, :-args.increment], targets)
        elif finetuning and args.representation_replay:
            loss = args.finetune_weight * criterion(main_output, targets)
        else:
            loss = criterion(main_output, targets) # bce_with_logits

    attn_loss = torch.tensor(0., device=loss.device)
    if teacher_model is not None:
        with torch.no_grad():
            main_output_old = None
            if args.multi_token_setup and buffer_iteration:
                teacher_outputs = teacher_model(samples, batch_tasks)
            elif args.multi_token_setup and not buffer_iteration:
                teacher_model.eval()
                teacher_outputs = teacher_model(samples, batch_tasks)
                teacher_outputs['logits'] = teacher_outputs['logits'].reshape(teacher_outputs['logits'].shape[0],
                                                                              int(teacher_outputs['logits'].shape[1] /
                                                                                   args.increment), args.increment)
                teacher_outputs['logits'] = teacher_outputs['logits'].sum(dim=1)
                teacher_model.train()
            else:
                if representation_tensors is not None:
                    teacher_outputs = teacher_model.forward_latter(representation_tensors.squeeze())
                else:
                    teacher_outputs = teacher_model(samples)

        if isinstance(outputs, dict):
            main_output_old = teacher_outputs['logits']
        else:
            main_output_old = teacher_outputs

        # attention loss
        teacher_attn = teacher_outputs['attention']
        if not args.representation_replay:
            attn_loss += args.attn_weight * F.mse_loss(main_attention[:teacher_attn.shape[0]], teacher_attn)

    kd_loss = None
    if teacher_model is not None and not args.no_distillation:
        logits_for_distil = main_output[:, :main_output_old.shape[1]]

        kd_loss = 0.
        if args.auto_kd:
            # Knowledge distillation on the probabilities
            # I called that 'auto_kd' because the right factor is automatically
            # computed, by interpolation between the main loss and the KD loss.
            # This is strongly inspired by WA (CVPR 2020) --> https://arxiv.org/abs/1911.07053
            lbd = main_output_old.shape[1] / main_output.shape[1]
            if args.data_set.lower() != 'rotmnist' and args.data_set.lower() != 'permnist':
                loss = (1 - lbd) * loss
            kd_factor = lbd

            tau = args.distillation_tau

            if args.taskwise_kd:
                single_head_size = main_output.shape[1]-main_output_old.shape[1]
                tasknum = int(main_output.shape[1] / single_head_size) - 1
                _kd_loss = torch.zeros(tasknum).cuda()
                for t in range(tasknum):
                    soft_target = F.softmax(main_output_old[:, t*10:(t+1)*10] / 2, dim=1) # make the num classes generic
                    output_log = F.log_softmax(main_output[:, t*10:(t+1)*10] / 2, dim=1)
                    _kd_loss[t] = F.kl_div(output_log, soft_target, reduction='batchmean') * (2 ** 2)
                _kd_loss = _kd_loss.sum()
            elif args.distill_version == 'kl' and not args.taskwise_kd:
                _kd_loss = F.kl_div(
                        F.log_softmax(logits_for_distil / tau, dim=1),
                        F.log_softmax(main_output_old / tau, dim=1),
                        reduction='mean',
                        log_target=True
                ) * (tau ** 2)
            elif args.distill_version == 'mse' and not args.taskwise_kd:
                _kd_loss = F.mse_loss(logits_for_distil, main_output_old)
            elif args.distill_version == 'linf' and not args.taskwise_kd:
                _kd_loss = torch.pairwise_distance(logits_for_distil, main_output_old, p=float('inf')).mean()
            elif args.distill_version == 'l2' and not args.taskwise_kd:
                if args.logit_noise_dev > 0. and args.batch_logitnoise and mixup_indexes is None:
                    batch_size = main_output_old.shape[0]
                    aug_indexes = torch.rand(batch_size) < 0.4
                    main_output_old[torch.where(aug_indexes)] += torch.normal(mean=0, std=args.logit_noise_dev,
                                                    size=main_output_old[torch.where(aug_indexes)].shape,
                                                    device='cuda')
                elif args.logit_noise_dev > 0. and args.batch_logitnoise and mixup_indexes is not None:
                    batch_size = main_output_old.shape[0]
                    indexes = torch.arange(batch_size, device='cuda')
                    clean_indexes = indexes[~mixup_indexes]
                    aug_indexes = clean_indexes[torch.rand(len(clean_indexes)) < 0.4]
                    main_output_old[aug_indexes] += torch.normal(mean=0, std=args.logit_noise_dev,
                                                    size=main_output_old[aug_indexes].shape,
                                                    device='cuda')
                elif args.logit_noise_dev > 0. and torch.rand(1) < 0.4 and not args.batch_logitnoise:
                    main_output_old += torch.normal(mean=0, std=args.logit_noise_dev,
                                                    size=main_output_old.shape,
                                                    device='cuda')
                _kd_loss = torch.pairwise_distance(logits_for_distil, main_output_old, p=2).mean()

            if representation_tensors is not None and args.distill_weight_buffer != 1.:
                kd_loss += args.distill_weight_buffer * kd_factor * _kd_loss
            else:
                kd_loss += args.distill_weight * kd_factor * _kd_loss

    div_loss = None
    if div_output is not None:
        # For the divergence heads, we need to create new targets.
        # If a class belong to old tasks, it will be 0.
        # If a class belong to the new task, it will be a class id between
        # 1 (not 0!) and 'nb_class_in_new_task'.
        # When doing that with mixup, some trickery is needed. (see if lam is not None).
        nb_classes = main_output.shape[1]
        nb_new_classes = div_output.shape[1] - 1
        nb_old_classes = nb_classes - nb_new_classes

        if lam is not None:  # 'lam' is the interpolation Lambda of mixup
            # If using mixup / cutmix
            div_targets = torch.zeros_like(div_output)
            nb_classes = main_output.shape[1]
            nb_new_classes = div_output.shape[1] - 1
            nb_old_classes = nb_classes - nb_new_classes

            div_targets[:, 0] = targets[:, :nb_old_classes].sum(-1)
            div_targets[:, 1:] = targets[:, nb_old_classes:]
        else:
            div_targets = torch.clone(targets)
            mask_old_cls = div_targets < nb_old_classes
            mask_new_cls = ~mask_old_cls

            div_targets[mask_old_cls] = 0
            div_targets[mask_new_cls] -= nb_old_classes - 1

        div_loss = args.head_div * criterion(div_output, div_targets) # 0.1, bce_with_logits

    cos_loss = torch.tensor(0., device=loss.device)
    cos_criterion = nn.CosineSimilarity(dim=2)
    if teacher_model is not None and args.cos_weight > 0:
        for tt in range(len(teacher_model.task_tokens)):
            pair_cos_loss = torch.mean(torch.abs(cos_criterion(teacher_model.task_tokens[tt],
                                                                   model.task_tokens[-1])))
            cos_loss += args.cos_weight * pair_cos_loss

    return loss, kd_loss, div_loss, attn_loss, cos_loss


@torch.no_grad()
def evaluate_teacher(data_loader, model, device, args=None):
    if args.single_head:
        print("Taking average of head logits")
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target, task_ids in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            if isinstance(output, dict):
                output = output['logits']
            loss = criterion(output, target)

        if args.single_head:
            if args.single_head_mode == 'average':
                output_new = output.reshape(output.shape[0], int(output.shape[1]/args.increment), args.increment)
                output = output_new.mean(dim=1)
            elif args.single_head_mode == 'top':
                output = output.reshape(output.shape[0], int(output.shape[1] / args.increment), args.increment)
                output = output[torch.arange(len(output)), output.amax(dim=2).max(dim=1)[1]]
            elif args.single_head_mode == 'voting':
                output_new = output.reshape(output.shape[0], int(output.shape[1]/args.increment), args.increment)
                output = torch.mean(F.one_hot(output_new.argmax(2), num_classes=args.increment).type(torch.float32),
                                        dim=1)
        acc1, acc5 = accuracy(output, target, topk=(1, min(5, output.shape[1])))

        batch_size = images.shape[0]
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f}'.format(top1=metric_logger.acc1))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, logger, args=None):
    if args.single_head:
        print("Taking average of head logits")
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target, task_ids in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            if isinstance(output, dict):
                output = output['logits']
            loss = criterion(output, target)

        if args.single_head:
            if args.single_head_mode == 'average':
                output = output.reshape(output.shape[0], int(output.shape[1] / args.increment), args.increment)
                output = output.sum(dim=1)
            elif args.single_head_mode == 'top':
                output = output.reshape(output.shape[0], int(output.shape[1] / args.increment), args.increment)
                output = output[torch.arange(len(output)), output.amax(dim=2).max(dim=1)[1]]
            elif args.single_head_mode == 'voting':
                output_new = output.reshape(output.shape[0], int(output.shape[1] / args.increment), args.increment)
                output = torch.mean(F.one_hot(output_new.argmax(2), num_classes=args.increment).type(torch.float32),
                                    dim=1)
        acc1, acc5 = accuracy(output, target, topk=(1, min(5, output.shape[1])))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        logger.add([output.cpu().argmax(dim=1), target.cpu(), task_ids], subset='test')

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f}  loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def eval_and_log(args, output_dir, model, model_without_ddp, optimizer, lr_scheduler,
                 epoch, task_id, loss_scaler, max_accuracy, accuracy_list,
                 n_parameters, device, data_loader_val, train_stats, log_store, log_path, logger,
                 model_log, skipped_task=False, acc5=None):
    if args.output_dir:
        if os.path.isdir(args.resume):
            checkpoint_paths = [os.path.join(args.resume, f'checkpoint_{task_id}.pth')]
        else:
            checkpoint_paths = [output_dir / f'checkpoint_{task_id}.pth']
        for checkpoint_path in checkpoint_paths:
            if skipped_task:
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

    test_stats = evaluate(data_loader_val, model, device, logger, args=args)
    print(f"Accuracy of the network on the {len(data_loader_val.dataset)} test images: {test_stats['acc1']:.1f}%")
    max_accuracy = max(max_accuracy, test_stats["acc1"])
    print(f'Max accuracy: {max_accuracy:.2f}%')
    accuracy_list.append(test_stats['acc1'])
    acc5.append(test_stats['acc5'])

    if test_stats is not None:
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                    'epoch': epoch,
                    'n_parameters': n_parameters}

    mean_acc5 = -1.0
    if log_store is not None:
        log_store['results'][task_id] = log_stats
        all_acc5 = [task_log['test_acc5'] for task_log in log_store['results'].values()]
        mean_acc5 = sum(all_acc5) / len(all_acc5)

    if log_path is not None and utils.is_main_process():
        with open(log_path, 'a+') as f:
            f.write(json.dumps({
                'task': task_id,
                'epoch': epoch,
                'acc': round(100 * logger.accuracy, 2),
                'avg_acc': round(100 * logger.average_incremental_accuracy, 2),
                'forgetting': round(100 * logger.forgetting, 6),
                'acc_per_task': [round(100 * acc_t, 2) for acc_t in logger.accuracy_per_task],
                'train_lr': log_stats.get('train_lr', 0.),
                'bwt': round(100 * logger.backward_transfer, 2),
                'fwt': round(100 * logger.forward_transfer, 2),
                'test_acc1': round(log_stats['test_acc1'], 2),
                'test_acc5': round(log_stats['test_acc5'], 2),
                'mean_acc5': round(mean_acc5, 2),
                'train_loss': round(log_stats.get('train_loss', 0.), 5),
                'test_loss': round(log_stats['test_loss'], 5),
                **model_log
            }) + '\n')
    if args.output_dir and utils.is_main_process():
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

    return max_accuracy


def indexes_task_outputs(logits, targets, increment_per_task):
    if increment_per_task[0] != increment_per_task[1]:
        raise NotImplementedError(f'Not supported yet for non equal task size')

    inc = increment_per_task[0]
    indexes = torch.zeros(len(logits), inc).long()

    for r in range(indexes.shape[0]):
        for c in range(indexes.shape[1]):
            indexes[r, c] = (targets[r] // inc) * inc + r * logits.shape[1] + c

    indexed_logits = logits.view(-1)[indexes.view(-1)].view(len(logits), inc)
    indexed_targets = targets % inc

    return indexed_logits, indexed_targets
