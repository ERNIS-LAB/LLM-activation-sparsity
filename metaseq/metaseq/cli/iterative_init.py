#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from argparse import Namespace
from itertools import chain
from time import time

import torch
from omegaconf import DictConfig, OmegaConf

from metaseq import checkpoint_utils, distributed_utils, options, utils
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.logging import metrics, progress_bar
from metaseq.trainer import Trainer

import numpy as np

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("metaseq.cli.iterative_init")


def main(cfg: DictConfig, override_args=None):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    if cfg.distributed_training.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None

    logger.info(overrides)
    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [cfg.common_eval.path],
        arg_overrides=overrides,
        suffix=cfg.checkpoint.checkpoint_suffix,
    )
    model = models[0]

    # Move models to GPU
    for model in models:
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Print args
    logger.info(saved_cfg)

    # Build criterion
    criterion = task.build_criterion(saved_cfg.criterion)
    criterion.eval()
    
    # trainer = Trainer(cfg, task, model, criterion)
    subset = cfg.dataset.valid_subset

    task.load_dataset(subset, combine=False, epoch=1, task_cfg=saved_cfg.task)
    dataset = task.dataset(subset)
    
    def validate_model():
        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=cfg.dataset.max_tokens,
            max_sentences=cfg.dataset.batch_size,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                *[m.max_positions() for m in models],
            ),
            ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
            seed=cfg.common.seed,
            num_shards=data_parallel_world_size,
            shard_id=data_parallel_rank,
            num_workers=cfg.dataset.num_workers,
            data_buffer_size=cfg.dataset.data_buffer_size,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.get_progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            prefix=f"valid on '{subset}' subset",
        )
        log_outputs = []
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            _loss, _sample_size, log_output = task.valid_step(sample, model, criterion)
            progress.log(log_output, step=i)
            log_outputs.append(log_output)

        if data_parallel_world_size > 1:
            log_outputs = distributed_utils.all_gather_list(
                log_outputs,
                max_size=cfg.common.all_gather_list_size,
                group=distributed_utils.get_data_parallel_group(),
            )
            log_outputs = list(chain.from_iterable(log_outputs))

        with metrics.aggregate() as agg:
            task.reduce_metrics(log_outputs, criterion)
            loss = agg['loss'].avg.item()
        return loss
    
    def set_module(model, module_name, new_module):
        # Split the module name to handle nested modules
        module_parts = module_name.split('.')
        current_module = model
        for part in module_parts[:-1]:
            current_module = getattr(current_module, part)
        setattr(current_module, module_parts[-1], new_module)
    
    def find_k_val(model, module_name, perc_vals, base_loss, loss_inc, n_embd, perc_hist, step='linear'):
        res_loss, res_perc = base_loss, 1.0
        init_perc = 1.0
        module_type = '_'.join(module_name.split('.')[-2:])
        if module_type in perc_hist.keys():
            values, counts = np.unique(perc_hist[module_type], return_counts=True)
            init_perc = values[np.argmax(counts)]
        if step == 'exp' or init_perc == 1.0 or init_perc == perc_vals[0]:
            for cur_perc in perc_vals:
                set_module(model, module_name, utils.TopK(int(n_embd * cur_perc)))
                cur_loss = validate_model()
                if cur_loss / base_loss < loss_inc:
                    res_loss, res_perc = cur_loss, cur_perc
                else:
                    break
        else:
            init_ind = np.where(perc_vals == init_perc)[0][0]
            cur_perc = init_perc
            set_module(model, module_name, utils.TopK(int(n_embd * cur_perc)))
            cur_loss = validate_model()
            if cur_loss / base_loss < loss_inc:
                res_loss, res_perc = cur_loss, cur_perc
                for i in range(init_ind + 1, len(perc_vals)):
                    cur_perc = perc_vals[i]
                    set_module(model, module_name, utils.TopK(int(n_embd * cur_perc)))
                    cur_loss = validate_model()
                    if cur_loss / base_loss < loss_inc:
                        res_loss, res_perc = cur_loss, cur_perc
                    else:
                        break
            else:
                for i in range(init_ind - 1, -1, -1):
                    cur_perc = perc_vals[i]
                    set_module(model, module_name, utils.TopK(int(n_embd * cur_perc)))
                    cur_loss = validate_model()
                    if cur_loss / base_loss < loss_inc:
                        res_loss, res_perc = cur_loss, cur_perc
                        break
                    else:
                        continue
        set_module(model, module_name, utils.TopK(int(n_embd * res_perc)))
        perc_hist[module_type] = perc_hist.get(module_type, []) + [res_perc]
        logger.info(f"{module_name}, {res_perc:.4f}")
        return res_loss, res_perc

    if cfg.iter_topk.step == "linear":
        perc_vals = np.arange(cfg.iter_topk.start_k, 0.0, -0.1)
    elif cfg.iter_topk.step == "exp":
        perc_vals = np.unique(np.round(np.geomspace(cfg.iter_topk.start_k, 0.01, cfg.iter_topk.k_num), 2))[::-1]

    logger.info(f'Iterating over k vals: {perc_vals}')

    iter_start_time = time()

    loss_inc = cfg.iter_topk.loss_inc
    step = cfg.iter_topk.step
    perc_hist = dict()
    base_loss = validate_model()
    for i, block in enumerate(model.decoder.layers):
        for name in ['qkv', 'up_proj', 'down_proj']:
            module_name = f'decoder.layers.{i}.sparsing_fns.{name}'
            if name == 'down_proj':
                n_embd = saved_cfg.model.decoder_ffn_embed_dim
            else:
                n_embd = saved_cfg.model.decoder_embed_dim
            base_loss, _ = find_k_val(model, module_name, perc_vals, base_loss, loss_inc, 
                                            n_embd, perc_hist, step)
        
    by_spars_fun = dict()
    for name, module in model.named_modules():
        if isinstance(module, utils.TopK):
            type_name = name.split('.')[-1]
            if 'down_proj' in name:
                perc = module.k.item() / saved_cfg.model.decoder_ffn_embed_dim
            else:
                perc = module.k.item() / saved_cfg.model.decoder_embed_dim
            by_spars_fun[type_name] = by_spars_fun.get(type_name, []) + [np.round(perc, 2)]

    logger.info(by_spars_fun)
    logger.info('Average K%: {:.4f}'.format(sum([sum(value) for value in by_spars_fun.values()]) / sum([len(value) for value in by_spars_fun.values()])))
    for key, value in by_spars_fun.items():
        logger.info(f'{key} {sum(value) / len(value):.4f}')

    model_name = cfg.common_eval.path.split('/')[-1].split('.pt')[0] + f'_sparse_topk-start_{cfg.iter_topk.start_k}-step_{cfg.iter_topk.step}'
    if cfg.iter_topk.step == "exp":
        model_name = model_name + f'{cfg.iter_topk.k_num}'
    if cfg.iter_topk.loss_inc != 1.001:
        model_name = model_name + f'-loss_inc_{cfg.iter_topk.loss_inc}'
    model_name = model_name + '-model_part-0.pt'
    

    saved_cfg.model.sparsity = {'sparsing_fn': 'topk'}
    
    model_state_dict = model.state_dict()

    state_dict = {
        "cfg": OmegaConf.to_container(saved_cfg)
                if OmegaConf.is_config(saved_cfg)
                else saved_cfg,
        "model": model_state_dict,
        "criterion": (
            criterion.state_dict()
            if utils.has_parameters(criterion)
            else None
        )
    }
    state_dict = utils.move_to_cpu(
                state_dict,
                cast_to_fp32=True,
            )
    
    save_dir = '/'.join(cfg.common_eval.path.split('/')[:-1])
    torch.save(state_dict, os.path.join(save_dir, model_name))

    logger.info(f'Model saved: {model_name}')
    logger.info('Total time: {:.0f} hours {:.0f} minutes'.format((time() - iter_start_time) // 3600, (time() - iter_start_time) % 3600 // 60))
        

def cli_main():
    parser = options.get_iter_topk_parser()
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_iter_topk_parser()
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    distributed_utils.call_main(
        convert_namespace_to_omegaconf(args), main, override_args=override_args
    )


if __name__ == "__main__":
    cli_main()