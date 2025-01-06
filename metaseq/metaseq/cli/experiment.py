import logging
import os
import sys
import copy
from argparse import Namespace
from itertools import chain
from time import time

import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf

from metaseq import checkpoint_utils, distributed_utils, options, utils
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.logging import metrics, progress_bar

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("metaseq.cli.experiment")

class ActivationHook:
    def __init__(self, percentiles):
        self.running_quantiles = None
        self.n_batches = 0
        # Store the percentiles we actually need
        self.percentiles = [1 - p for p in percentiles]  # Convert sparsity % to keep %
        
    def __call__(self, module, input, output):
        # Convert to float32 since quantile requires float or double
        tensor = torch.abs(output.detach().clone()).float()
        tensor = tensor.flatten(0, -2)
        
        percentiles_tensor = torch.tensor(self.percentiles, 
                                        device=tensor.device, 
                                        dtype=torch.float32)  # Explicitly use float32
        
        batch_quantiles = torch.quantile(tensor, percentiles_tensor, dim=1).mean(dim=1)
        
        # Update running average
        if self.running_quantiles is None:
            self.running_quantiles = batch_quantiles
        else:
            self.running_quantiles = (self.running_quantiles * self.n_batches + batch_quantiles) / (self.n_batches + 1)
        self.n_batches += 1
        
    def get_percentile_thresholds(self):
        if self.running_quantiles is None:
            return None
        return self.running_quantiles

def get_data_loader(models, cfg, task, dataset, data_parallel_world_size, data_parallel_rank):
    data_loader = task.get_batch_iterator(
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
        data_loader,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        prefix=f"validating model",
    )
    return data_loader

def validate_model(models, cfg, task, dataset, data_parallel_world_size, data_parallel_rank, criterion, use_cuda=True):
    data_loader = get_data_loader(models, cfg, task, dataset, data_parallel_world_size, data_parallel_rank)
    model = models[0]
    model.eval()
    log_outputs = []
    
    with torch.no_grad():
        for i, sample in enumerate(data_loader):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            loss, sample_size, logging_output = task.valid_step(sample, model, criterion)
            log_outputs.append(logging_output)
            
    with metrics.aggregate() as agg:
        task.reduce_metrics(log_outputs, criterion)
        loss = agg['loss'].avg.item()
    return loss

def find_threshold_val(models, cfg, task, dataset, data_parallel_world_size, data_parallel_rank, criterion, module_name, perc_vals, base_loss, loss_inc, thresh_hist, step='linear', use_cuda=True):
    parts = module_name.split('.')
    model = models[0]
    module = model
    for part in parts:
        module = getattr(module, part)
            
    logger.info(f"Collecting activation statistics for {module_name}...")
    hook = ActivationHook(perc_vals)
    handle = module.register_forward_hook(hook)

    data_loader = get_data_loader(models, cfg, task, dataset, data_parallel_world_size, data_parallel_rank)

    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(data_loader):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            _ = model(**sample["net_input"])
    
    handle.remove()
    
    thresholds = hook.get_percentile_thresholds()
    
    module_type = '_'.join(module_name.split('.')[-2:])
    res_loss, res_perc = base_loss, perc_vals[0]
    
    init_perc = 1.0
    if module_type in thresh_hist:
        values, counts = np.unique(thresh_hist[module_type], return_counts=True)
        init_perc = values[np.argmax(counts)]
    
    if step == 'exp' or init_perc == 1.0 or init_perc == perc_vals[0]:
        for i, perc in enumerate(perc_vals):
            module.reinit(thresholds[i])
            cur_loss = validate_model(models, cfg, task, dataset, data_parallel_world_size, data_parallel_rank, criterion, use_cuda)
            if cur_loss / base_loss < loss_inc:
                res_loss, res_perc = cur_loss, perc
            else:
                break
    else:
        init_idx = np.where(perc_vals == init_perc)[0][0]
        module.reinit(thresholds[init_idx])
        cur_loss = validate_model(models, cfg, task, dataset, data_parallel_world_size, data_parallel_rank, criterion, use_cuda)
        
        if cur_loss / base_loss < loss_inc:
            res_loss, res_perc = cur_loss, init_perc
            for i in range(init_idx + 1, len(perc_vals)):
                module.reinit(thresholds[i])
                cur_loss = validate_model(models, cfg, task, dataset, data_parallel_world_size, data_parallel_rank, criterion, use_cuda)
                if cur_loss / base_loss < loss_inc:
                    res_loss, res_perc = cur_loss, perc_vals[i]
                else:
                    break
        else:
            for i in range(init_idx - 1, -1, -1):
                module.reinit(thresholds[i])
                cur_loss = validate_model(models, cfg, task, dataset, data_parallel_world_size, data_parallel_rank, criterion, use_cuda)
                if cur_loss / base_loss < loss_inc:
                    res_loss, res_perc = cur_loss, perc_vals[i]
                    break
    
    final_idx = np.where(perc_vals == res_perc)[0][0]
    module.reinit(thresholds[final_idx])
    thresh_hist[module_type] = thresh_hist.get(module_type, []) + [res_perc]
    logger.info(f"{module_name}, percentile: {res_perc:.4f}")
    return res_loss, res_perc

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

    # Load model and task
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [cfg.common_eval.path],
        arg_overrides=overrides,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=False
    )
    model = models[0]

    # Move models to GPU
    for model in models:
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    criterion = task.build_criterion(saved_cfg.criterion)
    criterion.eval()

    # Load dataset
    task.load_dataset(cfg.dataset.valid_subset, combine=False, epoch=1, task_cfg=saved_cfg.task)
    dataset = task.dataset(cfg.dataset.valid_subset)

    if cfg.iter_topk.step == "linear":
        perc_vals = np.arange(cfg.iter_topk.start_k, 0.0, -0.1)
    else:
        perc_vals = np.unique(np.round(np.geomspace(cfg.iter_topk.start_k, 0.01, cfg.iter_topk.k_num), 2))[::-1]

    logger.info(f'Iterating over percentile vals: {perc_vals}')

    iter_start_time = time()
    loss_inc = cfg.iter_topk.loss_inc
    thresh_hist = dict()
    base_loss = validate_model(models, cfg, task, dataset, data_parallel_world_size, data_parallel_rank, criterion, use_cuda)
    # Process each layer
    for i in range(len(model.decoder.layers)):
        # Original OPT uses combined qkv module, not separate q,k,v
        module_name = f'decoder.layers.{i}.sparsing_fns.qkv'
        base_loss, _ = find_threshold_val(models, cfg, task, dataset, data_parallel_world_size, data_parallel_rank, criterion,
                                        module_name, perc_vals, base_loss, loss_inc, 
                                        thresh_hist, cfg.iter_topk.step, use_cuda)
        
        # up_proj and down_proj instead of fc1 and fc2
        module_name = f'decoder.layers.{i}.sparsing_fns.up_proj'
        base_loss, _ = find_threshold_val(models, cfg, task, dataset, data_parallel_world_size, data_parallel_rank, criterion,
                                        module_name, perc_vals, base_loss, loss_inc,
                                        thresh_hist, cfg.iter_topk.step, use_cuda)
        
        module_name = f'decoder.layers.{i}.sparsing_fns.down_proj'
        base_loss, _ = find_threshold_val(models, cfg, task, dataset, data_parallel_world_size, data_parallel_rank, criterion,
                                        module_name, perc_vals, base_loss, loss_inc,
                                        thresh_hist, cfg.iter_topk.step, use_cuda)

    logger.info('Average percentile: {:.4f}'.format(
        sum([sum(value) for value in thresh_hist.values()]) / 
        sum([len(value) for value in thresh_hist.values()])
    ))
    
    # Save model
    model_name = cfg.common_eval.path.split('/')[-1].split('.pt')[0]
    model_name += f'_sparse_hs-start_{cfg.iter_topk.start_k}-step_{cfg.iter_topk.step}'
    if cfg.iter_topk.step == "exp":
        model_name += f'{cfg.iter_topk.k_num}'
    if cfg.iter_topk.loss_inc != 1.001:
        model_name += f'-loss_inc_{cfg.iter_topk.loss_inc}'
    model_name += '-model_part-0.pt'

    saved_cfg.model.sparsity = {'sparsing_fn': 'hardshrink'}
    state_dict = {
        "cfg": OmegaConf.to_container(saved_cfg),
        "model": model.state_dict(),
        "criterion": criterion.state_dict() if utils.has_parameters(criterion) else None
    }
    state_dict = utils.move_to_cpu(state_dict, cast_to_fp32=True)
    
    save_dir = '/'.join(cfg.common_eval.path.split('/')[:-1])
    torch.save(state_dict, os.path.join(save_dir, model_name))

    logger.info(f'Model saved: {model_name}')
    logger.info('Total time: {:.0f} hours {:.0f} minutes'.format(
        (time() - iter_start_time) // 3600, 
        (time() - iter_start_time) % 3600 // 60
    ))

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