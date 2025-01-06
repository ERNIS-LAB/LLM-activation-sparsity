import logging
import os
import sys
import copy
from argparse import Namespace
from itertools import chain
from time import time
from tqdm import tqdm

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
logger = logging.getLogger("metaseq.cli.topk_2_hardshrink")

class Hook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.percenriles = []
        self.lower_bounds = torch.tensor([]).to('cuda')
        self.module = module

    def hook_fn(self, module, input, output):
        
        tensor = torch.abs(output.detach().clone())
        tensor = torch.min(tensor.masked_fill(tensor == 0, float('inf')).flatten(0, -2), dim=1).values
        self.lower_bounds = torch.cat([self.lower_bounds, torch.mean(tensor).unsqueeze(0)])

    def close(self):
        self.hook.remove()

def topk2hs(model_hs, model_topk, init_iterator, use_cuda=True):
    """
    Replaces Top-K layers (K values) with HardShrink (thresholds) in the model.

    Args:
        model (nn.Module): The model to apply thresholding to.
        model_topk (nn.Module): The model with Top-K layers to initialize the thresholds.
        iterator: An iterator for hardshrink initialization.

    Returns:
        model with topk layers replaced with hardshrink layers.
    """
    hooks = dict()
    for name, module in model_topk.named_modules():
        if len(name.split('.')) > 2 and name.split('.')[-2] == 'sparsing_fns':
            hooks[name] = Hook(module)

    model_topk.eval()
    with torch.no_grad():
        for i, sample in enumerate(init_iterator):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            _ = model_topk(**sample["net_input"])
    
    for name, module in model_hs.named_modules():
        if name in hooks.keys():
            module.reinit(hooks[name].lower_bounds.mean().type_as(module.thresholds.data))

    return model_hs

def main(cfg: DictConfig, override_args=None):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    use_fp16 = False
    use_cuda = torch.cuda.is_available()

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

    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, topk_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [cfg.common_eval.path],
        arg_overrides=overrides,
        suffix=cfg.checkpoint.checkpoint_suffix,
    )
    model_topk = models[0]

    hs_cfg = copy.deepcopy(topk_cfg)
    hs_cfg.model.sparsity = {'sparsing_fn': 'hardshrink'}
    model_hs = task.build_model(hs_cfg.model)
    model_hs.load_state_dict(model_topk.state_dict(), strict=False)
    # print(model_hs, file = sys.stderr)
    # print(model_topk, file = sys.stderr)

    if use_fp16:
        model_topk.half()
        model_hs.half()
    if use_cuda:
        model_topk.cuda()

    subset = cfg.dataset.valid_subset

    task.load_dataset(subset, combine=False, epoch=1, task_cfg=topk_cfg.task)
    dataset = task.dataset(subset)

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
        prefix="",
    )
    

    import time
    start_time = time.time()
    topk2hs(model_hs, model_topk, progress, use_cuda)
    logger.info(f'total time: {time.time() - start_time}')
    
    criterion = task.build_criterion(hs_cfg.criterion)

    model_state_dict = model_hs.state_dict()
    state_dict = {
        "cfg": OmegaConf.to_container(hs_cfg)
                if OmegaConf.is_config(hs_cfg)
                else hs_cfg,
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
    
    save_model = cfg.common_eval.path.replace('.pt', '_hs-model_part-0.pt')
    torch.save(state_dict, save_model)

    

def cli_main():
    parser = options.get_validation_parser()
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_validation_parser()
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    distributed_utils.call_main(
        convert_namespace_to_omegaconf(args), main, override_args=override_args
    )


if __name__ == "__main__":
    cli_main()