from typing import Tuple
from lm_eval import tasks, evaluator
from lm_eval.api.model import TemplateLM

import torch
import math
import sys
import time
from tqdm import tqdm
from argparse import Namespace

from omegaconf import DictConfig

from metaseq import checkpoint_utils, distributed_utils, options, utils
from metaseq.service.utils import normalize_newlines
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.tasks import BaseTask

class TokenizerWrapper:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.eos_token_id = 0

        def encode(self, string: str, add_special_tokens=False):
            return self.tokenizer.encode(string)

        def decode(self, tokens):
            return self.tokenizer.decode(tokens)
        
class EvalHarnessAdapter(TemplateLM):
    def __init__(self, model, task):
        super().__init__()
        self.task = task
        self.tokenizer = TokenizerWrapper(self.task.tokenizer)
        self.model = model

    def tok_encode(self, string: str, **kwargs):
        return self.tokenizer.encode(normalize_newlines(string)).ids
    
    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        context_enc = self.tok_encode(context)
        continuation_enc = self.tok_encode(continuation)

        return context_enc, continuation_enc
    
    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        return super().loglikelihood_rolling(requests, disable_tqdm)
    
    def generate_until(self, requests, disable_tqdm: bool = False) -> tasks.List[str]:
        return super().generate_until(requests, disable_tqdm)

    def eot_token_id(self):
        return self.tokenizer.eos_token_id
    
    # def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
    #     print('loglikelihood_rolling')
    #     return super().loglikelihood_rolling(requests, disable_tqdm)
    
    # def loglikelihood(self, requests, disable_tqdm: bool = False):
    #     return super().loglikelihood(requests, disable_tqdm)
    
    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
    
        self.model.eval()
        res = []
        req_tokens = [req[1] + req[2] for req in requests]
        req_len = [len(req) for req in req_tokens]


        for req_num in tqdm(range(0, len(requests), 16)):
            src = req_tokens[req_num: req_num + 16]
            max_len = max(req_len[req_num: req_num + 16])
            max_len = max(pow(2, math.ceil(math.log(max_len)/math.log(2))), 256)

            padded_src = torch.tensor([lst + [self.task.dictionary.pad()] * (max_len - len(lst)) for lst in src])
            
            with torch.no_grad():
                padded_src = padded_src.to('cuda')
                outputs = self.model(padded_src)[0].cpu().float()
            
            for req_i in range(len(src)):
                logit = 0
                correct = True
                req_i_global = req_num + req_i

                cur_out = outputs[req_i]
                for i in range(len(requests[req_i_global][1]) - 1, req_len[req_i_global] - 1):
                    oo = cur_out[i]
                    dst = src[req_i][i + 1]
                    logit += utils.log_softmax(oo, dim=-1)[dst].item() 
                    _, s_index = torch.sort(oo, descending=True)
                    pred = s_index[0].item()
                    if pred != dst:
                        correct = False
                cur_out = None
                pred = None

                res += [(logit, correct)]

        return res


def main(cfg: DictConfig, override_args=None):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

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

    start_time = time.time()

    tasks_list = ['piqa','hellaswag','winogrande','arc_easy','arc_challenge','lambada_openai','sciq']

    # tasks = ['piqa','hellaswag','winogrande','arc_easy','arc_challenge','lambada_openai', 'sciq', 'triviaqa'] # relu strikes back

    adapter = EvalHarnessAdapter(model, task)
    results = evaluator.evaluate(
                lm=adapter,
                task_dict=tasks.get_task_dict(tasks_list),
                limit=None,
                bootstrap_iters=10000,
            )

    bench_metr = [('piqa', 'acc,none'), ('winogrande', 'acc,none'), ('hellaswag', 'acc,none'), 
                ('arc_easy', 'acc,none'), ('arc_challenge', 'acc_norm,none'), 
                ('lambada_openai', 'acc,none'), ('openbookqa', 'acc_norm,none'), ('sciq', 'acc,none'),
                ('record', 'em,none')] # add other benchmarks

    import pandas as pd
    res_vals = []
    res_tasks = []
    for task, metric in bench_metr:
        if task in results["results"]:
            res_tasks.append(task + ' ' + metric)
            res_vals.append(results["results"][task][metric])
    # for res in results["results"]:
    # print(results["results"])
    
    df = pd.DataFrame([res_vals], columns=res_tasks)
    print()
    print('Model:', cfg.common_eval.path.split('/')[-1])
    print('Time taken in minutes:', (time.time() - start_time) / 60)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

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
