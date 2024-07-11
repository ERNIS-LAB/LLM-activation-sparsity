if __name__ == '__main__':
    import os, sys, types, json, math, time
    import numpy as np
    start_time = time.time()
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    # try:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    # except:
    #     pass
    import torch
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    from torch.nn import functional as F
    from tokenizers import Tokenizer

    os.environ["RWKV_CUDA_ON"] = '1'


    from lm_eval import tasks, evaluator
    from lm_eval.api.model import TemplateLM

    ########################################################################################################
    os.environ['RWKV_FLOAT_MODE'] = 'fp32' # bf16 or fp32
    os.environ['RWKV_RUN_DEVICE'] = 'cuda' # currently model requires CUDA
    RUN_DEVICE = os.environ['RWKV_RUN_DEVICE']

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--sparsity", default=None, type=json.loads)
    parser.add_argument("--load_model", default="", type=str)  # full path, with .pth
    parser.add_argument("--benchmarks", default='', type=str)

    args = parser.parse_args()
    
    WORD_NAME = ['20B_tokenizer.json', '20B_tokenizer.json']
    n_layer = 12
    n_embd = 768
    ctx_len = 1024

    from src.utils import get_model_settings

    sets = get_model_settings(args.load_model)
    if sets is not None:
        n_layer, n_embd, ctx_len = sets
        
    UNKNOWN_CHAR = None

    os.environ["RWKV_JIT_ON"] = '0'
    os.environ["RWKV_T_MAX"] = str(ctx_len)

    from src.model import RWKV


    from src.utils import TOKENIZER
    tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)

    os.environ["TOKENIZERS_PARALLELISM"] = "false" # to avoid warning but might be useful


    args.vocab_size = tokenizer.vocab_size
    args.ctx_len = ctx_len
    args.n_embd = n_embd
    args.n_layer = n_layer
    args.head_qk = 0
    args.pre_ffn = 0
    args.grad_cp = 0
    args.my_pos_emb = 0
    args.dropout = 0

    args.RUN_DEVICE = RUN_DEVICE
    args.FLOAT_MODE = os.environ['RWKV_FLOAT_MODE']
    if '_hs' in args.load_model:
        args.sparsity = {"sparsing_fn": "hardshrink"}

    from pytorch_lightning import seed_everything
    seed_everything(42)
    
    model = RWKV(args).to(RUN_DEVICE)

    m2 = torch.load(args.load_model, map_location='cpu')
    if hasattr(args, "sparsity"):
        model.load_state_dict(m2, strict=False)
    else:
        model.load_state_dict(m2)

    if os.environ['RWKV_FLOAT_MODE'] == 'fp16':
        model = model.half()
    elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
        model = model.bfloat16()

    tokenizer = tokenizer.tokenizer

    eval_tasks = args.benchmarks.split(',')
    # eval_tasks = []
    # eval_tasks += ['piqa', 'lambada_openai', 
    #                ]
    # eval_tasks += ['piqa', 'hellaswag','winogrande', 'arc_easy', 'arc_challenge', 'lambada_openai', 
    #                'openbookqa', 'sciq', 'record']
    # eval_tasks += ['storycloze_2016', 'headqa', 'triviaqa']
    # COPA???


    RWKV_PAD = tokenizer.encode('\n') # we will use '\n' as PAD

    print('RWKV_PAD', RWKV_PAD)

    ########################################################################################################

    logitBuf = {}
    correctBuf = {}

    class TokenizerWrapper:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.eos_token_id = 0

        def encode(self, string: str, add_special_tokens=False):
            return self.tokenizer.encode(string)

        def decode(self, tokens):
            return self.tokenizer.decode(tokens)

    class EvalHarnessAdapter(TemplateLM):
        def __init__(self):
            super().__init__()
            self.tokenizer = TokenizerWrapper(tokenizer)

        def eot_token_id(self):
            return self.tokenizer.eos_token_id
        
        def tok_encode(self, string: str, **kwargs):
            return self.tokenizer.encode(string)

        def generate_until(self, requests): # designed for coqa
            pass
            # res = []
            # for i in range(len(requests)):
            #     if i % 50 == 0:
            #         print(i)
            #     otoken = []
            #     while True:
            #         src = self.tokenizer.encode(requests[i][0]) + otoken

            #         src = src[-4096:]
            #         outputs, _ = model.forward(src, None)
                    
            #         otoken += [int(torch.argmax(outputs))]
            #         ss = self.tokenizer.decode(otoken)
            #         if '\n' in ss or len(ss) > 200:
            #             if not ss.endswith('\n'):
            #                 ss = ss + '\n'
            #             print(ss)
            #             res += [(ss)]
            #             break
            # print(res)
            # return res

        def _encode_pair(self, context, continuation):
            n_spaces = len(context) - len(context.rstrip())
            if n_spaces > 0:
                continuation = context[-n_spaces:] + continuation
                context = context[:-n_spaces]

            context_enc = self.tok_encode(context)
            continuation_enc = self.tok_encode(continuation)

            return context_enc, continuation_enc
        
        def loglikelihood_rolling(self, requests):
            # return super().loglikelihood_rolling(requests)
            pass
        
        def _loglikelihood_tokens(self, requests, disable_tqdm=False):
            global logitBuf, correctBuf

            res = []

            for COUNTER in range(len(requests)):
                n = COUNTER
                raw_src = requests[n][0][0] + requests[n][0][1]

                src = requests[n][1] + requests[n][2]

                raw_src = '\n' + raw_src
                src = RWKV_PAD + src

                sss = str(src)
                correct = True
                if sss in logitBuf:
                    logit = logitBuf[sss]
                    correct = correctBuf[sss]
                else:
                    q_len = len(requests[n][1])
                    q_len += len(RWKV_PAD)
                    logit = 0
                    
                    with torch.no_grad():
                        state = None
                        outputs = model.forward(torch.tensor([src]).to(RUN_DEVICE))[0].detach().cpu().float()
                        for i in range(q_len-1, len(src)-1):
                            oo = outputs[i]
                            dst = src[i+1]
                            logit += math.log(F.softmax(oo, dim=-1)[dst])
                            _, s_index = torch.sort(oo, descending=True)
                            pred = s_index[0].item()
                            if pred != dst:
                                correct = False
                        outputs = None
                        pred = None
                    logitBuf[sss] = logit
                    correctBuf[sss] = correct
                
                res += [(logit, correct)]
                if n % 1000 == 0:
                    print(f'{n//1000}/{len(requests)//1000}', end = ' ', flush=True)
            return res

        @torch.no_grad()
        def run_eval(self, eval_tasks=None, num_fewshot=0, bootstrap_iters=2):
            results = evaluator.evaluate(
                lm=self,
                task_dict=tasks.get_task_dict(eval_tasks),
                # provide_description=False,
                # num_fewshot=num_fewshot,
                limit=None,
                bootstrap_iters=bootstrap_iters,
            )
            # results = evaluator.simple_evaluate(
            #     model=self,
            #     task_dict=eval_tasks,
            #     # provide_description=False,
            #     num_fewshot=num_fewshot,
            #     limit=None,
            #     bootstrap_iters=bootstrap_iters,
            # )
            return results

    adapter = EvalHarnessAdapter()
    results = adapter.run_eval(
        eval_tasks=eval_tasks,
        bootstrap_iters=10000,
    )
    bench_metr = [('piqa', 'acc,none'), ('winogrande', 'acc,none'), ('hellaswag', 'acc_norm,none'), 
                ('arc_easy', 'acc,none'), ('arc_challenge', 'acc_norm,none'), ('lambada_openai', 'perplexity,none'), 
                ('lambada_openai', 'acc,none'), ('openbookqa', 'acc_norm,none'), ('sciq', 'acc,none'),
                ('record', 'em,none')] # add other benchmarks

    import pandas as pd
    res_vals = []
    res_tasks = []
    for task, metric in bench_metr:
        if task in results["results"]:
            res_tasks.append(task + ' ' + metric)
            res_vals.append(results["results"][task][metric])
    
    df = pd.DataFrame([res_vals], columns=res_tasks)
    print()
    print('Model:', args.load_model.split('/')[-1])
    print('Time taken in minutes:', (time.time() - start_time) / 60)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)
