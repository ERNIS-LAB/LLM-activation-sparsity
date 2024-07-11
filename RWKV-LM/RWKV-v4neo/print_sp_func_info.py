if __name__ == "__main__":
    import os
    import json
    import torch

    from argparse import ArgumentParser
    from pytorch_lightning import Trainer

    from torch.utils.data import DataLoader

    parser = ArgumentParser()

    parser.add_argument("--load_model", default="", type=str)
    parser.add_argument("--vocab_size", default=50277, type=int)  # vocab_size = 0 means auto (for char-level LM and .txt data)

    parser.add_argument("--micro_bsz", default=16, type=int)  # micro batch size (batch size per GPU)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--ctx_len", default=1024, type=int)

    parser.add_argument("--sparsity", default=None, type=json.loads)

    args = parser.parse_args()
    
    # args.load_model = "/mnt/space/ivan/transformers/rwkv/RWKV-LM/RWKV-v4neo/experiments/topk_init/RWKV-4-Pile-169M-20220807-8023_sparse_hs.pth"
    # args.sparsity = {'sparsing_fn': 'hardshrink', 'hardshrink_lambda': 0.0}
    # args.load_model = "/mnt/space/ivan/transformers/rwkv/RWKV-LM/RWKV-v4neo/experiments/topk_init/RWKV-4-Pile-169M-20220807-8023_sparse_topk.pth"
    args.load_model = '/mnt/space/ivan/transformers/rwkv/RWKV-LM/RWKV-v4neo/experiments/topk_init/RWKV-4-Pile-3B_sparse_topk-start_0.2-step_exp8.pth'
    args.sparsity = {'sparsing_fn': 'topk'}
    
    args.my_testing = ''
    args.my_pos_emb = 0
    args.dropout = 0
    args.my_qa_mask = 0
    args.grad_cp = 0
    args.head_qk = 0
    args.pre_ffn = 0
    args.precision = "bf16"

    os.environ["RWKV_JIT_ON"] = "0"

    os.environ["RWKV_FLOAT_MODE"] = args.precision

    if "32" in args.precision:
        args.precision = 32
    elif args.precision == "fp16":
        args.precision = 16
    else:
        args.precision = "bf16"

    from src.utils import get_model_settings

    sets = get_model_settings(args.load_model)
    if sets is not None:
        args.n_layer, args.n_embd, args.ctx_len = sets
        
    os.environ["RWKV_T_MAX"] = str(args.ctx_len)
    os.environ["RWKV_MY_TESTING"] = args.my_testing

    load_dict = torch.load(args.load_model, map_location="cpu")
    load_keys = list(load_dict.keys())
    for k in load_keys:
        if k.startswith('_forward_module.'):
            load_dict[k.replace('_forward_module.','')] = load_dict[k]
            del load_dict[k]

    from src.model import RWKV

    model = RWKV(args)
    if hasattr(args, 'sparsity'):
        print(f"Sparsity: {args.sparsity}")
        model.load_state_dict(load_dict, strict=False)
    else:
        model.load_state_dict(load_dict)
        
    model = model.bfloat16()

    from src.utils import HardShrink, TopK
    
    values = {}
    vals_dict = {}
    for name, module in model.named_modules():
        if len(name.split('.')) > 1 and 'sparsing_fns' == name.split('.')[-2]:
            module_type = name.split('.')[-3] + '_' + name.split('.')[-1]
            if isinstance(module, HardShrink):
                print(name, module.thresholds[0].item())
            elif isinstance(module, TopK):
                if 'val' in name and 'ffn' in name:
                    k_val = round(module.k.item() / args.dim_ffn, 2)
                else:
                    k_val = round(module.k.item() / args.n_embd, 2)
                values[k_val] = values.get(k_val, 0) + 1
                vals_dict[module_type] = vals_dict.get(module_type, []) + [k_val]
                print(name, k_val)
    if vals_dict != {}:
        print(vals_dict)
    values = list(values.items())
    values = sorted(values, key=lambda x: x[0])
    print(values)
            