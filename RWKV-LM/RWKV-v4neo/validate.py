if __name__ == "__main__":
    import os
    import json
    import torch

    from argparse import ArgumentParser
    from pytorch_lightning import Trainer

    from torch.utils.data import DataLoader

    parser = ArgumentParser()

    parser.add_argument("--load_model", default="", type=str)  # full path, with .pth
    parser.add_argument("--random_seed", default="-1", type=int)

    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--data_type", default="binidx", type=str)
    parser.add_argument("--vocab_size", default=50277, type=int)  # vocab_size = 0 means auto (for char-level LM and .txt data)

    parser.add_argument("--micro_bsz", default=16, type=int)  # micro batch size (batch size per GPU)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--ctx_len", default=1024, type=int)

    parser.add_argument("--sparsity", default=None, type=json.loads)


    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    
    args.my_testing = ''
    args.my_pos_emb = 0
    args.dropout = 0
    args.my_qa_mask = 0
    args.grad_cp = 0
    args.head_qk = 0
    args.pre_ffn = 0

    os.environ["RWKV_JIT_ON"] = "0"
    if "deepspeed_stage_3" in args.strategy:
        os.environ["RWKV_JIT_ON"] = "0"

    os.environ["RWKV_FLOAT_MODE"] = args.precision

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

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

    from pytorch_lightning import seed_everything

    if args.random_seed >= 0:
        print(f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n" * 3)
        seed_everything(args.random_seed)

    load_dict = torch.load(args.load_model, map_location="cpu")
    load_keys = list(load_dict.keys())
    for k in load_keys:
        if k.startswith('_forward_module.'):
            load_dict[k.replace('_forward_module.','')] = load_dict[k]
            del load_dict[k]

    from src.trainer import train_callback
            
    trainer = Trainer.from_argparse_args(
            args,
            logger=False,
            deterministic=True,
            callbacks=[train_callback(args)],
        )
    
    from src.dataset import ValDataset
    test_data = ValDataset(args)
    args.vocab_size = test_data.vocab_size

    data_loader = DataLoader(test_data, shuffle=False, pin_memory=True, batch_size=args.micro_bsz, num_workers=16, persistent_workers=False, drop_last=False)
    
    from src.model import RWKV
    def validate_model(args, load_dict, trainer, dataloader):

        model = RWKV(args)

        # print({ k: v.item() for k, v in load_dict.items() if k.split('.')[-1] == 'k'})
        
        if hasattr(args, 'sparsity'):
            print(f"Sparsity: {args.sparsity}")
            model.load_state_dict(load_dict, strict=False)
        else:
            model.load_state_dict(load_dict)
        
        # print({k[:k.rfind('.')]: v for k, v in load_dict.items() if k.split('.')[-1] == 'thresholds'})
        # print(load_dict)
        # print(model.blocks[0])
        
        return trainer.validate(model, dataloaders=dataloader, verbose=False)[0]
    
    results = validate_model(args, load_dict, trainer, data_loader)
    if 'sparsity_stats' in results:
        results['sparsity_stats'] = {k: v.item() for k, v in results['sparsity_stats'].items()}
    
    print(results)

    dict_by_fun_type = {}
    for k, v in results['sparsity_stats'].items():
        name = '_'.join(k.split('_')[2:])
        dict_by_fun_type[name] = dict_by_fun_type.get(name, []) + [v]
    
    for k, v in dict_by_fun_type.items():
        dict_by_fun_type[k] = sum(v) / len(v)
    print(dict_by_fun_type)