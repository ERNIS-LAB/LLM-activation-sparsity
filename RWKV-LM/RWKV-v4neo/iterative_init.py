if __name__ == "__main__":
    import os
    import torch
    import numpy as np
    from time import time

    from argparse import ArgumentParser
    from pytorch_lightning import Trainer

    from torch.utils.data import DataLoader

    parser = ArgumentParser()

    parser.add_argument("--load_model", default="", type=str)  # full path, with .pth
    parser.add_argument("--random_seed", default="-1", type=int)

    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--data_type", default="binidx", type=str)
    parser.add_argument("--vocab_size", default=50277, type=int)  # vocab_size = 0 means auto (for char-level LM and .txt data)
    parser.add_argument("--save_dir", default="out", type=str)

    parser.add_argument("--micro_bsz", default=16, type=int)  # micro batch size (batch size per GPU)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--ctx_len", default=1024, type=int)

    parser.add_argument("--loss_inc", default=1.001, type=float)
    parser.add_argument("--start_k", default=1.0, type=float)
    parser.add_argument("--step", default="linear", type=str)
    parser.add_argument("--k_num", default=8, type=int)

    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    
    args.my_testing = ''
    args.my_pos_emb = 0
    args.dropout = 0
    args.my_qa_mask = 0
    args.grad_cp = 0
    args.head_qk = 0
    args.pre_ffn = 0

    from src.utils import get_model_settings

    sets = get_model_settings(args.load_model)
    if sets is not None:
        args.n_layer, args.n_embd, args.ctx_len = sets

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
            enable_progress_bar=False,
            logger=False,
            deterministic=True,
            callbacks=[train_callback(args)],
            
        )
    
    from src.dataset import ValDataset
    test_data = ValDataset(args)
    args.vocab_size = test_data.vocab_size

    data_loader = DataLoader(test_data, shuffle=False, pin_memory=True, batch_size=args.micro_bsz, num_workers=16, persistent_workers=False, drop_last=False)
    
    from src.model import RWKV
    from src.utils import TopK

    model = RWKV(args)
    model.load_state_dict(load_dict, strict=False)
    
    import logging
    pl_loggers = [ logging.getLogger(name) for name in logging.root.manager.loggerDict if 'lightning' in name ]
    for logger in pl_loggers:
        logger.propagate = False
    
    if args.step == "linear":
        perc_vals = np.arange(args.start_k, 0.0, -0.1)
    elif args.step == "exp":
        perc_vals = np.unique(np.round(np.geomspace(args.start_k, 0.01, args.k_num), 2))[::-1]

    print('Iterating over k vals:', perc_vals)

    iter_start_time = time()

    loss_inc = args.loss_inc
    base_loss = trainer.validate(model, dataloaders=data_loader, verbose=False)[0]['val_loss']

    def set_module(model, module_name, new_module):
        # Split the module name to handle nested modules
        module_parts = module_name.split('.')
        current_module = model
        for part in module_parts[:-1]:
            current_module = getattr(current_module, part)
        setattr(current_module, module_parts[-1], new_module)

    def find_k_val(model, trainer, data_loader, module_name, perc_vals, base_loss, loss_inc, n_embd, perc_hist, step='linear'):
        res_loss, res_perc = base_loss, 1.0
        init_perc = 1.0
        module_type = '_'.join(module_name.split('.')[-3:])
        if module_type in perc_hist.keys():
            values, counts = np.unique(perc_hist[module_type], return_counts=True)
            init_perc = values[np.argmax(counts)]
        if step == 'exp' or init_perc == 1.0 or init_perc == perc_vals[0]:
            for cur_perc in perc_vals:
                set_module(model, module_name, TopK(int(n_embd * cur_perc)))
                cur_loss = trainer.validate(model, dataloaders=data_loader, verbose=False)[0]['val_loss']
                if cur_loss / base_loss < loss_inc:
                    res_loss, res_perc = cur_loss, cur_perc
                else:
                    break
        else:
            init_ind = np.where(perc_vals == init_perc)[0][0]
            cur_perc = init_perc
            set_module(model, module_name, TopK(int(n_embd * cur_perc)))
            cur_loss = trainer.validate(model, dataloaders=data_loader, verbose=False)[0]['val_loss']
            if cur_loss / base_loss < loss_inc:
                res_loss, res_perc = cur_loss, cur_perc
                for i in range(init_ind + 1, len(perc_vals)):
                    cur_perc = perc_vals[i]
                    set_module(model, module_name, TopK(int(n_embd * cur_perc)))
                    cur_loss = trainer.validate(model, dataloaders=data_loader, verbose=False)[0]['val_loss']
                    if cur_loss / base_loss < loss_inc:
                        res_loss, res_perc = cur_loss, cur_perc
                    else:
                        break
            else:
                for i in range(init_ind - 1, -1, -1):
                    cur_perc = perc_vals[i]
                    set_module(model, module_name, TopK(int(n_embd * cur_perc)))
                    cur_loss = trainer.validate(model, dataloaders=data_loader, verbose=False)[0]['val_loss']
                    if cur_loss / base_loss < loss_inc:
                        res_loss, res_perc = cur_loss, cur_perc
                        break
                    else:
                        continue
        set_module(model, module_name, TopK(int(n_embd * res_perc)))
        perc_hist[module_type] = perc_hist.get(module_type, []) + [res_perc]
        print(f"{module_name}, {res_perc:.4f}")
        return res_loss, res_perc
        
    perc_hist = dict()
    for i, block in enumerate(model.blocks):
        for name in ['key', 'rec', 'val']:
            module_name = f'blocks.{i}.att.sparsing_fns.{name}'
            base_loss, res_perc = find_k_val(model, trainer, data_loader, 
                                            module_name, perc_vals, base_loss, loss_inc, 
                                            args.n_embd, perc_hist, args.step)
            
        module_name = f'blocks.{i}.att.sparsing_fns.out'
        base_loss, res_perc = find_k_val(model, trainer, data_loader, 
                                        module_name, perc_vals, base_loss, loss_inc, 
                                        args.n_embd, perc_hist, args.step)
        
        for name in ['rec', 'key']:
            module_name = f'blocks.{i}.ffn.sparsing_fns.{name}'
            base_loss, res_perc = find_k_val(model, trainer, data_loader,
                                             module_name, perc_vals, base_loss, loss_inc, 
                                            args.n_embd, perc_hist, args.step)

        module_name = f'blocks.{i}.ffn.sparsing_fns.val'
        base_loss, res_perc = find_k_val(model, trainer, data_loader,
                                        module_name, perc_vals, base_loss, loss_inc, 
                                        args.dim_ffn, perc_hist, args.step)
    
    by_spars_fun = dict()
    for name, module in model.named_modules():

        if isinstance(module, TopK):
            type_name = '.'.join(name.split('.')[2:])
            if 'ffn' in name and 'val' in name:
                perc = module.k.item() / args.dim_ffn
            else:
                perc = module.k.item() / args.n_embd
            by_spars_fun[type_name] = by_spars_fun.get(type_name, []) + [perc]
        
    print(by_spars_fun)
    print('Average K%:', sum([sum(value) for value in by_spars_fun.values()]) / sum([len(value) for value in by_spars_fun.values()])) 
    for key, value in by_spars_fun.items():
        print(key, sum(value) / len(value))

    model_name = args.load_model.split('/')[-1].split('.')[0] + f'_sparse_topk-start_{args.start_k}-step_{args.step}'
    if args.step == "exp":
        model_name = model_name + f'{args.k_num}'
    if args.loss_inc != 1.001:
        model_name = model_name + f'-loss_inc_{args.loss_inc}'
    model_name = model_name + '.pth'
    torch.save(model.state_dict(), os.path.join(args.save_dir, model_name))

    print('Model saved:', model_name)
    
    print('Total time:', (time() - iter_start_time) // 3600, 'hours', (time() - iter_start_time) % 3600 // 60, 'minutes')
        
    
    
    
