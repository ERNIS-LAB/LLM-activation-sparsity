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
    parser.add_argument("--vocab_size", default=50277, type=int)
    parser.add_argument("--save_dir", default="out", type=str)

    parser.add_argument("--micro_bsz", default=16, type=int)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--ctx_len", default=1024, type=int)

    parser.add_argument("--loss_inc", default=1.001, type=float)
    parser.add_argument("--start_k", default=1.0, type=float)
    parser.add_argument("--step", default="linear", type=str)
    parser.add_argument("--k_num", default=8, type=int)

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    # Additional args setup
    args.my_testing = ''
    args.my_pos_emb = 0
    args.dropout = 0
    args.my_qa_mask = 0
    args.grad_cp = 0
    args.head_qk = 0
    args.pre_ffn = 0

    # Model settings
    from src.utils import get_model_settings
    sets = get_model_settings(args.load_model)
    if sets is not None:
        args.n_layer, args.n_embd, args.ctx_len = sets

    # Environment setup
    os.environ["RWKV_JIT_ON"] = "0"
    if "deepspeed_stage_3" in args.strategy:
        os.environ["RWKV_JIT_ON"] = "0"
    os.environ["RWKV_FLOAT_MODE"] = args.precision

    # CUDA setup
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # Precision setup
    if "32" in args.precision:
        args.precision = 32
    elif args.precision == "fp16":
        args.precision = 16
    else:
        args.precision = "bf16"
    
    os.environ["RWKV_T_MAX"] = str(args.ctx_len)

    # Random seed
    from pytorch_lightning import seed_everything
    if args.random_seed >= 0:
        print(f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n" * 3)
        seed_everything(args.random_seed)

    # Load model
    load_dict = torch.load(args.load_model, map_location="cpu")
    load_keys = list(load_dict.keys())
    for k in load_keys:
        if k.startswith('_forward_module.'):
            load_dict[k.replace('_forward_module.','')] = load_dict[k]
            del load_dict[k]

    # Setup trainer and data
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

    data_loader = DataLoader(
        test_data, 
        shuffle=False, 
        pin_memory=True, 
        batch_size=args.micro_bsz, 
        num_workers=16, 
        persistent_workers=False, 
        drop_last=False
    )

    # Initialize model
    from src.model import RWKV
    args.sparsity = {'sparsing_fn': 'hardshrink', 'hardshrink_lambda': 0.0}
    model = RWKV(args)
    model.load_state_dict(load_dict, strict=False)
    model = model.to(trainer.strategy.root_device)
    print(f"Model device: {model.device}")
    
    # Disable Lightning logging
    import logging
    pl_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict if 'lightning' in name]
    for logger in pl_loggers:
        logger.propagate = False

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

    def collect_activation_stats(model, data_loader, target_module, percentiles):
        hook = ActivationHook(percentiles)
        handle = target_module.register_forward_hook(hook)
        
        model.eval()
        with torch.no_grad():
            for x, _ in data_loader:
                model(x.to(model.device))
                
        handle.remove()
        return hook

    def find_threshold_val(model, trainer, data_loader, module_name, perc_vals, base_loss, loss_inc, thresh_hist, step='linear'):
        # Get the module
        parts = module_name.split('.')
        module = model
        for part in parts:
            module = getattr(module, part)
            
        # Collect activation statistics
        # print(f"Collecting activation statistics for {module_name}...")
        model = model.to(trainer.strategy.root_device)
        hook = collect_activation_stats(model, data_loader, module, perc_vals)
        # print(f"Collected activation statistics for {module_name}")
        thresholds = hook.get_percentile_thresholds()
        
        module_type = '_'.join(module_name.split('.')[-3:])
        res_loss, res_perc = base_loss, perc_vals[0]  # Start with highest percentile
        
        # Initialize with most common percentile if available
        init_perc = 1.0
        if module_type in thresh_hist:
            values, counts = np.unique(thresh_hist[module_type], return_counts=True)
            init_perc = values[np.argmax(counts)]
        
        if step == 'exp' or init_perc == 1.0 or init_perc == perc_vals[0]:
            # Try each percentile from start
            for i, perc in enumerate(perc_vals):
                if perc == 1.0: 
                    continue
                module.reinit(thresholds[i])
                cur_loss = trainer.validate(model, dataloaders=data_loader, verbose=False)[0]['val_loss']
                if cur_loss / base_loss < loss_inc:
                    res_loss, res_perc = cur_loss, perc
                else:
                    break
        else:
            # Start from most common percentile
            init_idx = np.where(perc_vals == init_perc)[0][0]
            module.reinit(thresholds[init_idx])
            cur_loss = trainer.validate(model, dataloaders=data_loader, verbose=False)[0]['val_loss']
            
            if cur_loss / base_loss < loss_inc:
                res_loss, res_perc = cur_loss, init_perc
                # Try higher sparsity (lower percentiles)
                for i in range(init_idx + 1, len(perc_vals)):
                    module.reinit(thresholds[i])
                    cur_loss = trainer.validate(model, dataloaders=data_loader, verbose=False)[0]['val_loss']
                    if cur_loss / base_loss < loss_inc:
                        res_loss, res_perc = cur_loss, perc_vals[i]
                    else:
                        break
            else:
                # Try lower sparsity (higher percentiles)
                for i in range(init_idx - 1, -1, -1):
                    module.reinit(thresholds[i])
                    cur_loss = trainer.validate(model, dataloaders=data_loader, verbose=False)[0]['val_loss']
                    if cur_loss / base_loss < loss_inc:
                        res_loss, res_perc = cur_loss, perc_vals[i]
                        break
        
        # Set final threshold and update history
        final_idx = np.where(perc_vals == res_perc)[0][0]
        module.reinit(thresholds[final_idx])
        thresh_hist[module_type] = thresh_hist.get(module_type, []) + [res_perc]
        print(f"{module_name}, percentile: {res_perc:.4f}")
        return res_loss, res_perc

    # Main initialization loop
    iter_start_time = time()
    
    # Convert percentile values to thresholds
    perc_vals = np.arange(args.start_k, 0.0, -0.1) if args.step == "linear" else \
                np.unique(np.round(np.geomspace(args.start_k, 0.01, args.k_num), 2))[::-1]
    
    print('Starting threshold optimization with perc_vals:', perc_vals)
    loss_inc = args.loss_inc
    base_loss = trainer.validate(model, dataloaders=data_loader, verbose=False)[0]['val_loss']
    
    thresh_hist = dict()
    for i, block in enumerate(model.blocks):
        # Process attention layers
        for name in ['key', 'rec', 'val']:
            module_name = f'blocks.{i}.att.sparsing_fns.{name}'
            base_loss, res_perc = find_threshold_val(model, trainer, data_loader, 
                                         module_name, perc_vals, base_loss, loss_inc, thresh_hist, args.step)
            
        # Process attention output
        module_name = f'blocks.{i}.att.sparsing_fns.out'
        base_loss, res_perc = find_threshold_val(model, trainer, data_loader, 
                                     module_name, perc_vals, base_loss, loss_inc, thresh_hist, args.step)
        
        # Process FFN layers
        for name in ['rec', 'key']:
            module_name = f'blocks.{i}.ffn.sparsing_fns.{name}'
            base_loss, res_perc = find_threshold_val(model, trainer, data_loader,
                                         module_name, perc_vals, base_loss, loss_inc, thresh_hist, args.step)

        module_name = f'blocks.{i}.ffn.sparsing_fns.val'
        base_loss, res_perc = find_threshold_val(model, trainer, data_loader,
                                     module_name, perc_vals, base_loss, loss_inc, thresh_hist, args.step)

    # Print statistics and save model
    print('\nFinal threshold statistics:')
    for module_type, thresholds in thresh_hist.items():
        avg_thresh = sum(thresholds) / len(thresholds)
        print(f'{module_type}: {avg_thresh:.4f}')

    model_name = args.load_model.split('/')[-1].split('.')[0] + f'_sparse_hs-start_{args.start_k}-step_{args.step}'
    if args.step == "exp":
        model_name = model_name + f'{args.k_num}'
    if args.loss_inc != 1.001:
        model_name = model_name + f'-loss_inc_{args.loss_inc}'
    model_name = model_name + '.pth'
    
    torch.save(model.state_dict(), os.path.join(args.save_dir, model_name))
    print('Model saved:', model_name)
    print('Total time:', (time() - iter_start_time) // 3600, 'hours', 
          (time() - iter_start_time) % 3600 // 60, 'minutes')
