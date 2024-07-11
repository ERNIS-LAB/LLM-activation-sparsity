from src.utils import TopK, HardShrink
import torch
from tqdm import tqdm

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

        # self.lower_bounds = torch.cat([self.lower_bounds, torch.mean(torch.tensor([row[row != 0].min() for row in torch.flatten(output, 0, -2)])).unsqueeze(0)])
        

    def close(self):
        self.hook.remove()

def topk2hs(model_hs, model_topk, init_iterator):
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
        for x, y in tqdm(init_iterator):
            model_topk(x.to('cuda'))
    
    for name, module in model_hs.named_modules():
        if name in hooks.keys():
            module.reinit(hooks[name].lower_bounds.mean().type_as(module.thresholds.data))

    return model_hs

if __name__ == '__main__':
    import argparse, os

    os.environ["RWKV_JIT_ON"] = "0"
    os.environ["RWKV_T_MAX"] = "1024"
    os.environ["RWKV_FLOAT_MODE"] = "bf32"

    parser = argparse.ArgumentParser()

    parser.add_argument("--load_model", default="", type=str)  # full path, with .pth
    parser.add_argument("--data_file", default="", type=str)

    args = parser.parse_args()
    args.n_layer = 12
    args.n_embd = 768
    args.vocab_size = 50277
    args.ctx_len = 1024
    args.dropout = 0
    args.head_qk = 0
    args.my_pos_emb = 0
    args.pre_ffn = 0
    args.grad_cp = 0

    args.sparsity = {'sparsing_fn': 'hardshrink', 'hardshrink_lambda': 0.0}

    from src.utils import get_model_settings
    sets = get_model_settings(args.load_model)
    if sets is not None:
        args.n_layer, args.n_embd, args.ctx_len = sets

    from src.model import RWKV
    model_hs = RWKV(args)
    

    model_dict = torch.load(args.load_model, map_location='cpu')
    model_hs.load_state_dict(model_dict, strict=False)

    args.sparsity = {'sparsing_fn': 'topk'}
    model_topk = RWKV(args)
    model_topk.load_state_dict(model_dict, strict=False)
    model_topk.to('cuda')

    if os.environ['RWKV_FLOAT_MODE'] == 'fp16':
        model_topk = model_topk.half()
        model_hs = model_hs.half()
    elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
        model_topk = model_topk.bfloat16()
        model_hs = model_hs.bfloat16()

    from src.dataset import ValDataset
    from torch.utils.data import DataLoader

    args.micro_bsz = 16
    test_data = ValDataset(args)
    args.vocab_size = test_data.vocab_size

    iterator = DataLoader(test_data, shuffle=True, pin_memory=True, batch_size=args.micro_bsz, num_workers=16, persistent_workers=False, drop_last=False)

    import time
    start_time = time.time()
    topk2hs(model_hs, model_topk, iterator)
    print('total time:', time.time() - start_time)
    torch.save(model_hs.state_dict(), args.load_model.replace('.pth', '_hs.pth'))