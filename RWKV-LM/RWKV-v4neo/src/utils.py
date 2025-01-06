import json, time, random, os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

time_slot = {}
time_ref = time.time_ns()

def record_time(name):
    if name not in time_slot:
        time_slot[name] = 1e20
    tt = (time.time_ns() - time_ref) / 1e9
    if tt < time_slot[name]:
        time_slot[name] = tt

class TOKENIZER():
    def __init__(self, WORD_NAME, UNKNOWN_CHAR='\ue083'):
        if 'list' in str(type(WORD_NAME)):
            self.charMode = False
            if WORD_NAME[0] == WORD_NAME[1]:
                from transformers import PreTrainedTokenizerFast
                self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=WORD_NAME[0])
            else:
                from transformers import GPT2TokenizerFast
                self.tokenizer = GPT2TokenizerFast(WORD_NAME[0], WORD_NAME[1])
            self.vocab_size = len(self.tokenizer)
        else:
            self.charMode = True
            with open(WORD_NAME + '.json', "r", encoding="utf-16") as result_file:
                self.word_table = json.load(result_file)

            self.vocab_size = len(self.word_table)

            self.stoi = {v: int(k) for k, v in self.word_table.items()}
            self.itos = {int(k): v for k, v in self.word_table.items()}

            self.UNKNOWN_CHAR = self.stoi[UNKNOWN_CHAR]

    def refine_context(self, context):
        context = context.strip().split('\n')
        for c in range(len(context)):
            context[c] = context[c].strip().strip('\u3000').strip('\r')
        context = list(filter(lambda c: c != '', context))
        context = '\n' + ('\n'.join(context)).strip()
        if context == '':
            context = '\n'
        return context

    def sample_logits(self, out, x, ctx_len, temperature=1.0, top_p_usual=None, top_p_newline=None):
        # out[self.UNKNOWN_CHAR] = -float('Inf')
        lastChar = int(x[-1])

        probs = F.softmax(out, dim=-1)

        if self.charMode:
            if self.itos[lastChar] == '\n':
                top_p = top_p_newline
            else:
                top_p = top_p_usual
        else:
            top_p = top_p_usual

        if os.environ["RWKV_RUN_DEVICE"] == "cpu":
            probs = probs.numpy()
            sorted_probs = np.sort(probs)[::-1]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
            probs[probs < cutoff] = 0
            if temperature != 1.0:
                probs = probs.pow(1.0 / temperature)
            probs = probs / np.sum(probs)
            out = np.random.choice(a=len(probs), p=probs)
            return out
        else:
            sorted_probs = torch.sort(probs, descending=True)[0]
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
            probs[probs < cutoff] = 0
            if temperature != 1.0:
                probs = probs.pow(1.0 / temperature)
            out = torch.multinomial(probs, num_samples=1)[0]
            return out

def MaybeIsPrime(number):
    if FermatPrimalityTest(number) and MillerRabinPrimalityTest(number):
        return True
    else:
        return False


def FermatPrimalityTest(number):
    if number > 1:
        for time in range(3):
            randomNumber = random.randint(2, number) - 1
            if pow(randomNumber, number - 1, number) != 1:
                return False
        return True
    else:
        return False


def MillerRabinPrimalityTest(number):
    if number == 2:
        return True
    elif number == 1 or number % 2 == 0:
        return False
    oddPartOfNumber = number - 1
    timesTwoDividNumber = 0
    while oddPartOfNumber % 2 == 0:
        oddPartOfNumber = oddPartOfNumber // 2
        timesTwoDividNumber = timesTwoDividNumber + 1

    for time in range(3):
        while True:
            randomNumber = random.randint(2, number) - 1
            if randomNumber != 0 and randomNumber != 1:
                break

        randomNumberWithPower = pow(randomNumber, oddPartOfNumber, number)

        if (randomNumberWithPower != 1) and (randomNumberWithPower != number - 1):
            iterationNumber = 1

            while (iterationNumber <= timesTwoDividNumber - 1) and (randomNumberWithPower != number - 1):
                randomNumberWithPower = pow(randomNumberWithPower, 2, number)
                iterationNumber = iterationNumber + 1
            if randomNumberWithPower != (number - 1):
                return False

    return True

class TopK(torch.nn.Module):
    def __init__(self, k: int = 1):
        super().__init__()
        self.k = nn.Parameter(torch.tensor(k), requires_grad=False)

    def forward(self, x: torch.Tensor):
        inv_k = x.size(-1) - self.k
        _, indices = torch.topk(torch.abs(x), inv_k, dim=-1, largest=False)
        x = x.scatter(dim=-1, index=indices, value=0)

        return x
    
def calc_sparsity(net_activation):
    """
    Calculate the sparsity of a network activation tensor.

    Parameters:
        net_activation (torch.Tensor): The network activation tensor.

    Returns:
        float: The percentage of zero values in the tensor.
    """
    with torch.no_grad():
        num_nonzero = torch.count_nonzero(net_activation)
        percent_zero = 100.0 * (1.0 - (num_nonzero / net_activation.numel()))
    return percent_zero.item()

def calc_threshold_avg(model):
    """
    Calculate the average threshold of a model's HardShrink layers.

    Parameters:
        model (torch.nn.Module): The model to calculate the average threshold for.

    Returns:
        float: The average threshold of the model's HardShrink layers.
    """
    with torch.no_grad():
        num_hardshrink = 0
        total_threshold = 0.0
        for module in model.modules():
            if isinstance(module, HardShrink):
                num_hardshrink += 1
                total_threshold += module.thresholds.mean().item()
        avg_threshold = total_threshold / num_hardshrink
    return avg_threshold

class Binarize(torch.autograd.Function): # same as a spike function
    """
    Spike function with derivative of arctan surrogate gradient.
    Featured in Fang et al. 2020/2021.
    """
 
    @staticmethod
    def forward(ctx, x, width):
        ctx.save_for_backward(x, width)
        out = x.gt(0).type_as(x)


        return out
 
    @staticmethod
    def backward(ctx, grad_output):
        x, width = ctx.saved_tensors
        grad_input = grad_output.clone()

        sg = 1 / (1 + width * x * x)
           
        return grad_input * sg, None
    
class HardShrink(torch.nn.Module):
    def __init__(self, lambd: float = 0.5, embed_dim: int = 768):
        super().__init__()
        self.thresholds = torch.nn.Parameter(torch.full((embed_dim, ), lambd))
        self.width = torch.tensor(5.0)
        self.binarize = Binarize.apply

    def forward(self, x: torch.Tensor):
        pos_mask = self.binarize(x - self.thresholds, self.width)
        neg_mask = self.binarize(-x - self.thresholds, self.width)
        
        return x * pos_mask + x * neg_mask
        # return x * (x >= self.thresholds).type_as(x) + x * (x <= -self.thresholds).type_as(x)
    
    def reinit(self, x):
        device = self.thresholds.device
        init_type = self.thresholds.dtype
        if torch.is_tensor(x) and sum(x.shape) != 0:
            self.thresholds.data = x.detach().clone()
        else:
            self.thresholds.data = torch.full(self.thresholds.data.size(), x)
        self.thresholds.data = self.thresholds.data.type(init_type).to(device)

def get_sparsing_fn(args, embed_dim: int = 768):
    if args.sparsity['sparsing_fn'] == "topk":
        return TopK(int(embed_dim * args.sparsity.get('topk_pc', 1.0)))
    elif args.sparsity['sparsing_fn'] == "hardshrink":
        return HardShrink(lambd=args.sparsity.get('hardshrink_lambda', 0.0), embed_dim=embed_dim)
    elif args.sparsity['sparsing_fn'] == "relu":
        return torch.nn.ReLU()
    else:
        raise ValueError(f"Unknown sparsing_fn: {args.sparsity.sparsing_fn}")
    
def get_regularization_loss(regularization):
    if regularization == "l1":
        return lambda x: torch.sum(torch.abs(x))
    elif regularization == "l2":
        return lambda x: torch.sum(x ** 2)
    elif regularization == "hoyer":
        return lambda x: torch.sum(torch.abs(x)) ** 2 / torch.sum(x ** 2)
    else:
        raise ValueError(f"Unknown regularization type: {regularization}")
    
def get_model_settings(model_name):
    """
    Get the settings for a model based on its name.

    Args:
        model_name (str): The name of the model.
        
    Returns:
        tuple or None: A tuple containing the settings for the model in the format (num_layers, emb_dim, ctx_len).
        Returns None if the model name is not recognized.
    """
    if 'RWKV-4-Pile-169M' in model_name:
        return 12, 768, 1024
    elif 'RWKV-4-Pile-430M' in model_name:
        return 24, 1024, 1024
    elif 'RWKV-4-Pile-1B5' in model_name:
        return 24, 2048, 1024
    elif 'RWKV-4-Pile-3B' in model_name:
        return 32, 2560, 1024
    elif 'RWKV-4-Pile-7B' in model_name:
        return 32, 4096, 1024
    else:
        return None
    