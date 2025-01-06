# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn as nn

from metaseq.modules import Linear
from metaseq import utils


def FeedForward(x, fc1, activation_fn, fc2, dropout_module, sparsing_fns, sparse_stats=None):
    """
    Feedforward network consisting of two linear layers (fc1, fc2), where activation_fn is applied
    between the two layers and dropout_module is applied at the end.
    """
    # apex fused bias gelu is not yet supported with megatron model parallel
    # TODO [namangoyal]: Find better way to do this
    model_parallel = not isinstance(fc1, nn.Linear) and not isinstance(fc1, Linear)
    x = sparsing_fns['up_proj'](x)
    if sparse_stats is not None:
        sparse_stats['up_proj'] = utils.calc_sparsity(x)
    if model_parallel:
        # here, we do the bias computation inside fc1 and fc2 AND gather_output
        x = activation_fn(x, fc1(x)[0], model_parallel=True)
        x = sparsing_fns['down_proj'](x)
        if sparse_stats is not None:
            sparse_stats['down_proj'] = utils.calc_sparsity(x)
        x, _ = fc2(x)
    else:
        x = activation_fn(x, fc1(x), model_parallel=False)
        x = sparsing_fns['down_proj'](x)
        if sparse_stats is not None:
            sparse_stats['down_proj'] = utils.calc_sparsity(x)
        x = fc2(x)
    x = dropout_module(x)
    return x
