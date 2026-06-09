# @lint-ignore-every LICENSELINT
# Adapted from MMAudio (https://github.com/hkchengrex/MMAudio), licensed under the MIT License.
# Includes portions from black-forest-labs/flux
# (https://github.com/black-forest-labs/flux, licensed under the Apache License 2.0)
# and lucidrains/rotary-embedding-torch
# (https://github.com/lucidrains/rotary-embedding-torch, licensed under the MIT License).
# See the NOTICE.txt file in the root of this source tree for the upstream licenses.
#
# Modifications:
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import torch
from einops import rearrange
from torch import Tensor

# Ref: https://github.com/black-forest-labs/flux/blob/main/src/flux/math.py
# Ref: https://github.com/lucidrains/rotary-embedding-torch


def compute_rope_rotations(
    length: int,
    dim: int,
    theta: int,
    *,
    freq_scaling: float = 1.0,
    device: Union[torch.device, str] = "cpu",
) -> Tensor:
    assert dim % 2 == 0

    with torch.amp.autocast(device_type="cuda", enabled=False):
        pos = torch.arange(length, dtype=torch.float32, device=device)
        inv_freq_exponent = torch.arange(
            0, dim, 2, dtype=torch.float32, device=device
        ).div(dim)
        freqs = torch.pow(theta, -inv_freq_exponent)
        freqs *= freq_scaling

        rot = torch.einsum("..., f -> ... f", pos, freqs)
        rot = torch.stack(
            [torch.cos(rot), -torch.sin(rot), torch.sin(rot), torch.cos(rot)], dim=-1
        )
        rot = rearrange(rot, "n d (i j) -> 1 n d i j", i=2, j=2)
        return rot


def apply_rope(x: Tensor, rot: Tensor) -> Tensor:
    with torch.amp.autocast(device_type="cuda", enabled=False):
        _x = x.float()
        _x = _x.view(*_x.shape[:-1], -1, 1, 2)
        x_out = rot[..., 0] * _x[..., 0] + rot[..., 1] * _x[..., 1]
        return x_out.reshape(*x.shape).to(dtype=x.dtype)
