# @lint-ignore-every LICENSELINT
# Adapted from MMAudio (https://github.com/hkchengrex/MMAudio), licensed under the MIT License.
# Includes portions from facebookresearch/DiT
# (https://github.com/facebookresearch/DiT, licensed under CC BY-NC 4.0)
# and openai/glide-text2im
# (https://github.com/openai/glide-text2im, licensed under the MIT License).
# See the NOTICE.txt file in the root of this source tree for the upstream licenses.
#
# Modifications:
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

# https://github.com/facebookresearch/DiT


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self, dim: int, frequency_embedding_size: int, max_period: int
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.dim = dim
        self.max_period = max_period
        assert dim % 2 == 0, "dim must be even."

        with torch.autocast("cuda", enabled=False):
            freqs = 1.0 / (
                10000
                ** (
                    torch.arange(0, frequency_embedding_size, 2, dtype=torch.float32)
                    / frequency_embedding_size
                )
            )
            freq_scale = 10000 / max_period
            self.freqs = nn.Buffer(freq_scale * freqs, persistent=False)

    def timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py

        args = t[:, None].float() * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb
