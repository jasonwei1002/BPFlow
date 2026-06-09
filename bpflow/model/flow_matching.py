# @lint-ignore-every LICENSELINT
# Adapted from Meta WavFlow (wavflow/model/flow_matching.py), itself adapted
# from JiT (https://github.com/LTH14/JiT, MIT) and gle-bellier/flow-matching.
#
# Copied into bpflow (rather than re-exported) for ONE reason: the upstream
# module does ``from torchdiffeq import odeint`` at import time, which is not
# installed and is only needed for the 'adaptive' ODE solver. Here that import
# is made lazy so the default 'euler' path is fully self-contained on CPU.
# The maths is byte-identical to upstream.

import logging
from typing import Callable, Optional

import torch

log = logging.getLogger(__name__)


def log_normal_sample(
    x: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    m: float = 0.0,
    s: float = 1.0,
) -> torch.Tensor:
    """Sample training timesteps t ~ sigmoid(N(m, s)) in (0, 1), one per batch."""
    bs = x.shape[0]
    sample = torch.randn(bs, device=x.device, generator=generator) * s + m
    return torch.sigmoid(sample)


class FlowMatching:
    def __init__(
        self,
        min_sigma: float = 0.0,
        inference_mode: str = "euler",
        num_steps: int = 25,
        prediction_type: str = "x",
        noise_scale: float = 1.0,
        noise_shift: float = 1.0,
    ) -> None:
        super().__init__()
        self.min_sigma = min_sigma
        self.inference_mode = inference_mode
        self.num_steps = num_steps
        self.prediction_type = prediction_type
        self.noise_scale = noise_scale
        self.noise_shift = noise_shift

        assert self.inference_mode in ["euler", "adaptive"]
        assert self.prediction_type in ["x", "v"]
        if self.inference_mode == "adaptive" and num_steps > 0:
            log.info("The number of steps is ignored in adaptive inference mode")

    def shift_timestep(self, t: torch.Tensor) -> torch.Tensor:
        if self.noise_shift == 1.0:
            return t
        return t / (t + self.noise_shift * (1 - t))

    def get_conditional_flow(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        t = t[:, None, None].expand_as(x0)
        return (1 - (1 - self.min_sigma) * t) * x0 + t * x1

    def x_pred_x_loss(self, pred, x0, xt, x1, t) -> torch.Tensor:
        reduce_dim = list(range(1, len(pred.shape)))
        return (pred - x1).pow(2).mean(dim=reduce_dim)

    def x_pred_v_loss(self, pred, x0, xt, x1, t) -> torch.Tensor:
        t_expanded = t[:, None, None].expand_as(pred)
        one_minus_t = (1 - t_expanded).clamp(min=1e-6)
        predicted_v = (pred - xt) / one_minus_t
        target_v = (x1 - xt) / one_minus_t
        reduce_dim = list(range(1, len(pred.shape)))
        return (predicted_v - target_v).pow(2).mean(dim=reduce_dim)

    def v_pred_v_loss(self, pred, x0, xt, x1, t) -> torch.Tensor:
        target_v = x1 - (1 - self.min_sigma) * x0
        reduce_dim = list(range(1, len(pred.shape)))
        return (pred - target_v).pow(2).mean(dim=reduce_dim)

    def v_pred_x_loss(self, pred, x0, xt, x1, t) -> torch.Tensor:
        t_expanded = t[:, None, None].expand_as(pred)
        one_minus_t = 1 - t_expanded
        predicted_x1 = xt + one_minus_t * pred
        reduce_dim = list(range(1, len(pred.shape)))
        return (predicted_x1 - x1).pow(2).mean(dim=reduce_dim)

    def get_x0_xt_c(
        self,
        x1: torch.Tensor,
        t: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ):
        x0 = torch.empty_like(x1).normal_(generator=generator) * self.noise_scale
        t_shifted = self.shift_timestep(t)
        xt = self.get_conditional_flow(x0, x1, t_shifted)
        return x0, x1, xt, t_shifted

    def loss(self, prediction_type: str, loss_type: str, pred, x0, xt, x1, t) -> torch.Tensor:
        """Dispatch to the (prediction_type, loss_type) loss combo.

        Only ``v_pred_v_loss`` accounts for ``min_sigma`` exactly; the three
        "off-diagonal" variants compare against ``x1`` / ``(x1-xt)/(1-t)`` and
        silently assume ``min_sigma == 0`` (true endpoint is ``min_sigma*x0 +
        x1``). They are correct at the default ``min_sigma=0``. Fail loudly
        rather than train on a wrong target if someone sets ``min_sigma>0`` on
        an untested combo.
        """
        key = (prediction_type, loss_type)
        table = {
            ("x", "x"): self.x_pred_x_loss,
            ("x", "v"): self.x_pred_v_loss,
            ("v", "v"): self.v_pred_v_loss,
            ("v", "x"): self.v_pred_x_loss,
        }
        if key not in table:
            raise ValueError(f"Unknown (prediction_type, loss_type) combo: {key}")
        if self.min_sigma != 0.0 and key != ("v", "v"):
            raise ValueError(
                f"min_sigma={self.min_sigma} is only handled exactly by the (v, v) loss; "
                f"combo {key} would train on an incorrect target. Use min_sigma=0 or (v, v)."
            )
        return table[key](pred, x0, xt, x1, t)

    def to_data(self, fn: Callable, x0: torch.Tensor) -> torch.Tensor:
        return self.run_t0_to_t1(fn, x0, 0, 1)

    def run_t0_to_t1(self, fn: Callable, x0: torch.Tensor, t0: float, t1: float) -> torch.Tensor:
        if self.inference_mode == "adaptive":
            from torchdiffeq import odeint  # lazy: only needed for adaptive

            def velocity_fn(t_scalar: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                t_s = self.shift_timestep(t_scalar)
                pred = fn(t_s, x)
                if self.prediction_type == "x":
                    one_minus_t = (1 - t_s).clamp(min=1e-6)
                    return (pred - x) / one_minus_t
                return pred

            result = odeint(
                velocity_fn,
                x0,
                torch.tensor([t0, t1], device=x0.device, dtype=x0.dtype),
            )
            return result[-1]
        elif self.inference_mode == "euler":
            x = x0
            raw_steps = torch.linspace(t0, t1 - self.min_sigma, self.num_steps + 1)
            steps = self.shift_timestep(raw_steps)
            for ti, t in enumerate(steps[:-1]):
                pred = fn(t, x)
                if self.prediction_type == "x":
                    one_minus_t = max(1 - t, 1e-6)
                    v = (pred - x) / one_minus_t
                else:
                    v = pred
                next_t = steps[ti + 1]
                dt = next_t - t
                x = x + dt * v
            return x
        raise ValueError(f"Unknown inference mode: {self.inference_mode}")
