"""Clinical + waveform evaluation for ECG+PPG -> ABP reconstruction.

Metrics follow the BP literature convention (see plan/notes.md):
- waveform: MAE, RMSE (mmHg), per-segment Pearson r;
- derived BP values: SBP = segment max, DBP = segment min, MAP = segment mean;
- AAMI: mean error (ME) and std of error (SDE); pass = |ME|<=5 and SDE<=8 mmHg;
- BHS: cumulative % of |error| within 5/10/15 mmHg -> grade A/B/C.

These are the GATE for ABP reconstruction quality (more honest than sample
MSE, which hides DC-offset / systematic SBP-DBP bias).
"""

from typing import Dict

import torch


def _pearson(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Per-segment Pearson r. pred/true: (B, L) -> (B,)."""
    p = pred - pred.mean(dim=1, keepdim=True)
    t = true - true.mean(dim=1, keepdim=True)
    num = (p * t).sum(dim=1)
    den = torch.sqrt((p * p).sum(dim=1) * (t * t).sum(dim=1)).clamp_min(1e-8)
    return num / den


def segment_bp(wave: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Per-segment SBP/DBP/MAP. wave: (B, L) mmHg."""
    return {
        "SBP": wave.amax(dim=1),
        "DBP": wave.amin(dim=1),
        "MAP": wave.mean(dim=1),
    }


def waveform_metrics(pred: torch.Tensor, true: torch.Tensor) -> Dict[str, float]:
    """pred/true: (B, L) mmHg."""
    mae = (pred - true).abs().mean().item()
    rmse = torch.sqrt(((pred - true) ** 2).mean()).item()
    pearson = _pearson(pred, true).mean().item()
    return {"MAE": mae, "RMSE": rmse, "Pearson": pearson}


def aami(pred_vals: torch.Tensor, true_vals: torch.Tensor) -> Dict[str, float]:
    """AAMI: mean error and std of error (mmHg). pred/true: (B,)."""
    err = (pred_vals - true_vals).float()
    me = err.mean().item()
    sde = err.std(unbiased=True).item() if err.numel() > 1 else 0.0
    return {"ME": me, "SDE": sde, "pass": float(abs(me) <= 5.0 and sde <= 8.0)}


def bhs(pred_vals: torch.Tensor, true_vals: torch.Tensor) -> Dict[str, object]:
    """BHS grading from cumulative |error| percentages."""
    abs_err = (pred_vals - true_vals).abs().float()
    p5 = (abs_err <= 5).float().mean().item() * 100
    p10 = (abs_err <= 10).float().mean().item() * 100
    p15 = (abs_err <= 15).float().mean().item() * 100
    if p5 >= 60 and p10 >= 85 and p15 >= 95:
        grade = "A"
    elif p5 >= 50 and p10 >= 75 and p15 >= 90:
        grade = "B"
    elif p5 >= 40 and p10 >= 65 and p15 >= 85:
        grade = "C"
    else:
        grade = "D"
    return {"<=5mmHg": p5, "<=10mmHg": p10, "<=15mmHg": p15, "grade": grade}


def evaluate(pred_mmhg: torch.Tensor, true_mmhg: torch.Tensor) -> Dict[str, dict]:
    """Full report. pred/true: (B, L) mmHg."""
    report: Dict[str, dict] = {"waveform": waveform_metrics(pred_mmhg, true_mmhg)}
    pred_bp = segment_bp(pred_mmhg)
    true_bp = segment_bp(true_mmhg)
    for key in ("SBP", "DBP", "MAP"):
        report[key] = {
            "AAMI": aami(pred_bp[key], true_bp[key]),
            "BHS": bhs(pred_bp[key], true_bp[key]),
        }
    return report


def format_report(report: Dict[str, dict]) -> str:
    w = report["waveform"]
    lines = [
        f"Waveform: MAE={w['MAE']:.3f}  RMSE={w['RMSE']:.3f}  Pearson={w['Pearson']:.4f}  (mmHg)",
    ]
    for key in ("SBP", "DBP", "MAP"):
        a = report[key]["AAMI"]
        b = report[key]["BHS"]
        lines.append(
            f"{key}: ME={a['ME']:+.2f} SDE={a['SDE']:.2f} "
            f"AAMI={'PASS' if a['pass'] else 'FAIL'} | "
            f"BHS {b['grade']} (<=5:{b['<=5mmHg']:.0f}% <=10:{b['<=10mmHg']:.0f}% <=15:{b['<=15mmHg']:.0f}%)"
        )
    return "\n".join(lines)
