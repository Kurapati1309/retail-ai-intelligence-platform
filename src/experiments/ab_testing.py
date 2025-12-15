"""Simple A/B-style comparison framework.

Use this to compare:
- baseline model vs tuned model
- feature set A vs feature set B
- threshold strategies for fraud detection

This aligns to: 'Design and conduct experiments to test hypotheses and validate model assumptions'.
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class ABResult:
    metric_a: float
    metric_b: float
    delta: float
    winner: str

def compare(metric_a: float, metric_b: float, higher_is_better=True) -> ABResult:
    delta = metric_b - metric_a
    if higher_is_better:
        winner = "B" if metric_b > metric_a else "A"
    else:
        winner = "B" if metric_b < metric_a else "A"
    return ABResult(metric_a=float(metric_a), metric_b=float(metric_b), delta=float(delta), winner=winner)
