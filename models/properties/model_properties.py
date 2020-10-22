from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelProperties:
    model_name: str
    embedding: Optional[str]
    bp_emb_size: Optional[int]
    n_epochs: int
    batch_size: int
    initial_lr: float
    step_size: int
    gamma: float
