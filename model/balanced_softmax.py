"""
Balanced Softmax (BS) for Long-Tail Classification

Reference:
    Ren et al., "Balanced Meta-Softmax for Long-Tailed Visual Recognition", NeurIPS 2020
    
Formula:
    BS(z_i) = n_i * exp(z_i) / Σ_j (n_j * exp(z_j))
    
where n_i is the number of training samples for class i.
"""

import numpy as np
from typing import List


def balanced_softmax_probs(logits: np.ndarray, cls_num_list: List) -> np.ndarray:

    cls_num = np.array(cls_num_list, dtype=np.float64)
    
    cls_num = np.maximum(cls_num, 1e-8)
    
    log_prior = np.log(cls_num)
    

    adjusted_logits = logits - log_prior
    
    adjusted_logits = adjusted_logits - np.max(adjusted_logits, axis=-1, keepdims=True)
    exp_logits = np.exp(adjusted_logits)
    bs_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    return bs_probs
