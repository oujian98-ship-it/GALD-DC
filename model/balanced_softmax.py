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
    """
    Apply Balanced Softmax adjustment to logits at inference time.
    
    This adjusts the raw logits by the class prior (training sample count),
    effectively compensating for the long-tail distribution bias.
    
    Args:
        logits: Raw logits from classifier, shape (N, C)
                N = number of samples, C = number of classes
        cls_num_list: Number of training samples per class, length C
    
    Returns:
        bs_probs: Balanced Softmax probabilities, shape (N, C)
    """
    # Convert to numpy array and ensure float type
    cls_num = np.array(cls_num_list, dtype=np.float64)
    
    # Avoid division by zero
    cls_num = np.maximum(cls_num, 1e-8)
    
    # Log of class priors (for numerical stability)
    log_prior = np.log(cls_num)
    
    # Adjust logits: logits - log(n_i)
    # This is equivalent to: exp(z_i) / n_i in the softmax numerator
    # for post-hoc correction of CE-trained models.
    adjusted_logits = logits - log_prior
    
    # Apply softmax to adjusted logits
    # Subtract max for numerical stability
    adjusted_logits = adjusted_logits - np.max(adjusted_logits, axis=-1, keepdims=True)
    exp_logits = np.exp(adjusted_logits)
    bs_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    return bs_probs
