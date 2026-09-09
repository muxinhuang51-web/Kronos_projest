"""S1 cross-entropy, perplexity, accuracy, bit accuracy, and Hamming-distance metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_s1_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    s1_bits: int,
) -> dict[str, float]:
    """Compute all primary S1 evaluation metrics.

    Args:
        logits: ``[N, vocab_s1]`` model logits.
        targets: ``[N]`` ground-truth S1 token IDs.
        s1_bits: Number of bits in the S1 token (e.g. 10).

    Returns:
        Dict with keys ``s1_cross_entropy``, ``s1_perplexity``,
        ``s1_top1_accuracy``, ``s1_top5_accuracy``, ``s1_bit_accuracy``,
        ``s1_hamming_distance``.
    """
    vocab_s1 = 2 ** s1_bits
    N = logits.size(0)

    ce = F.cross_entropy(logits, targets).item()

    pred = logits.argmax(dim=-1)  # [N]
    top1 = (pred == targets).float().mean().item()

    # Top-5 accuracy
    _, top5_indices = logits.topk(min(5, vocab_s1), dim=-1)
    top5 = top5_indices.eq(targets.unsqueeze(-1)).any(dim=-1).float().mean().item()

    # Bit accuracy: decompose both pred and target into s1_bits binary digits,
    # then compare bit-by-bit.
    pred_bits = _token_to_bits(pred, s1_bits)        # [N, s1_bits]
    target_bits = _token_to_bits(targets, s1_bits)   # [N, s1_bits]
    bit_acc = (pred_bits == target_bits).float().mean().item()
    hamming = (pred_bits != target_bits).float().sum().item() / N

    return {
        "s1_cross_entropy": ce,
        "s1_perplexity": float(torch.exp(torch.tensor(ce)).item()),
        "s1_top1_accuracy": top1,
        "s1_top5_accuracy": top5,
        "s1_bit_accuracy": bit_acc,
        "s1_hamming_distance": hamming,
    }


def _token_to_bits(ids: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Convert integer token IDs to binary bit vectors.

    Args:
        ids: ``[N]`` integer token IDs in ``[0, 2^n_bits - 1]``.
        n_bits: Number of bits.

    Returns:
        ``[N, n_bits]`` binary tensor, MSB first.
    """
    bits = []
    for b in reversed(range(n_bits)):
        bits.append((ids >> b) & 1)
    return torch.stack(bits, dim=-1).float()
