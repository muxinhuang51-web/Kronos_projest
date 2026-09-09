"""S1-only training wrapper around the Kronos predictor.

This module exposes deterministic S1 logits for the final position without
executing the stochastic S2 branch and owns the head-only parameter-freezing
policy (D-014).

The official tokenizer is a permanently frozen experiment dependency.  The
wrapper places it in evaluation mode, disables gradients for every tokenizer
parameter, and fails fast if any tokenizer parameter becomes trainable.

Only ``head.proj_s1.weight`` and ``head.proj_s1.bias`` may be trainable.
The wrapper fails fast when the trainable-parameter names or total count
differ from the computed expectation.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

# The Kronos source lives one level above post-train/.
_Kronos_ROOT = Path(__file__).resolve().parents[3]
if str(_Kronos_ROOT) not in sys.path:
    sys.path.insert(0, str(_Kronos_ROOT))

from model.kronos import KronosTokenizer, Kronos  # noqa: E402

logger = logging.getLogger(__name__)


class S1HeadOnlyWrapper:
    """Wrap Kronos tokenizer + predictor for head-only S1 next-token training.

    Forward flow::

        context_x [B, 256, 6]
            → tokenizer.encode(half=True) → (s1_ids, s2_ids)  [B, 256]
            → predictor(s1_ids, s2_ids, stamp)
            → s1_logits [B, 256, vocab_s1]
            → s1_logits[:, -1, :]  [B, vocab_s1]
    """

    def __init__(
        self,
        tokenizer_path: str,
        predictor_path: str,
        device: str = "cuda:0",
    ):
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda:0"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        logger.info("Loading tokenizer from %s", tokenizer_path)
        self.tokenizer: KronosTokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.to(self.device)

        logger.info("Loading predictor from %s", predictor_path)
        self.predictor: Kronos = Kronos.from_pretrained(predictor_path)
        self.predictor.to(self.device)

        # Convenience attributes
        self.vocab_s1: int = 2 ** self.predictor.s1_bits
        self.d_model: int = self.predictor.d_model
        self._expected_trainable: int = self.vocab_s1 * (self.d_model + 1)

        self.freeze_tokenizer()
        self.freeze_predictor_except_s1_head()

        logger.info(
            "S1HeadOnlyWrapper ready: vocab_s1=%d, d_model=%d, trainable=%d",
            self.vocab_s1, self.d_model, self._expected_trainable,
        )

    # ------------------------------------------------------------------
    # Freezing policies
    # ------------------------------------------------------------------

    def freeze_tokenizer(self) -> None:
        """Permanently freeze the tokenizer (D-013)."""
        self.tokenizer.eval()
        for p in self.tokenizer.parameters():
            p.requires_grad = False
        logger.info("Tokenizer frozen.")

    def freeze_predictor_except_s1_head(self) -> None:
        """Freeze all predictor parameters except ``head.proj_s1.*`` (D-014).

        Fails fast if the resulting trainable parameter set does not exactly
        match the expected allowlist or count.
        """
        allowed_prefixes = ("head.proj_s1.weight", "head.proj_s1.bias")
        trainable_names: list[str] = []
        trainable_count = 0

        for name, p in self.predictor.named_parameters():
            p.requires_grad = any(name == prefix for prefix in allowed_prefixes)
            if p.requires_grad:
                trainable_names.append(name)
                trainable_count += p.numel()

        # --- Validation ---
        unexpected = [n for n in trainable_names if not any(n == ap for ap in allowed_prefixes)]
        if unexpected:
            raise RuntimeError(
                f"Unexpected trainable parameters: {unexpected}. "
                f"Only {list(allowed_prefixes)} are allowed."
            )

        if trainable_count != self._expected_trainable:
            raise RuntimeError(
                f"Trainable parameter count mismatch: "
                f"got {trainable_count}, expected {self._expected_trainable} "
                f"(vocab_s1={self.vocab_s1}, d_model={self.d_model})."
            )

        missing = [ap for ap in allowed_prefixes if ap not in trainable_names]
        if missing:
            raise RuntimeError(f"Expected trainable parameters not found: {missing}.")

        logger.info(
            "Predictor frozen: %d trainable params (%s)",
            trainable_count, trainable_names,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, stamp: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return S1 logits for the **last** position of each sequence.

        Args:
            x: ``[B, T, 6]`` normalized OHLCVA context.
            stamp: ``[B, T, 5]`` optional temporal features.

        Returns:
            ``[B, vocab_s1]`` logits for the next-token S1 prediction.
        """
        B, T, _ = x.shape

        with torch.no_grad():
            s1_ids, s2_ids = self.tokenizer.encode(x, half=True)
            # s1_ids: [B, T], s2_ids: [B, T]

        s1_logits, _ = self.predictor(
            s1_ids.to(self.device),
            s2_ids.to(self.device),
            stamp.to(self.device) if stamp is not None else None,
        )
        # s1_logits: [B, T, vocab_s1]

        return s1_logits[:, -1, :]  # [B, vocab_s1]

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    def trainable_parameters(self):
        """Yield only the trainable parameters (for the optimizer)."""
        return filter(lambda p: p.requires_grad, self.predictor.parameters())

    def train(self, mode: bool = True) -> None:
        """Set training mode; tokenizer always stays in eval mode."""
        self.tokenizer.eval()
        self.predictor.train(mode)

    def eval(self) -> None:
        """Set evaluation mode."""
        self.tokenizer.eval()
        self.predictor.eval()
