"""Planned dataset-contract tests.

The finalized tests will verify:
- no sample crosses an asset boundary;
- target dates do not overlap across splits;
- normalization statistics exclude the target row;
- input and target token alignment is exact.

Model-contract tests will separately verify that every tokenizer parameter stays
frozen and receives no gradient during S1 training.

They will also verify that the predictor exposes exactly 525,312 trainable
parameters and that all of them belong to ``head.proj_s1``.
"""
