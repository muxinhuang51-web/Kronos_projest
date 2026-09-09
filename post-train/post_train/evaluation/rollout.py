"""Autoregressive rollout and downstream full-model evaluation boundaries.

Observational metrics (D-005, D-020) that are recorded for experimental
impact analysis but do not contribute to training loss or checkpoint
selection.
"""

# Placeholder for future implementation of:
#   - s2_degradation: compare s2 token prediction quality vs baseline
#   - full_kline_error: auto-regressive K-line reconstruction error
#   - rank_ic: cross-sectional Rank IC
#   - strategy_return: simple strategy backtest return
#   - five_step_rollout: multi-step autoregressive quality
#   - multi_step_trading_value: trading value from multi-step predictions
#   - ohlc_violation_rate: rate of OHLC constraint violations
