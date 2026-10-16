"""Phase 8 — concept drift / adaptive updating implementation.

Implements the design locked in experiments/phase8_adaptive_drift/DESIGN_BEFORE_EXECUTION.md.
Nothing in this package is invoked at import time; every function that performs real training
(GPU/CPU forward+backward passes) is written to be called explicitly by a scripts/train_phase8_*.py
entry point, never as a side effect of importing the module.
"""
