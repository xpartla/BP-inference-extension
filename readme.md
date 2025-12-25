# Rust ONNX inference engine for MCI prediction
Standalone, "production-ready" Rust library for running a LightGBM model exported to ONNX. The engine consumes frozen preprocessing artifacts generated during model training. Contains a parity test with Python inference.

## Overview
- Loads a trained LightGBM model exported to ONNX
- Re-implements training-time preprocessing steps in Rust
  - Numerical imputation, outlier handling, scaling
  - Categorical encoding
  - Feature ordering
- Produces deterministic predictions
- Standalone inference (no other dependencies)

## Testing
Parity tests validate correctness by comparing Rust inference outputs against Python-generated ground truth. It guarantees identical predicted labels, and probability difference < 1e-4, meaning the inference is numerically equivalent.

## Related Work
- [Bachelor's Thesis repo](https://github.com/xpartla/Bakalarka)
- [Model export service](https://github.com/xpartla/BP-model-export)

## Notes
Research-oriented engineering project, not intended for clinical deployment.