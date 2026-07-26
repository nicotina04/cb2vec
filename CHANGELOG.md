# Changelog

All notable changes to CB2Vec are documented in this file.

## 0.2.1 - 2026-07-26

- Build the same crate as both a Rust `rlib` and native `cdylib`, producing
  `cb2vec.dll`, `libcb2vec.so`, or `libcb2vec.dylib`.
- Add stable C ABI 1.0 opaque trainer/model handles with thread-local errors,
  panic containment, fixed-width layouts, checked pointers, and caller-owned
  buffers.
- Expose deterministic trainer creation and FP32 artifact restore, flattened
  batch evaluation/training, full epochs, prediction, PTQ, artifact export,
  artifact loading, and single/batch quantized inference.
- Add a checked-in C11/C++ header and a Unity C# binding with layout checks,
  `SafeHandle`, pinned arrays, and a trainer/PTQ/reload smoke project.
- Document Windows, Linux, macOS, and `cargo-ndk` Android builds plus Unity
  plug-in placement.
- Verify Windows C11 and C# end-to-end dynamic-library calls and build
  `libcb2vec.so` for `arm64-v8a`, `armeabi-v7a`, and `x86_64`.

## 0.2.0 - 2026-07-26

- Add the site-preserving `GroupedTokens` input and the same activation/grouped
  pooling topology used by the deployed codebook evaluator.
- Add a deterministic pure-Rust FP32 `Trainer` with weighted mini-batches,
  Adam, stable binary cross entropy with logits, and raw-score mean squared
  error.
- Add deterministic model initialization, optional epoch shuffling, batch and
  epoch metrics, evaluation, and probability prediction.
- Cover embedding, linear-head, FM-factor, and bias gradients with central
  differences; cover repeated-token accumulation, ReLU boundaries, rank-zero
  heads, convergence, invalid-input atomicity, and partial batches.
- Add `InferenceConfig`, unequal-group quantized scoring, and a checked
  token-to-quantized-score path; verify nontrivial ReLU/mean inference through
  a `CB2VEC01` artifact.
- Reject feature-sum and sample-weight overflow, and preflight Adam updates so
  numerical errors do not partially mutate trainer state.
- Add an end-to-end `train_value` example.
- Keep artifact format version 1 and all 0.1 inference APIs compatible.

## 0.1.1 - 2026-07-26

- Move the canonical repository to
  [github.com/nicotina04/cb2vec](https://github.com/nicotina04/cb2vec).
- Add Windows, Linux, macOS, MSRV, lint, documentation, and package CI.
- Document the default JSON feature and cover canonical, legacy, and rejected
  JSON model inputs.
- Keep the runtime API and artifact format unchanged from 0.1.0.

## 0.1.0 - 2026-07-26

- Extract the game-independent codebook model and integer access kernel from
  FIGRID.
- Add floating-point and quantized linear/FM scoring.
- Add exact flat and class-base-plus-residual embedding representations.
- Add the canonical `CB2VEC01` artifact and legacy `NORUCBF1` reader.
- Add the preallocated reversible token journal used by make/undo search.
