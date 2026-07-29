# Changelog

All notable changes to CB2Vec are documented in this file.

## 0.3.0 - 2026-07-30

### Incremental search sessions

- Add `IncrementalSession`, a preallocated make/predict/undo evaluator for
  alpha-beta search. Pushing a move records a reversible frame; scoring
  materializes only the sites that changed; popping restores the previous
  position exactly.
- Scores are bit-identical to `predict_quantized` on the same tokens and
  inference recipe, under both activations and both pooling modes.
- After construction the push/materialize/predict/pop loop performs no heap
  allocation, verified by a counting global allocator in the test binary and
  again through the C ABI.
- A whole frame is validated before anything mutates, so an invalid delta, a
  stale expected old token, a duplicate slot, or an exceeded limit leaves the
  session byte-for-byte unchanged and does not consume depth.
- Sessions are single-owner; any number may share one immutable model.

### C ABI 1.1

- Bump to `0x00010001`. The revision is strictly additive: all 23 ABI 1.0
  symbols keep their signatures and behavior, and every versioned struct still
  accepts `CB2VEC_ABI_VERSION_1_0`. Check the major version only.
- Add nine session entry points, three checkpoint entry points, and four
  artifact/metadata entry points, with `Cb2VecSessionConfigV1`,
  `Cb2VecTokenDeltaV1`, `Cb2VecSessionInfoV1`, `Cb2VecArtifactMetadataV1`, and
  `Cb2VecArtifactInfoV1`.
- Add `CB2VEC_ERROR_LIMIT_EXCEEDED`, `CB2VEC_ERROR_STATE`,
  `CB2VEC_ERROR_CHECKPOINT`, and `CB2VEC_ERROR_OUT_OF_MEMORY`.
- Model weights are now reference counted, so a session outliving its model
  handle is defined behavior rather than a dangling pointer. Handles may be
  freed in any order.
- `cb2vec_session_push_v1` borrows the caller's delta array directly;
  `Cb2VecTokenDeltaV1` and `SessionDelta` are pinned to the same layout by
  compile-time size, alignment, and offset assertions.
- Clearing the thread-local error no longer allocates, so the session loop is
  allocation-free all the way down to the C boundary.

### Trainer checkpoints

- Add the `CB2VECCK` format: FP32 weights, Adam first and second moments, both
  bias-correction powers, the optimizer step, the shuffle RNG including its
  buffered normal, completed epochs, and the full trainer config.
- Resuming reproduces an uninterrupted run bit for bit, including the epoch
  shuffle order.
- A CRC-32 over the content plus header/payload shape agreement rejects
  corrupted, truncated, and incompatible files without building a trainer.
- Inference artifacts stay the lightweight deployment format; restoring one
  still deliberately resets optimizer and epoch state.

### Artifact version 2

- Add a 128-byte header carrying the activation and pooling the model was
  trained with, plus a consumer-defined `schema_version` and 16-byte
  `schema_digest` extension point for vocabulary identity.
- `cb2vec_model_load_v2` loads using the stored recipe and rejects a
  conflicting one; `cb2vec_artifact_probe_v1` reads all of it without building
  a model.
- Version 1 is unchanged, byte for byte, in both directions. Readers accept
  both versions.

### C# binding

- Add `Cb2VecSession` with a `SafeHandle`, plus `Cb2VecPinnedBuffer<T>`,
  `Cb2VecPinnedInput`, `Cb2VecPinnedBatch`, `PredictInto`, and
  `PredictBatchInto` for allocation-free repeated inference.
- The search loop and the pinned whole-input path allocate zero managed bytes
  over 5,000 iterations, asserted in the smoke test.
- Add checkpoint and artifact-v2 wrappers, `ProbeArtifact`, and `GetMetadata`.
- Using a disposed session throws `ObjectDisposedException`; double dispose is
  a no-op; a session may be disposed before or after its model.
- No `unsafe` code and no `/unsafe` switch is required.

### Unity packaging

- Add `unity/Assets/Plugins`, a ready-to-copy tree with stable `.meta` files
  that enable exactly one platform per library, with the right CPU.
- Add an Editor script that validates and repairs `PluginImporter` settings on
  import, so a binary copied in without its `.meta` is still configured
  correctly instead of defaulting to "every platform, CPU ARMv7".
- Add `tools/verify_unity_plugins.py`, which checks the `.meta` settings, GUID
  uniqueness, each ELF's machine type against its ABI directory, and the full
  exported symbol list including the frozen ABI 1.0 set. CI runs it against
  the real Android builds.

### Other

- Add the `search_session` and `resume_training` examples.
- Freeze `include/cb2vec.symbols.abi10` as a regression baseline.

## 0.2.2 - 2026-07-27

- Lead with the domain-independent workload: mutable categorical state with
  small local changes, repeated evaluation, and exact undo.
- Add a practical selection guide covering suitable non-game workloads,
  cases where NORU or a conventional dense model is a better fit, and the
  boundary between representation choice and universal speed claims.
- Add `BENCHMARKS.md` with a scoped same-engine Gomoku evaluator case study,
  exact incremental-update evidence, correctness requirements, and a template
  for benchmarking new consumers.
- Include the benchmark guide in the published crate package.
- No Rust API, C ABI, artifact format, training, quantization, or inference
  behavior changed.

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
