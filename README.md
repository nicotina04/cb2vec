<h1 align="center">CB2Vec</h1>

<p align="center">
Train, quantize, and incrementally deploy categorical value models in pure Rust.
</p>

<p align="center">
Change one token. Update one embedding. Undo exactly.
</p>

<p align="center">
  <a href="https://crates.io/crates/cb2vec"><img alt="crates.io" src="https://img.shields.io/crates/v/cb2vec.svg"></a>
  <a href="https://crates.io/crates/cb2vec"><img alt="license" src="https://img.shields.io/crates/l/cb2vec.svg"></a>
  <a href="https://docs.rs/cb2vec"><img alt="docs.rs" src="https://docs.rs/cb2vec/badge.svg"></a>
  <a href="https://github.com/nicotina04/cb2vec/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/nicotina04/cb2vec/actions/workflows/ci.yml/badge.svg"></a>
</p>

## What is CB2Vec?

CB2Vec is a compact, domain-independent categorical value-model toolkit for
mutable discrete state. It provides:

- deterministic FP32 training with Adam, BCE-with-logits, and MSE;
- site activation and grouped sum/mean pooling;
- floating-point and `i16` codebook model representations;
- post-training quantization and a versioned binary artifact;
- one-crate Rust `rlib` plus C ABI `cdylib` deployment;
- native FP32 training, PTQ, artifact I/O, and inference from Unity/C/C++;
- exact integer embedding lookup and replacement deltas;
- grouped linear and factorization-machine scoring;
- exact class-base plus `i8` residual storage;
- a preallocated reversible token journal for make/undo search.

CB2Vec is not Word2Vec and does not invent a vocabulary or tokenize a domain
for you. A consuming application decides what its integer tokens mean, how
sites map to groups, and what the supervised targets mean. Once those tokens
and targets exist, CB2Vec owns the reusable numerical path from FP32 training
through quantized deployment.

## Why this crate?

Codebook evaluators are often written directly inside one game engine. That
makes the useful runtime machinery difficult to reuse and easy to couple to a
specific board size, vocabulary, or color convention.

CB2Vec separates the reusable numerical core and keeps domain rules outside:

```text
tokenized training samples
  -> deterministic FP32 Trainer + Adam
  -> post-training i16 quantization
  -> versioned artifact
  -> deployment runtime

domain state at runtime
  -> categorical tokens at (site, lane)
  -> shared embedding rows
  -> activation and grouped pooling
  -> linear or factorization-machine head
```

The domain adapter remains responsible for token production and perspective
mapping. The core never imports a board, move, ruleset, or search type.

## Is CB2Vec a fit?

CB2Vec is a strong candidate when most of these are true:

- state has fixed or bounded sites, entities, slots, or variables;
- observations at those sites are categorical rather than dense continuous
  vectors;
- one action changes only a small number of tokens;
- search repeatedly branches, evaluates, and undoes state;
- shared token meaning across sites is useful;
- CPU latency, predictable allocation, or native deployment matters.

Examples include backtracking and branch-and-bound search, scheduling and
constraint solvers, graph coloring, dependency resolution, compiler search,
circuit placement, planning, and game AI.

Prefer [NORU](https://github.com/nicotina04/noru) when the representation is
naturally a sparse global feature set feeding an NNUE accumulator and dense
MLP. Prefer a conventional dense or batched model when most state changes
globally, inference is one-shot, inputs are primarily continuous, or GPU
throughput matters more than make/undo latency.

The distinction is architectural, not a universal performance ranking. A
same-engine Gomoku case study
([FIGRID](https://github.com/nicotina04/figrid-board)) measured a much smaller
codebook evaluator and higher search throughput than its earlier flat NNUE,
but that result depends on the domain, feature vocabulary, model shape, and
search workload. See [Benchmark methodology and evidence](BENCHMARKS.md).

## Quick start

Add CB2Vec to `Cargo.toml`:

```toml
[dependencies]
cb2vec = "0.2"
```

Train a two-token value model, quantize it, and pack a deployment artifact:

```rust
use cb2vec::{
    Activation, GroupedTokens, Loss, ModelShape, PackedCodebookArtifact,
    Pooling, Trainer, TrainerConfig, TrainingSample, predict_quantized,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let state = |token| GroupedTokens::new(vec![token], vec![0, 1], vec![0]);
    let samples = vec![
        TrainingSample::new(state(0)?, 0.0),
        TrainingSample::new(state(1)?, 1.0),
    ];
    let mut trainer = Trainer::from_shape(
        ModelShape::new(2, 1, 8, 2)?,
        TrainerConfig {
            activation: Activation::Relu,
            pooling: Pooling::Sum,
            loss: Loss::BinaryCrossEntropyWithLogits,
            batch_size: 2,
            ..TrainerConfig::default()
        },
    )?;
    trainer.train_epochs(&samples, 100)?;

    let (source, inference) = trainer.into_parts();
    let quantized = source.quantize_i16_s32_s64();
    let _deployment_score = predict_quantized(&samples[0].input, &quantized, inference)?;
    let artifact = PackedCodebookArtifact::new_flat(source, quantized, [0; 32])?;
    let bytes = artifact.to_bytes()?;
    assert!(!bytes.is_empty());
    Ok(())
}
```

See [`examples/train_value.rs`](examples/train_value.rs) for a complete
end-to-end example with loss reporting and predictions.

The v1 artifact stores model weights and quantization metadata, but not
`InferenceConfig`. Persist the activation/pooling recipe beside the artifact
or provide it in the deployment adapter. `Trainer::into_parts` makes that
boundary explicit.

## Features and MSRV

| Feature | Default | Purpose |
|---|---:|---|
| `json` | yes | Parse canonical CB2Vec and supported legacy JSON model files through `serde_json`. |

Consumers that load only binary artifacts can avoid the JSON dependency:

```toml
[dependencies]
cb2vec = { version = "0.2", default-features = false }
```

CB2Vec 0.2 requires Rust 1.88 or newer. Training and binary artifacts do not
require an optional feature.

## Native library and Unity

CB2Vec follows the same single-crate native-library shape as NORU:

```toml
[lib]
crate-type = ["rlib", "cdylib"]
```

One build therefore produces both the ordinary Rust library and a native
dynamic library:

| Platform | Native output |
|---|---|
| Windows | `target/release/cb2vec.dll` |
| Linux | `target/release/libcb2vec.so` |
| macOS | `target/release/libcb2vec.dylib` |
| Android | ABI-specific `libcb2vec.so` |

The stable C ABI 1.0 is declared in
[`include/cb2vec.h`](include/cb2vec.h). The ready-to-copy Unity binding in
[`bindings/csharp/CB2VecNative.cs`](bindings/csharp/CB2VecNative.cs) uses
`SafeHandle`, pinned caller-owned arrays, fixed C layouts, and
`[DllImport("cb2vec")]`. It exposes:

- deterministic trainer creation and FP32 artifact restore;
- flattened weighted evaluation, one-update batches, and full epochs;
- trainer logits and probabilities;
- PTQ into an independent immutable inference handle;
- caller-buffer artifact export and artifact reload;
- single and batched quantized inference.

The flattened batch format concatenates all samples' tokens and sites. It has
one global token-offset prefix table and a second prefix table delimiting the
sites owned by each sample. No per-sample pointer graph crosses P/Invoke.

The three version numbers are deliberately independent: crate `0.2.1`,
artifact format `1`, and C ABI `1.0` (`0x00010000`). Artifact v1 does not store
activation/pooling, so native model loading also takes an explicit
`Cb2VecInferenceConfigV1`.

### Windows, Linux, and macOS

Build without the optional JSON parser when the native application only needs
the binary artifact and trainer:

```sh
cargo build --release --no-default-features
```

For a Windows Unity Editor, copy `cb2vec.dll` and
`bindings/csharp/CB2VecNative.cs` into the project, for example:

```text
Assets/
  Plugins/
    x86_64/
      cb2vec.dll
    CB2VecNative.cs
```

For Linux or macOS, copy `libcb2vec.so` or `libcb2vec.dylib` under
`Assets/Plugins`, then select the matching Editor/Standalone OS and CPU in the
Unity Plugin Inspector. The C# import name stays `cb2vec` on every platform;
omit `lib` and the file extension.

### Android `.so`

Rust's Android targets require an Android NDK. `cargo-ndk` supplies the linker
configuration and creates Android's ABI directory layout:

```powershell
rustup target add aarch64-linux-android `
  armv7-linux-androideabi `
  x86_64-linux-android
cargo install cargo-ndk --locked --version 4.1.2

# Unity Hub's NDK is suitable; change this to the installed Editor version.
$env:ANDROID_NDK_HOME = `
  'C:\Program Files\Unity\Hub\Editor\<version>\Editor\Data\PlaybackEngines\AndroidPlayer\NDK'

# Match the Unity project's Minimum API Level.
$env:CARGO_NDK_PLATFORM = '23'

cargo ndk `
  -t arm64-v8a `
  -t armeabi-v7a `
  -t x86_64 `
  -o .\build\android\jniLibs `
  build --release --no-default-features
```

The outputs are:

```text
build/android/jniLibs/
  arm64-v8a/libcb2vec.so
  armeabi-v7a/libcb2vec.so
  x86_64/libcb2vec.so
```

Copy the required `.so` files anywhere under the Unity project's `Assets`
folder (a conventional destination is
`Assets/Plugins/Android/<abi>/libcb2vec.so`). In each file's Plugin Inspector,
enable Android and select the CPU matching its ABI. `arm64-v8a` is the normal
device build; `x86_64` is useful for an emulator, and `armeabi-v7a` is only
needed when the project still supports 32-bit ARM.

See the official
[`cargo-ndk` build instructions](https://github.com/bbqsrc/cargo-ndk),
[Rust Android target support](https://doc.rust-lang.org/rustc/platform-support/android.html),
and [Unity Android native plug-in import guide](https://docs.unity3d.com/Manual/android-native-plugins-import.html).

### Minimal Unity flow

```csharp
using CB2Vec;

var shape = Cb2VecNative.DefaultShape();
shape.TokenCount = 4;
shape.GroupCount = 2;
shape.Dim = 3;
shape.FmRank = 2;

var config = Cb2VecNative.DefaultTrainerConfig();
config.BatchSize = 2;
config.Shuffle = 0;

var dataset = new Cb2VecTrainingBatch(
    new ushort[] { 0, 0, 1, 2, 3, 1 },
    new uint[] { 0, 2, 3, 3, 4, 6 }, // global site -> token offsets
    new uint[] { 0, 0, 1, 0, 1 },    // global site -> group
    new uint[] { 0, 3, 5 },          // sample -> site offsets
    new float[] { 0.2f, 0.8f });
var input = new Cb2VecInput(
    new ushort[] { 0, 0, 1 },
    new uint[] { 0, 2, 3, 3 },
    new uint[] { 0, 0, 1 });

using (var trainer = Cb2VecTrainer.Create(shape, config))
{
    var report = trainer.TrainEpoch(dataset);
    var quant = Cb2VecNative.DefaultQuantization();
    using (var model = trainer.Quantize(quant))
    {
        float score = model.Predict(input);
    }
    byte[] artifact = trainer.WriteArtifact(quant, new byte[32]);
}
```

Trainer handles are not internally synchronized; serialize every call that
uses the same trainer. Quantized model handles are immutable and may serve
concurrent prediction calls as long as they are not disposed concurrently.
The binding never retains managed pointers after a native call.
The crate's direct release build pins `panic = "unwind"` so Rust panics can be
converted into an ABI status. A parent Cargo workspace can override dependency
profiles; if it deliberately selects `panic = "abort"`, no native library can
recover from a Rust panic.

## Training model

`GroupedTokens` preserves site boundaries in a flat allocation-friendly
layout:

```text
tokens:       [t0, t1, t2, ...]
site_offsets: [0,     2,  3, ...]
site_groups:  [g0,    g1, ...]
```

For each site, CB2Vec sums all referenced embedding rows, applies `Identity`
or `Relu`, and then reduces activated sites into model groups with `Sum` or
`Mean`. The pooled vector is evaluated by the same `score_f32` linear/FM head
used after training.

Inputs must include every site that contributes to the pooling denominator,
including sites with an empty token range. Perspective or color remapping is a
domain responsibility and must happen before token IDs are passed to CB2Vec.
The topology matches FIGRID's trainer, but 0.2 does not promise bit-for-bit
reproduction of historical checkpoints, RNG streams, or sparse-Adam updates.

`Trainer` uses deterministic initialization and Fisher-Yates shuffling from a
caller-controlled seed. A mini-batch is evaluated against frozen weights,
weighted gradients are averaged once, and one bias-corrected Adam step is
applied. Repeated appearances of the same token accumulate into the same
embedding gradient.

Available objectives are:

- `BinaryCrossEntropyWithLogits`, with stable logit-space loss and targets in
  `0.0..=1.0`;
- `MeanSquaredError`, applied to the raw score.

`evaluate`, `train_batch`, `train_epoch`, and `train_epochs` return loss,
sample, batch, optimizer-step, and epoch metrics. Invalid offsets, tokens,
groups, targets, or sample weights are rejected before a batch mutates the
model.

This is ordinary FP32 training followed by post-training quantization (PTQ).
Quantization-aware training (QAT) is not part of 0.2.

Version 0.2 checkpoints model weights through the existing JSON/artifact
paths. It does not serialize Adam moments, shuffle RNG state, or epoch state;
constructing a new `Trainer` from saved weights starts a fresh optimizer.

## Attaching a policy

CB2Vec can provide the state representation for a policy without making
actions part of the core crate. A domain adapter maps state into tokens and
maintains the embedding or pooled feature vector; a `PolicyHead` in the
consumer then maps that vector to action logits and applies the domain's
legal-action mask.

This boundary lets the same CB2Vec model support a value head, a policy head,
or both, while action encoding, legality, and search remain game-specific. A
reusable policy implementation can therefore live beside CB2Vec, or in a
future companion crate, rather than inside the token and journal primitives.

## Reversible token updates

A token state consists of:

- a `site`, such as a cell or entity;
- one or more categorical `lanes` at that site;
- a logical search depth;
- a materialized depth already applied to the numeric state.

`ReversibleTokenJournal<T, LANES, MAX_DELTAS>` owns the logical tokens and
preallocates every frame at construction. `push_after` records changed lanes,
`materialize_pending` applies grouped site deltas to a consumer-defined sink,
and `pop` reverses an applied frame when necessary.

See [`examples/reversible_journal.rs`](examples/reversible_journal.rs) for a
complete make/materialize/undo round trip.

## Model and head

`CodebookWeights` stores:

- `token_count * dim` floating-point embedding values;
- `group_count * dim` linear-head values;
- `(group_count * dim) * fm_rank` optional FM factors;
- one floating-point bias.

`QuantizedCodebookWeights` stores the same logical model with separate
positive scales for embeddings, the linear head, and FM factors. The initial
FIGRID deployment uses scales 32, 64, and 64; callers may choose other scales
with `quantize_i16`. `Trainer::into_weights` returns the existing
`CodebookWeights`, so no conversion layer is needed before quantization or
artifact packing.

The checked `score_f32` function consumes already normalized floating-point
features. `score_quantized_uniform` consumes integer grouped sums when every
group has the same pooling divisor. `score_quantized_grouped` accepts a
different positive divisor per group, and `predict_quantized` performs the
complete checked token → activation → grouping → quantized-score path. These
functions reject invalid feature shapes, token/group IDs, arithmetic overflow,
and zero divisors.

## Flat and factored storage

`FactoredQuantizedCodebookWeights` stores each embedding row as:

```text
class_base[token_class] + i8_residual[token]
```

Reconstruction is exact and checked for `i16` overflow. This representation
can reduce serialized size, but it is not assumed to be the fastest hot-loop
layout. Use `reconstruct_flat` at load time when flat row access is faster on
the target workload.

## Artifact format

The canonical v1 artifact uses:

- magic `CB2VEC01`;
- little-endian integers and floating-point bit patterns;
- explicit model shape and quantization scales;
- a flat or factored payload kind;
- exact payload lengths and zeroed reserved bytes;
- a caller-supplied 32-byte source provenance digest;
- rejection of unknown versions, malformed shapes, non-finite source values,
  and trailing bytes.

`PackedCodebookArtifact::to_bytes` writes the canonical format.
`PackedCodebookArtifact::parse` also reads the legacy `NORUCBF1` magic so
FIGRID artifacts can migrate without changing their numerical payload.
Legacy input is always rewritten with the canonical CB2Vec magic.

The source digest is provenance metadata supplied by the packer. CB2Vec does
not claim that it hashes or authenticates the original training file.

See [`examples/flat_roundtrip.rs`](examples/flat_roundtrip.rs).

## Core guarantees

- Integer embedding replacement is exact for a valid model.
- Journal validation errors do not mutate logical tokens or logical depth.
- Journal push, materialize, and pop allocate nothing after construction.
- Artifact parsing uses checked length arithmetic and rejects trailing data.
- The crate contains no game, board, action, vocabulary, or search policy.
- The numerical core contains no `unsafe` code; audited raw-pointer handling is
  isolated to `ffi`.

A `TokenDeltaSink` must not panic. Sink mutations are deliberately
non-transactional because rollback would add overhead and cannot generally
undo arbitrary external side effects.

## Scope and non-goals

CB2Vec 0.2 covers categorical value-model training, PTQ, artifact storage,
inference, and reversible state changes. It does not provide:

- vocabulary construction or tokenization;
- QAT or reinforcement-learning orchestration;
- legal-action masking or a policy head;
- board symmetry or color-perspective semantics;
- SIMD, optimizer-state checkpoints, an incremental C session, or a `no_std`
  contract.

Policy or value heads built for one game should remain in that game's adapter
until a second domain demonstrates the same abstraction.

The C ABI intentionally keeps the const-generic reversible journal on the Rust
side. A future incremental native session can be added as a separate opaque
handle without freezing game-specific state into ABI 1.0.

## Evidence and provenance

The runtime originated in a deployed Gomoku engine
([FIGRID](https://github.com/nicotina04/figrid-board)) and was extracted after
the following integration gates:

- exact mixed make/undo comparison against full rebuild over 100,000
  transitions;
- exact search decision and node-count comparison on a sealed 1,022-root
  corpus;
- bit-exact reconstruction of the deployed flat integer model from factored
  storage;
- same-binary measurement of the directional token-delta evaluator.

Those are FIGRID integration results, not universal speed claims for every
CB2Vec consumer. Workload-level performance still depends on token locality,
embedding dimension, group layout, and search behavior.

The standalone `v0.1.0` root preserves the exact `crates/cb2vec` tree from
FIGRID commit `54d5807`. Version 0.2 generalizes the FP32 training graph used
by FIGRID while keeping the v1 binary artifact and all 0.1 inference APIs
compatible.

## Relationship to NORU

[NORU](https://crates.io/crates/noru) and CB2Vec are sibling Rust primitives:

- NORU maps sparse global features through an NNUE accumulator and dense MLP.
- CB2Vec maps local categorical tokens through shared embeddings and a small
  grouped linear/FM head.

Both now provide a pure-Rust FP32 trainer and integer deployment weights. Use
NORU when your representation is naturally a sparse global feature set; use
CB2Vec when multiple local categorical observations should share learned
embedding rows before grouped pooling.

A Gomoku engine
([FIGRID](https://github.com/nicotina04/figrid-board)) uses NORU for its NNUE
lineage and learned ordering model, and CB2Vec for its promoted codebook leaf
evaluator. It is one validation case, not the definition of CB2Vec's scope.

## Development

```sh
cargo fmt --all --check
cargo test --locked --all-features
cargo test --locked --no-default-features
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo doc --locked --all-features --no-deps
cargo run --locked --example train_value
cargo build --release --no-default-features
dotnet build bindings/csharp/CB2VecNative.csproj --configuration Release
cargo package --locked
```

Numerical gradient, convergence, model, journal, factored-storage, and
artifact tests live in this crate. ABI tests additionally cover layout,
Rust/FFI training parity, invalid-input atomicity, PTQ/artifact parity,
factored-load flattening, panic containment, C11/C++ header compilation, and
the C# trainer-to-reload smoke flow. Android release builds have been checked
as ELF64 AArch64, ELF32 ARM, and ELF64 x86-64, each exporting the same 23
`cb2vec_*` symbols. Game-level integration tests remain in
[figrid-board](https://github.com/nicotina04/figrid-board) beside its adapter.

## License

Licensed under either of:

- Apache License, Version 2.0
- MIT License

at your option.
