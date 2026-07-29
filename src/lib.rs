//! Trainable and incrementally updatable categorical codebook embeddings.
//!
//! CB2Vec provides the game-independent part of an evaluator built from
//! integer token IDs, shared embedding rows, grouped pooling, and a small
//! linear or factorization-machine head. It includes a deterministic FP32
//! trainer with Adam, post-training `i16` quantization, a versioned artifact,
//! and a preallocated reversible journal for search engines that use
//! make/undo.
//!
//! For alpha-beta style search, [`IncrementalSession`] combines that journal
//! with the integer evaluator state so a move can be applied, scored, and
//! undone without rebuilding features and without allocating.
//! [`TrainerCheckpoint`] resumes a long training run exactly where it stopped.
//!
//! Vocabulary construction, legal actions, board mutation, perspective
//! mapping, and search policy remain responsibilities of the consuming
//! application.

#![deny(unsafe_code)]

mod artifact;
mod checkpoint;
mod factored;
#[allow(unsafe_code)]
pub mod ffi;
mod journal;
mod model;
mod session;
#[cfg(test)]
mod testing;
mod trainer;

#[cfg(test)]
#[global_allocator]
static COUNTING_ALLOCATOR: testing::CountingAllocator = testing::CountingAllocator;

pub use artifact::{
    ArtifactError, ArtifactMetadata, CB2VEC_ARTIFACT_HEADER_LEN, CB2VEC_ARTIFACT_MAGIC,
    CB2VEC_ARTIFACT_V2_HEADER_LEN, CB2VEC_ARTIFACT_VERSION, CB2VEC_ARTIFACT_VERSION_V2,
    LEGACY_NORU_CBF_MAGIC, PackedCodebookArtifact, PackedCodebookKind, PackedQuantizedPayload,
};
pub use checkpoint::{
    CB2VEC_CHECKPOINT_HEADER_LEN, CB2VEC_CHECKPOINT_MAGIC, CB2VEC_CHECKPOINT_VERSION,
    CheckpointError, TrainerCheckpoint,
};
pub use factored::FactoredQuantizedCodebookWeights;
pub use journal::{
    JournalError, ReversibleTokenJournal, TokenDelta, TokenDeltaPop, TokenDeltaReplay,
    TokenDeltaSink,
};
pub use model::{
    CodebookWeights, FloatCodebookAccess, ModelError, ModelShape, QUANT_EMBED_SCALE,
    QUANT_FACTOR_SCALE, QUANT_HEAD_SCALE, QuantizedCodebookAccess, QuantizedCodebookWeights,
    add_embedding_delta_to, add_embedding_to, quantize_i16, score_f32, score_quantized_grouped,
    score_quantized_uniform,
};
pub use session::{
    IncrementalSession, SESSION_MAX_TOKEN_SLOTS, SessionDelta, SessionError, SessionInfo,
    SessionLimits,
};
pub use trainer::{
    Activation, AdamConfig, GroupedTokens, InferenceConfig, Loss, Pooling,
    QuantizedGroupedFeatures, SampleIssue, Trainer, TrainerConfig, TrainingError, TrainingMetrics,
    TrainingSample, materialize_features_f32, materialize_features_f32_into,
    materialize_features_quantized, predict_quantized,
};
