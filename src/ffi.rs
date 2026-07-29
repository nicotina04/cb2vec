//! Stable C ABI for native and Unity deployments.
//!
//! The numerical core remains safe Rust. This module is the only place where
//! raw foreign pointers are converted into Rust values. See
//! `include/cb2vec.h` for the complete ownership and thread-safety contract.

#![deny(unsafe_op_in_unsafe_fn)]

use std::cell::RefCell;
use std::ffi::c_char;
use std::fmt;
use std::mem::{align_of, offset_of, size_of};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::slice;
use std::sync::Arc;

use crate::{
    Activation, AdamConfig, ArtifactMetadata, CheckpointError, GroupedTokens, IncrementalSession,
    InferenceConfig, Loss, ModelShape, PackedCodebookArtifact, PackedQuantizedPayload, Pooling,
    SessionDelta, SessionError, SessionLimits, Trainer, TrainerCheckpoint, TrainerConfig,
    TrainingError, TrainingMetrics, TrainingSample, predict_quantized,
};

/// ABI major 1, minor 1.
///
/// Minor revisions are purely additive: every ABI 1.0 symbol keeps its
/// signature and semantics, and every versioned struct still accepts
/// [`CB2VEC_ABI_VERSION_1_0`] in its `abi_version` field. Callers should
/// require `major == 1` rather than an exact match.
pub const CB2VEC_ABI_VERSION: u32 = 0x0001_0001;

/// The original ABI revision, still accepted by every `_v1` entry point.
pub const CB2VEC_ABI_VERSION_1_0: u32 = 0x0001_0000;

pub const CB2VEC_OK: i32 = 0;
pub const CB2VEC_ERROR_NULL_POINTER: i32 = -1;
pub const CB2VEC_ERROR_INVALID_ARGUMENT: i32 = -2;
pub const CB2VEC_ERROR_ABI_MISMATCH: i32 = -3;
pub const CB2VEC_ERROR_ARTIFACT: i32 = -4;
pub const CB2VEC_ERROR_MODEL: i32 = -5;
pub const CB2VEC_ERROR_NUMERIC: i32 = -6;
pub const CB2VEC_ERROR_BUFFER_TOO_SMALL: i32 = -7;
/// A session capacity chosen at creation was exceeded. Added in ABI 1.1.
pub const CB2VEC_ERROR_LIMIT_EXCEEDED: i32 = -8;
/// The operation is not valid for the handle's current state. Added in 1.1.
pub const CB2VEC_ERROR_STATE: i32 = -9;
/// A trainer checkpoint was corrupt or incompatible. Added in ABI 1.1.
pub const CB2VEC_ERROR_CHECKPOINT: i32 = -10;
/// A required allocation failed. Added in ABI 1.1.
pub const CB2VEC_ERROR_OUT_OF_MEMORY: i32 = -11;
pub const CB2VEC_ERROR_PANIC: i32 = -127;

pub const CB2VEC_ACTIVATION_IDENTITY: u32 = 0;
pub const CB2VEC_ACTIVATION_RELU: u32 = 1;
pub const CB2VEC_POOLING_SUM: u32 = 0;
pub const CB2VEC_POOLING_MEAN: u32 = 1;
pub const CB2VEC_LOSS_BCE_WITH_LOGITS: u32 = 0;
pub const CB2VEC_LOSS_MSE: u32 = 1;
pub const CB2VEC_MODEL_KIND_FLAT: u32 = 0;
pub const CB2VEC_MODEL_KIND_FACTORED: u32 = 1;
pub const CB2VEC_MODEL_KIND_FP32: u32 = 2;
pub const CB2VEC_MODEL_FLAG_LEGACY_MAGIC: u32 = 1;
pub const CB2VEC_MODEL_FLAG_FLATTENED_AT_LOAD: u32 = 2;

static LIBRARY_VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();

thread_local! {
    /// NUL-terminated message buffer, never empty.
    ///
    /// A plain byte buffer rather than a `CString` so that clearing the error
    /// at the start of every call reuses its capacity. That keeps the
    /// documented allocation-free session loop allocation-free all the way
    /// down to the C boundary.
    static LAST_ERROR: RefCell<Vec<u8>> = RefCell::new(vec![0u8]);
}

/// Opaque FP32 trainer handle.
pub struct Cb2VecTrainer {
    trainer: Trainer,
}

/// Opaque immutable quantized-model handle.
///
/// The weights sit behind an [`Arc`] so a session created from this model
/// keeps them alive even if the model handle is freed first.
pub struct Cb2VecWeights {
    payload: Arc<PackedQuantizedPayload>,
    inference: InferenceConfig,
    original_kind: u32,
    flags: u32,
    artifact_version: u32,
    metadata: ArtifactMetadata,
}

/// Opaque single-owner incremental search session handle.
pub struct Cb2VecSession {
    session: IncrementalSession<Arc<PackedQuantizedPayload>>,
}

/// Fixed-layout model shape for ABI v1.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Cb2VecModelShapeV1 {
    pub struct_size: u32,
    pub abi_version: u32,
    pub token_count: u32,
    pub group_count: u32,
    pub dim: u32,
    pub fm_rank: u32,
    pub reserved: [u32; 2],
}

impl Default for Cb2VecModelShapeV1 {
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: CB2VEC_ABI_VERSION,
            token_count: 2,
            group_count: 1,
            dim: 8,
            fm_rank: 0,
            reserved: [0; 2],
        }
    }
}

/// Fixed-layout trainer configuration for ABI v1.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cb2VecTrainerConfigV1 {
    pub struct_size: u32,
    pub abi_version: u32,
    pub activation: u32,
    pub pooling: u32,
    pub loss: u32,
    pub batch_size: u32,
    pub shuffle: u32,
    pub flags: u32,
    pub seed: u64,
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub reserved: [u32; 2],
}

impl Default for Cb2VecTrainerConfigV1 {
    fn default() -> Self {
        let config = TrainerConfig::default();
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: CB2VEC_ABI_VERSION,
            activation: activation_to_ffi(config.activation),
            pooling: pooling_to_ffi(config.pooling),
            loss: loss_to_ffi(config.loss),
            batch_size: config.batch_size as u32,
            shuffle: u32::from(config.shuffle),
            flags: 0,
            seed: config.seed,
            learning_rate: config.adam.learning_rate,
            beta1: config.adam.beta1,
            beta2: config.adam.beta2,
            epsilon: config.adam.epsilon,
            reserved: [0; 2],
        }
    }
}

/// Fixed-layout post-training quantization configuration for ABI v1.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Cb2VecQuantizationConfigV1 {
    pub struct_size: u32,
    pub abi_version: u32,
    pub embedding_scale: i32,
    pub head_scale: i32,
    pub factor_scale: i32,
    pub flags: u32,
    pub reserved: [u32; 2],
}

impl Default for Cb2VecQuantizationConfigV1 {
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: CB2VEC_ABI_VERSION,
            embedding_scale: crate::QUANT_EMBED_SCALE,
            head_scale: crate::QUANT_HEAD_SCALE,
            factor_scale: crate::QUANT_FACTOR_SCALE,
            flags: 0,
            reserved: [0; 2],
        }
    }
}

/// Activation and pooling recipe supplied beside an artifact-v1 model.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Cb2VecInferenceConfigV1 {
    pub struct_size: u32,
    pub activation: u32,
    pub pooling: u32,
    pub flags: u32,
}

impl Default for Cb2VecInferenceConfigV1 {
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            activation: CB2VEC_ACTIVATION_RELU,
            pooling: CB2VEC_POOLING_MEAN,
            flags: 0,
        }
    }
}

/// Flattened ragged batch view for ABI v1.
///
/// The library copies this view into owned Rust samples before touching a
/// trainer. No pointer is retained after the call returns.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Cb2VecTrainingBatchV1 {
    pub struct_size: u32,
    pub flags: u32,
    pub tokens: *const u16,
    pub site_token_offsets: *const u32,
    pub site_groups: *const u32,
    pub sample_site_offsets: *const u32,
    pub targets: *const f32,
    pub weights: *const f32,
    pub tokens_len: u32,
    pub site_count: u32,
    pub sample_count: u32,
    pub reserved: u32,
}

/// Fixed-layout training report for ABI v1.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cb2VecTrainingMetricsV1 {
    pub struct_size: u32,
    pub abi_version: u32,
    pub mean_loss: f32,
    pub reserved: u32,
    pub total_weight: f64,
    pub sample_count: u64,
    pub batch_count: u64,
    pub optimizer_step: u64,
    pub completed_epochs: u64,
    pub reserved_tail: u64,
}

impl Default for Cb2VecTrainingMetricsV1 {
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: CB2VEC_ABI_VERSION,
            mean_loss: 0.0,
            reserved: 0,
            total_weight: 0.0,
            sample_count: 0,
            batch_count: 0,
            optimizer_step: 0,
            completed_epochs: 0,
            reserved_tail: 0,
        }
    }
}

impl From<TrainingMetrics> for Cb2VecTrainingMetricsV1 {
    fn from(metrics: TrainingMetrics) -> Self {
        Self {
            mean_loss: metrics.mean_loss,
            total_weight: metrics.total_weight,
            sample_count: metrics.sample_count as u64,
            batch_count: metrics.batch_count as u64,
            optimizer_step: metrics.optimizer_step,
            completed_epochs: metrics.completed_epochs,
            ..Self::default()
        }
    }
}

/// Fixed-layout model metadata for ABI v1.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cb2VecModelInfoV1 {
    pub struct_size: u32,
    pub abi_version: u32,
    pub artifact_version: u32,
    pub flags: u32,
    pub token_count: u32,
    pub group_count: u32,
    pub dim: u32,
    pub fm_rank: u32,
    pub kind: u32,
    pub activation: u32,
    pub pooling: u32,
    pub embedding_scale: i32,
    pub head_scale: i32,
    pub factor_scale: i32,
    pub reserved: [u32; 2],
}

impl Default for Cb2VecModelInfoV1 {
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: CB2VEC_ABI_VERSION,
            artifact_version: u32::from(crate::CB2VEC_ARTIFACT_VERSION),
            flags: 0,
            token_count: 0,
            group_count: 0,
            dim: 0,
            fm_rank: 0,
            kind: CB2VEC_MODEL_KIND_FLAT,
            activation: CB2VEC_ACTIVATION_IDENTITY,
            pooling: CB2VEC_POOLING_SUM,
            embedding_scale: 0,
            head_scale: 0,
            factor_scale: 0,
            reserved: [0; 2],
        }
    }
}

/// Fixed capacities for one incremental search session. Added in ABI 1.1.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Cb2VecSessionConfigV1 {
    pub struct_size: u32,
    pub abi_version: u32,
    pub max_sites: u32,
    pub max_token_slots: u32,
    pub max_deltas_per_frame: u32,
    pub max_depth: u32,
    pub flags: u32,
    pub reserved: [u32; 1],
}

impl Default for Cb2VecSessionConfigV1 {
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: CB2VEC_ABI_VERSION,
            max_sites: 256,
            max_token_slots: 1024,
            max_deltas_per_frame: 8,
            max_depth: 64,
            flags: 0,
            reserved: [0; 1],
        }
    }
}

/// One token replacement in a pushed search frame. Added in ABI 1.1.
///
/// Layout-identical to [`SessionDelta`], so `cb2vec_session_push_v1` borrows a
/// caller array directly instead of converting it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Cb2VecTokenDeltaV1 {
    pub site: u32,
    pub lane: u32,
    pub old_token: u16,
    pub new_token: u16,
}

/// Observable session state. Added in ABI 1.1.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Cb2VecSessionInfoV1 {
    pub struct_size: u32,
    pub abi_version: u32,
    pub site_count: u32,
    pub token_slots: u32,
    pub group_count: u32,
    pub depth: u32,
    pub materialized_depth: u32,
    pub pending_deltas: u32,
    pub max_sites: u32,
    pub max_token_slots: u32,
    pub max_deltas_per_frame: u32,
    pub max_depth: u32,
    pub activation: u32,
    pub pooling: u32,
    pub flags: u32,
    pub reserved: [u32; 1],
}

impl Default for Cb2VecSessionInfoV1 {
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: CB2VEC_ABI_VERSION,
            site_count: 0,
            token_slots: 0,
            group_count: 0,
            depth: 0,
            materialized_depth: 0,
            pending_deltas: 0,
            max_sites: 0,
            max_token_slots: 0,
            max_deltas_per_frame: 0,
            max_depth: 0,
            activation: CB2VEC_ACTIVATION_IDENTITY,
            pooling: CB2VEC_POOLING_SUM,
            flags: 0,
            reserved: [0; 1],
        }
    }
}

/// Consumer-defined schema identity written into a v2 artifact. Added in 1.1.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Cb2VecArtifactMetadataV1 {
    pub struct_size: u32,
    pub abi_version: u32,
    pub schema_version: u32,
    pub flags: u32,
    pub schema_digest: [u8; 16],
}

impl Default for Cb2VecArtifactMetadataV1 {
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: CB2VEC_ABI_VERSION,
            schema_version: 0,
            flags: 0,
            schema_digest: [0; 16],
        }
    }
}

/// Everything readable from artifact bytes without building a model.
/// Added in ABI 1.1.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Cb2VecArtifactInfoV1 {
    pub struct_size: u32,
    pub abi_version: u32,
    pub artifact_version: u32,
    pub kind: u32,
    pub token_count: u32,
    pub group_count: u32,
    pub dim: u32,
    pub fm_rank: u32,
    pub has_inference_config: u32,
    pub activation: u32,
    pub pooling: u32,
    pub schema_version: u32,
    pub embedding_scale: i32,
    pub head_scale: i32,
    pub factor_scale: i32,
    pub flags: u32,
    pub source_sha256: [u8; 32],
    pub schema_digest: [u8; 16],
}

impl Default for Cb2VecArtifactInfoV1 {
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: CB2VEC_ABI_VERSION,
            artifact_version: 0,
            kind: CB2VEC_MODEL_KIND_FLAT,
            token_count: 0,
            group_count: 0,
            dim: 0,
            fm_rank: 0,
            has_inference_config: 0,
            activation: CB2VEC_ACTIVATION_IDENTITY,
            pooling: CB2VEC_POOLING_SUM,
            schema_version: 0,
            embedding_scale: 0,
            head_scale: 0,
            factor_scale: 0,
            flags: 0,
            source_sha256: [0; 32],
            schema_digest: [0; 16],
        }
    }
}

const _: [(); 32] = [(); size_of::<Cb2VecModelShapeV1>()];
const _: [(); 64] = [(); size_of::<Cb2VecTrainerConfigV1>()];
const _: [(); 32] = [(); size_of::<Cb2VecQuantizationConfigV1>()];
const _: [(); 16] = [(); size_of::<Cb2VecInferenceConfigV1>()];
const _: [(); 64] = [(); size_of::<Cb2VecTrainingMetricsV1>()];
const _: [(); 64] = [(); size_of::<Cb2VecModelInfoV1>()];
const _: [(); 32] = [(); size_of::<Cb2VecSessionConfigV1>()];
const _: [(); 12] = [(); size_of::<Cb2VecTokenDeltaV1>()];
const _: [(); 64] = [(); size_of::<Cb2VecSessionInfoV1>()];
const _: [(); 32] = [(); size_of::<Cb2VecArtifactMetadataV1>()];
const _: [(); 112] = [(); size_of::<Cb2VecArtifactInfoV1>()];
#[cfg(target_pointer_width = "64")]
const _: [(); 72] = [(); size_of::<Cb2VecTrainingBatchV1>()];
#[cfg(target_pointer_width = "32")]
const _: [(); 48] = [(); size_of::<Cb2VecTrainingBatchV1>()];

// `cb2vec_session_push_v1` reinterprets the caller's `Cb2VecTokenDeltaV1`
// array as `&[SessionDelta]`. These assertions are what make that sound; if
// either type's layout ever drifts, the crate stops compiling.
const _: [(); size_of::<Cb2VecTokenDeltaV1>()] = [(); size_of::<SessionDelta>()];
const _: [(); align_of::<Cb2VecTokenDeltaV1>()] = [(); align_of::<SessionDelta>()];
const _: [(); offset_of!(Cb2VecTokenDeltaV1, site)] = [(); offset_of!(SessionDelta, site)];
const _: [(); offset_of!(Cb2VecTokenDeltaV1, lane)] = [(); offset_of!(SessionDelta, lane)];
const _: [(); offset_of!(Cb2VecTokenDeltaV1, old_token)] = [(); offset_of!(SessionDelta, old)];
const _: [(); offset_of!(Cb2VecTokenDeltaV1, new_token)] = [(); offset_of!(SessionDelta, new)];

#[derive(Debug)]
enum FfiError {
    Null(&'static str),
    Invalid(String),
    Abi(String),
    Artifact(String),
    Model(String),
    Numeric(String),
    BufferTooSmall { required: usize, capacity: usize },
    LimitExceeded(String),
    State(String),
    Checkpoint(String),
    OutOfMemory(String),
}

impl FfiError {
    fn status(&self) -> i32 {
        match self {
            Self::Null(_) => CB2VEC_ERROR_NULL_POINTER,
            Self::Invalid(_) => CB2VEC_ERROR_INVALID_ARGUMENT,
            Self::Abi(_) => CB2VEC_ERROR_ABI_MISMATCH,
            Self::Artifact(_) => CB2VEC_ERROR_ARTIFACT,
            Self::Model(_) => CB2VEC_ERROR_MODEL,
            Self::Numeric(_) => CB2VEC_ERROR_NUMERIC,
            Self::BufferTooSmall { .. } => CB2VEC_ERROR_BUFFER_TOO_SMALL,
            Self::LimitExceeded(_) => CB2VEC_ERROR_LIMIT_EXCEEDED,
            Self::State(_) => CB2VEC_ERROR_STATE,
            Self::Checkpoint(_) => CB2VEC_ERROR_CHECKPOINT,
            Self::OutOfMemory(_) => CB2VEC_ERROR_OUT_OF_MEMORY,
        }
    }
}

impl fmt::Display for FfiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null(name) => write!(f, "{name} is null"),
            Self::Invalid(message)
            | Self::Abi(message)
            | Self::Artifact(message)
            | Self::Model(message)
            | Self::Numeric(message)
            | Self::LimitExceeded(message)
            | Self::State(message)
            | Self::Checkpoint(message)
            | Self::OutOfMemory(message) => f.write_str(message),
            Self::BufferTooSmall { required, capacity } => write!(
                f,
                "output buffer is too small: capacity {capacity}, required {required}"
            ),
        }
    }
}

/// Maps a session failure onto the narrowest status a caller can act on.
fn map_session_error(error: SessionError) -> FfiError {
    let message = error.to_string();
    match error {
        SessionError::Model(_) => FfiError::Model(message),
        SessionError::LimitExceeded { .. } => FfiError::LimitExceeded(message),
        SessionError::NotReady | SessionError::EmptyStack => FfiError::State(message),
        SessionError::AllocationFailed { .. } => FfiError::OutOfMemory(message),
        _ => FfiError::Invalid(message),
    }
}

fn map_checkpoint_error(error: CheckpointError) -> FfiError {
    FfiError::Checkpoint(error.to_string())
}

fn clear_last_error() {
    LAST_ERROR.with(|slot| {
        let mut buffer = slot.borrow_mut();
        buffer.truncate(1);
        buffer[0] = 0;
    });
}

fn set_last_error(message: impl fmt::Display) {
    let text = message.to_string();
    LAST_ERROR.with(|slot| {
        let mut buffer = slot.borrow_mut();
        buffer.clear();
        // Interior NUL bytes would truncate the C string, so escape them.
        for &byte in text.as_bytes() {
            if byte == 0 {
                buffer.extend_from_slice(b"\\0");
            } else {
                buffer.push(byte);
            }
        }
        buffer.push(0);
    });
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        format!("panic in cb2vec FFI call: {message}")
    } else if let Some(message) = payload.downcast_ref::<String>() {
        format!("panic in cb2vec FFI call: {message}")
    } else {
        "panic in cb2vec FFI call".to_string()
    }
}

fn ffi_guard(function: impl FnOnce() -> Result<(), FfiError>) -> i32 {
    clear_last_error();
    match catch_unwind(AssertUnwindSafe(function)) {
        Ok(Ok(())) => CB2VEC_OK,
        Ok(Err(error)) => {
            let status = error.status();
            set_last_error(error);
            status
        }
        Err(payload) => {
            set_last_error(panic_message(payload));
            CB2VEC_ERROR_PANIC
        }
    }
}

fn validate_pointer<T>(pointer: *const T, name: &'static str) -> Result<(), FfiError> {
    if pointer.is_null() {
        return Err(FfiError::Null(name));
    }
    if (pointer as usize) % align_of::<T>() != 0 {
        return Err(FfiError::Invalid(format!("{name} is misaligned")));
    }
    Ok(())
}

fn validate_mut_pointer<T>(pointer: *mut T, name: &'static str) -> Result<(), FfiError> {
    validate_pointer(pointer.cast_const(), name)
}

fn validate_slice_len<T>(len: usize, name: &'static str) -> Result<(), FfiError> {
    let bytes = len
        .checked_mul(size_of::<T>())
        .ok_or_else(|| FfiError::Invalid(format!("{name} byte length overflow")))?;
    if bytes > isize::MAX as usize {
        return Err(FfiError::Invalid(format!(
            "{name} exceeds the maximum Rust slice size"
        )));
    }
    Ok(())
}

unsafe fn raw_slice<'a, T>(
    pointer: *const T,
    len: usize,
    name: &'static str,
) -> Result<&'a [T], FfiError> {
    validate_slice_len::<T>(len, name)?;
    if len == 0 {
        return Ok(&[]);
    }
    validate_pointer(pointer, name)?;
    // SAFETY: The caller contract guarantees a live readable allocation for
    // `len` elements. Null, alignment, and representable byte length were
    // checked above; the slice is retained only for this FFI call.
    Ok(unsafe { slice::from_raw_parts(pointer, len) })
}

unsafe fn raw_mut_slice<'a, T>(
    pointer: *mut T,
    len: usize,
    name: &'static str,
) -> Result<&'a mut [T], FfiError> {
    validate_slice_len::<T>(len, name)?;
    if len == 0 {
        return Ok(&mut []);
    }
    validate_mut_pointer(pointer, name)?;
    // SAFETY: The caller contract guarantees an exclusive live writable
    // allocation for `len` elements. The validated slice is call-scoped.
    Ok(unsafe { slice::from_raw_parts_mut(pointer, len) })
}

unsafe fn read_copy<T: Copy>(pointer: *const T, name: &'static str) -> Result<T, FfiError> {
    validate_pointer(pointer, name)?;
    // SAFETY: The pointer is caller-provided readable storage for one T and
    // has been checked for null and alignment.
    Ok(unsafe { ptr::read(pointer) })
}

unsafe fn write_out<T>(pointer: *mut T, value: T, name: &'static str) -> Result<(), FfiError> {
    validate_mut_pointer(pointer, name)?;
    // SAFETY: The pointer is caller-provided exclusive writable storage for
    // one T and has been checked for null and alignment.
    unsafe { ptr::write(pointer, value) };
    Ok(())
}

unsafe fn trainer_ref<'a>(pointer: *const Cb2VecTrainer) -> Result<&'a Cb2VecTrainer, FfiError> {
    validate_pointer(pointer, "trainer")?;
    // SAFETY: A non-null, aligned pointer returned by trainer_new_v1 remains
    // valid until its matching free call; that ownership is a caller contract.
    Ok(unsafe { &*pointer })
}

unsafe fn trainer_mut<'a>(pointer: *mut Cb2VecTrainer) -> Result<&'a mut Cb2VecTrainer, FfiError> {
    validate_mut_pointer(pointer, "trainer")?;
    // SAFETY: The caller guarantees exclusive access to a live trainer handle
    // for the duration of this mutating call.
    Ok(unsafe { &mut *pointer })
}

unsafe fn weights_ref<'a>(pointer: *const Cb2VecWeights) -> Result<&'a Cb2VecWeights, FfiError> {
    validate_pointer(pointer, "weights")?;
    // SAFETY: A non-null, aligned pointer returned by a weights constructor
    // remains valid until its matching free call.
    Ok(unsafe { &*pointer })
}

unsafe fn session_ref<'a>(pointer: *const Cb2VecSession) -> Result<&'a Cb2VecSession, FfiError> {
    validate_pointer(pointer, "session")?;
    // SAFETY: A non-null, aligned pointer returned by cb2vec_session_create_v1
    // remains valid until its matching free call.
    Ok(unsafe { &*pointer })
}

unsafe fn session_mut<'a>(pointer: *mut Cb2VecSession) -> Result<&'a mut Cb2VecSession, FfiError> {
    validate_mut_pointer(pointer, "session")?;
    // SAFETY: A session is single-owner, so the caller guarantees exclusive
    // access to a live handle for the duration of this mutating call.
    Ok(unsafe { &mut *pointer })
}

fn activation_from_ffi(value: u32) -> Result<Activation, FfiError> {
    match value {
        CB2VEC_ACTIVATION_IDENTITY => Ok(Activation::Identity),
        CB2VEC_ACTIVATION_RELU => Ok(Activation::Relu),
        _ => Err(FfiError::Invalid(format!(
            "unsupported activation value {value}"
        ))),
    }
}

const fn activation_to_ffi(value: Activation) -> u32 {
    match value {
        Activation::Identity => CB2VEC_ACTIVATION_IDENTITY,
        Activation::Relu => CB2VEC_ACTIVATION_RELU,
    }
}

fn pooling_from_ffi(value: u32) -> Result<Pooling, FfiError> {
    match value {
        CB2VEC_POOLING_SUM => Ok(Pooling::Sum),
        CB2VEC_POOLING_MEAN => Ok(Pooling::Mean),
        _ => Err(FfiError::Invalid(format!(
            "unsupported pooling value {value}"
        ))),
    }
}

const fn pooling_to_ffi(value: Pooling) -> u32 {
    match value {
        Pooling::Sum => CB2VEC_POOLING_SUM,
        Pooling::Mean => CB2VEC_POOLING_MEAN,
    }
}

fn loss_from_ffi(value: u32) -> Result<Loss, FfiError> {
    match value {
        CB2VEC_LOSS_BCE_WITH_LOGITS => Ok(Loss::BinaryCrossEntropyWithLogits),
        CB2VEC_LOSS_MSE => Ok(Loss::MeanSquaredError),
        _ => Err(FfiError::Invalid(format!("unsupported loss value {value}"))),
    }
}

const fn loss_to_ffi(value: Loss) -> u32 {
    match value {
        Loss::BinaryCrossEntropyWithLogits => CB2VEC_LOSS_BCE_WITH_LOGITS,
        Loss::MeanSquaredError => CB2VEC_LOSS_MSE,
    }
}

/// Accepts any ABI revision this build is source-compatible with.
///
/// Minor revisions are additive, so a caller built against ABI 1.0 keeps
/// working against a 1.1 library. A newer minor than this build understands,
/// or any other major, is rejected.
fn check_abi_version(value: u32, what: &'static str) -> Result<(), FfiError> {
    let major = value >> 16;
    let minor = value & 0xFFFF;
    if major != CB2VEC_ABI_VERSION >> 16 || minor > (CB2VEC_ABI_VERSION & 0xFFFF) {
        return Err(FfiError::Abi(format!(
            "{what} ABI is 0x{value:08x}, but this build supports 0x{CB2VEC_ABI_VERSION:08x}"
        )));
    }
    Ok(())
}

fn decode_shape(shape: Cb2VecModelShapeV1) -> Result<ModelShape, FfiError> {
    if shape.struct_size != size_of::<Cb2VecModelShapeV1>() as u32 {
        return Err(FfiError::Abi(format!(
            "model shape size is {}, expected {}",
            shape.struct_size,
            size_of::<Cb2VecModelShapeV1>()
        )));
    }
    check_abi_version(shape.abi_version, "model shape")?;
    if shape.reserved != [0; 2] {
        return Err(FfiError::Abi(
            "model shape reserved fields must be zero".to_string(),
        ));
    }
    ModelShape::new(
        shape.token_count as usize,
        shape.group_count as usize,
        shape.dim as usize,
        shape.fm_rank as usize,
    )
    .map_err(|error| FfiError::Model(error.to_string()))
}

fn decode_config(config: Cb2VecTrainerConfigV1) -> Result<TrainerConfig, FfiError> {
    if config.struct_size != size_of::<Cb2VecTrainerConfigV1>() as u32 {
        return Err(FfiError::Abi(format!(
            "trainer config size is {}, expected {}",
            config.struct_size,
            size_of::<Cb2VecTrainerConfigV1>()
        )));
    }
    check_abi_version(config.abi_version, "trainer config")?;
    if config.flags != 0 || config.reserved != [0; 2] {
        return Err(FfiError::Abi(
            "trainer config flags and reserved fields must be zero".to_string(),
        ));
    }
    let shuffle = match config.shuffle {
        0 => false,
        1 => true,
        value => {
            return Err(FfiError::Invalid(format!(
                "shuffle must be 0 or 1, got {value}"
            )));
        }
    };
    let trainer = TrainerConfig {
        activation: activation_from_ffi(config.activation)?,
        pooling: pooling_from_ffi(config.pooling)?,
        loss: loss_from_ffi(config.loss)?,
        adam: AdamConfig {
            learning_rate: config.learning_rate,
            beta1: config.beta1,
            beta2: config.beta2,
            epsilon: config.epsilon,
        },
        batch_size: config.batch_size as usize,
        shuffle,
        seed: config.seed,
    };
    Ok(trainer)
}

fn decode_quantization(
    config: Cb2VecQuantizationConfigV1,
) -> Result<Cb2VecQuantizationConfigV1, FfiError> {
    if config.struct_size != size_of::<Cb2VecQuantizationConfigV1>() as u32 {
        return Err(FfiError::Abi(format!(
            "quantization config size is {}, expected {}",
            config.struct_size,
            size_of::<Cb2VecQuantizationConfigV1>()
        )));
    }
    check_abi_version(config.abi_version, "quantization config")?;
    if config.flags != 0 || config.reserved != [0; 2] {
        return Err(FfiError::Abi(
            "quantization flags and reserved fields must be zero".to_string(),
        ));
    }
    if config.embedding_scale <= 0 || config.head_scale <= 0 || config.factor_scale <= 0 {
        return Err(FfiError::Invalid(
            "quantization scales must all be positive".to_string(),
        ));
    }
    Ok(config)
}

fn decode_inference(config: Cb2VecInferenceConfigV1) -> Result<InferenceConfig, FfiError> {
    if config.struct_size != size_of::<Cb2VecInferenceConfigV1>() as u32 {
        return Err(FfiError::Abi(format!(
            "inference config size is {}, expected {}",
            config.struct_size,
            size_of::<Cb2VecInferenceConfigV1>()
        )));
    }
    if config.flags != 0 {
        return Err(FfiError::Abi(
            "inference config flags must be zero".to_string(),
        ));
    }
    Ok(InferenceConfig::new(
        activation_from_ffi(config.activation)?,
        pooling_from_ffi(config.pooling)?,
    ))
}

fn decode_metadata(metadata: Cb2VecArtifactMetadataV1) -> Result<ArtifactMetadata, FfiError> {
    if metadata.struct_size != size_of::<Cb2VecArtifactMetadataV1>() as u32 {
        return Err(FfiError::Abi(format!(
            "artifact metadata size is {}, expected {}",
            metadata.struct_size,
            size_of::<Cb2VecArtifactMetadataV1>()
        )));
    }
    check_abi_version(metadata.abi_version, "artifact metadata")?;
    if metadata.flags != 0 {
        return Err(FfiError::Abi(
            "artifact metadata flags must be zero".to_string(),
        ));
    }
    Ok(ArtifactMetadata::new(
        metadata.schema_version,
        metadata.schema_digest,
    ))
}

fn decode_session_config(config: Cb2VecSessionConfigV1) -> Result<SessionLimits, FfiError> {
    if config.struct_size != size_of::<Cb2VecSessionConfigV1>() as u32 {
        return Err(FfiError::Abi(format!(
            "session config size is {}, expected {}",
            config.struct_size,
            size_of::<Cb2VecSessionConfigV1>()
        )));
    }
    check_abi_version(config.abi_version, "session config")?;
    if config.flags != 0 || config.reserved != [0; 1] {
        return Err(FfiError::Abi(
            "session config flags and reserved fields must be zero".to_string(),
        ));
    }
    Ok(SessionLimits::new(
        config.max_sites as usize,
        config.max_token_slots as usize,
        config.max_deltas_per_frame as usize,
        config.max_depth as usize,
    ))
}

fn map_training_error(error: TrainingError) -> FfiError {
    let message = error.to_string();
    match error {
        TrainingError::Model(_) => FfiError::Model(message),
        TrainingError::NonFiniteComputation(_) => FfiError::Numeric(message),
        _ => FfiError::Invalid(message),
    }
}

unsafe fn samples_from_batch(
    batch: *const Cb2VecTrainingBatchV1,
) -> Result<Vec<TrainingSample>, FfiError> {
    let batch =
        // SAFETY: The caller supplies one complete, aligned batch descriptor.
        unsafe { read_copy(batch, "batch")? };
    if batch.struct_size != size_of::<Cb2VecTrainingBatchV1>() as u32 {
        return Err(FfiError::Abi(format!(
            "training batch size is {}, expected {}",
            batch.struct_size,
            size_of::<Cb2VecTrainingBatchV1>()
        )));
    }
    if batch.flags != 0 || batch.reserved != 0 {
        return Err(FfiError::Abi(
            "training batch flags and reserved fields must be zero".to_string(),
        ));
    }
    // SAFETY: Descriptor pointers and fixed-width lengths are validated and
    // copied by training_samples_from_raw.
    unsafe {
        training_samples_from_raw(
            batch.tokens,
            batch.tokens_len,
            batch.site_token_offsets,
            batch.site_groups,
            batch.site_count,
            batch.sample_site_offsets,
            batch.targets,
            batch.weights,
            batch.sample_count,
        )
    }
}

unsafe fn inputs_from_batch(
    batch: *const Cb2VecTrainingBatchV1,
) -> Result<Vec<GroupedTokens>, FfiError> {
    let batch =
        // SAFETY: The caller supplies one complete, aligned batch descriptor.
        unsafe { read_copy(batch, "batch")? };
    if batch.struct_size != size_of::<Cb2VecTrainingBatchV1>() as u32 {
        return Err(FfiError::Abi(format!(
            "training batch size is {}, expected {}",
            batch.struct_size,
            size_of::<Cb2VecTrainingBatchV1>()
        )));
    }
    if batch.flags != 0 || batch.reserved != 0 {
        return Err(FfiError::Abi(
            "training batch flags and reserved fields must be zero".to_string(),
        ));
    }
    // SAFETY: Descriptor pointers and fixed-width lengths are validated and
    // copied by batch_inputs_from_raw. Targets and weights are intentionally
    // ignored for inference.
    unsafe {
        batch_inputs_from_raw(
            batch.tokens,
            batch.tokens_len,
            batch.site_token_offsets,
            batch.site_groups,
            batch.site_count,
            batch.sample_site_offsets,
            batch.sample_count,
        )
    }
}

unsafe fn grouped_input_from_raw(
    tokens: *const u16,
    tokens_len: u32,
    site_offsets: *const u32,
    site_groups: *const u32,
    site_count: u32,
) -> Result<GroupedTokens, FfiError> {
    let token_values =
        // SAFETY: Forwarded caller buffer; raw_slice performs structural checks.
        unsafe { raw_slice(tokens, tokens_len as usize, "tokens")? }.to_vec();
    let site_count = site_count as usize;
    let offset_count = site_count
        .checked_add(1)
        .ok_or_else(|| FfiError::Invalid("site offset count overflow".to_string()))?;
    let offsets =
        // SAFETY: Forwarded caller buffer with the derived prefix-table length.
        unsafe { raw_slice(site_offsets, offset_count, "site_offsets")? }
            .iter()
            .map(|&value| value as usize)
            .collect();
    let groups =
        // SAFETY: Forwarded caller buffer with one group per site.
        unsafe { raw_slice(site_groups, site_count, "site_groups")? }
            .iter()
            .map(|&value| value as usize)
            .collect();
    GroupedTokens::new(token_values, offsets, groups)
        .map_err(|error| FfiError::Invalid(error.to_string()))
}

#[allow(clippy::too_many_arguments)]
unsafe fn batch_inputs_from_raw(
    tokens: *const u16,
    tokens_len: u32,
    site_offsets: *const u32,
    site_groups: *const u32,
    total_site_count: u32,
    sample_site_offsets: *const u32,
    sample_count: u32,
) -> Result<Vec<GroupedTokens>, FfiError> {
    if sample_count == 0 {
        return Err(FfiError::Invalid(
            "sample_count must be non-zero".to_string(),
        ));
    }
    let tokens =
        // SAFETY: Forwarded caller buffer; raw_slice performs structural checks.
        unsafe { raw_slice(tokens, tokens_len as usize, "tokens")? };
    let total_site_count = total_site_count as usize;
    let site_offset_count = total_site_count
        .checked_add(1)
        .ok_or_else(|| FfiError::Invalid("site offset count overflow".to_string()))?;
    let site_offsets =
        // SAFETY: Prefix table has exactly total_site_count + 1 entries.
        unsafe { raw_slice(site_offsets, site_offset_count, "site_offsets")? };
    let site_groups =
        // SAFETY: There is one group value per global site.
        unsafe { raw_slice(site_groups, total_site_count, "site_groups")? };
    let sample_count = sample_count as usize;
    let sample_offset_count = sample_count
        .checked_add(1)
        .ok_or_else(|| FfiError::Invalid("sample offset count overflow".to_string()))?;
    let sample_site_offsets =
        // SAFETY: Prefix table has exactly sample_count + 1 entries.
        unsafe {
            raw_slice(
                sample_site_offsets,
                sample_offset_count,
                "sample_site_offsets",
            )?
        };

    if site_offsets.first().copied() != Some(0) {
        return Err(FfiError::Invalid(
            "site_offsets must start at zero".to_string(),
        ));
    }
    if site_offsets.last().copied().map(|value| value as usize) != Some(tokens.len()) {
        return Err(FfiError::Invalid(format!(
            "site_offsets must end at tokens_len {}",
            tokens.len()
        )));
    }
    if site_offsets.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(FfiError::Invalid(
            "site_offsets must be monotonic".to_string(),
        ));
    }
    if sample_site_offsets.first().copied() != Some(0) {
        return Err(FfiError::Invalid(
            "sample_site_offsets must start at zero".to_string(),
        ));
    }
    if sample_site_offsets
        .last()
        .copied()
        .map(|value| value as usize)
        != Some(total_site_count)
    {
        return Err(FfiError::Invalid(format!(
            "sample_site_offsets must end at total_site_count {total_site_count}"
        )));
    }
    if sample_site_offsets.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(FfiError::Invalid(
            "sample_site_offsets must be monotonic".to_string(),
        ));
    }

    let mut inputs = Vec::with_capacity(sample_count);
    for sample in 0..sample_count {
        let first_site = sample_site_offsets[sample] as usize;
        let last_site = sample_site_offsets[sample + 1] as usize;
        let first_token = site_offsets[first_site] as usize;
        let last_token = site_offsets[last_site] as usize;
        let local_offsets = site_offsets[first_site..=last_site]
            .iter()
            .map(|&offset| offset as usize - first_token)
            .collect();
        let local_groups = site_groups[first_site..last_site]
            .iter()
            .map(|&group| group as usize)
            .collect();
        let input = GroupedTokens::new(
            tokens[first_token..last_token].to_vec(),
            local_offsets,
            local_groups,
        )
        .map_err(|error| FfiError::Invalid(format!("sample {sample}: {error}")))?;
        inputs.push(input);
    }
    Ok(inputs)
}

#[allow(clippy::too_many_arguments)]
unsafe fn training_samples_from_raw(
    tokens: *const u16,
    tokens_len: u32,
    site_offsets: *const u32,
    site_groups: *const u32,
    total_site_count: u32,
    sample_site_offsets: *const u32,
    targets: *const f32,
    weights: *const f32,
    sample_count: u32,
) -> Result<Vec<TrainingSample>, FfiError> {
    let inputs =
        // SAFETY: The caller buffers and their derived lengths are forwarded.
        unsafe {
            batch_inputs_from_raw(
                tokens,
                tokens_len,
                site_offsets,
                site_groups,
                total_site_count,
                sample_site_offsets,
                sample_count,
            )?
        };
    let targets =
        // SAFETY: There is one target per decoded sample.
        unsafe { raw_slice(targets, sample_count as usize, "targets")? };
    let weights = if weights.is_null() {
        None
    } else {
        Some(
            // SAFETY: A non-null optional weight buffer has one value per sample.
            unsafe { raw_slice(weights, sample_count as usize, "sample_weights")? },
        )
    };
    Ok(inputs
        .into_iter()
        .enumerate()
        .map(|(index, input)| {
            TrainingSample::weighted(
                input,
                targets[index],
                weights.map_or(1.0, |values| values[index]),
            )
        })
        .collect())
}

fn payload_scales(
    payload: &PackedQuantizedPayload,
) -> Result<(ModelShape, i32, i32, i32), FfiError> {
    Ok(match payload {
        PackedQuantizedPayload::Flat(model) => (
            model
                .validate()
                .map_err(|error| FfiError::Model(error.to_string()))?,
            model.embedding_scale,
            model.head_scale,
            model.factor_scale,
        ),
        PackedQuantizedPayload::Factored(model) => (
            model
                .validate()
                .map_err(|error| FfiError::Model(error.to_string()))?,
            model.embedding_scale(),
            model.head_scale(),
            model.factor_scale(),
        ),
    })
}

fn quantized_info(weights: &Cb2VecWeights) -> Result<Cb2VecModelInfoV1, FfiError> {
    let (shape, embedding_scale, head_scale, factor_scale) = payload_scales(&weights.payload)?;
    let mut info = model_info(
        shape,
        weights.original_kind,
        weights.flags,
        weights.inference,
        embedding_scale,
        head_scale,
        factor_scale,
    )?;
    info.artifact_version = weights.artifact_version;
    Ok(info)
}

fn model_info(
    shape: ModelShape,
    kind: u32,
    flags: u32,
    inference: InferenceConfig,
    embedding_scale: i32,
    head_scale: i32,
    factor_scale: i32,
) -> Result<Cb2VecModelInfoV1, FfiError> {
    Ok(Cb2VecModelInfoV1 {
        token_count: shape_field(shape.token_count(), "token_count")?,
        group_count: shape_field(shape.group_count(), "group_count")?,
        dim: shape_field(shape.dim(), "dim")?,
        fm_rank: shape_field(shape.fm_rank(), "fm_rank")?,
        kind,
        flags,
        activation: activation_to_ffi(inference.activation),
        pooling: pooling_to_ffi(inference.pooling),
        embedding_scale,
        head_scale,
        factor_scale,
        ..Cb2VecModelInfoV1::default()
    })
}

fn shape_field(value: usize, name: &'static str) -> Result<u32, FfiError> {
    u32::try_from(value).map_err(|_| FfiError::Model(format!("{name} does not fit u32")))
}

fn predict_payload(input: &GroupedTokens, weights: &Cb2VecWeights) -> Result<f32, FfiError> {
    match &*weights.payload {
        PackedQuantizedPayload::Flat(model) => {
            predict_quantized(input, model, weights.inference).map_err(map_training_error)
        }
        PackedQuantizedPayload::Factored(model) => {
            predict_quantized(input, model, weights.inference).map_err(map_training_error)
        }
    }
}

/// Returns [`CB2VEC_ABI_VERSION`] without changing the thread-local error.
#[unsafe(no_mangle)]
pub extern "C" fn cb2vec_abi_version() -> u32 {
    CB2VEC_ABI_VERSION
}

/// Returns a process-lifetime UTF-8 version string.
#[unsafe(no_mangle)]
pub extern "C" fn cb2vec_library_version() -> *const c_char {
    LIBRARY_VERSION.as_ptr().cast()
}

/// Returns the current thread's last error. Copy it before the next ABI call.
#[unsafe(no_mangle)]
pub extern "C" fn cb2vec_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr().cast::<c_char>())
}

/// Writes the ABI-v1 default trainer configuration.
///
/// # Safety
///
/// `out_config` must be aligned, writable storage for one complete
/// [`Cb2VecTrainerConfigV1`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_trainer_config_default_v1(
    out_config: *mut Cb2VecTrainerConfigV1,
) -> i32 {
    ffi_guard(|| {
        // SAFETY: Required by this exported function's caller contract.
        unsafe { write_out(out_config, Cb2VecTrainerConfigV1::default(), "out_config") }
    })
}

/// Writes the ABI-v1 default model shape.
///
/// # Safety
///
/// `out_shape` must be aligned, writable storage for one complete
/// [`Cb2VecModelShapeV1`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_model_shape_default_v1(out_shape: *mut Cb2VecModelShapeV1) -> i32 {
    ffi_guard(|| {
        // SAFETY: Required by this exported function's caller contract.
        unsafe { write_out(out_shape, Cb2VecModelShapeV1::default(), "out_shape") }
    })
}

/// Writes the ABI-v1 default PTQ configuration.
///
/// # Safety
///
/// `out_config` must be aligned, writable storage for one complete
/// [`Cb2VecQuantizationConfigV1`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_quantization_config_default_v1(
    out_config: *mut Cb2VecQuantizationConfigV1,
) -> i32 {
    ffi_guard(|| {
        // SAFETY: Required by this exported function's caller contract.
        unsafe {
            write_out(
                out_config,
                Cb2VecQuantizationConfigV1::default(),
                "out_config",
            )
        }
    })
}

/// Writes the ABI-v1 default inference recipe.
///
/// # Safety
///
/// `out_config` must be aligned, writable storage for one complete
/// [`Cb2VecInferenceConfigV1`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_inference_config_default_v1(
    out_config: *mut Cb2VecInferenceConfigV1,
) -> i32 {
    ffi_guard(|| {
        // SAFETY: Required by this exported function's caller contract.
        unsafe { write_out(out_config, Cb2VecInferenceConfigV1::default(), "out_config") }
    })
}

/// Creates a deterministic FP32 trainer.
///
/// # Safety
///
/// `shape` and `config` must point to complete ABI-v1 structures.
/// `out_trainer` must be writable pointer storage. The returned handle must be
/// freed exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_trainer_create_v1(
    shape: *const Cb2VecModelShapeV1,
    config: *const Cb2VecTrainerConfigV1,
    out_trainer: *mut *mut Cb2VecTrainer,
) -> i32 {
    ffi_guard(|| {
        // SAFETY: Validated output storage is initialized before fallible work.
        unsafe { write_out(out_trainer, ptr::null_mut(), "out_trainer")? };
        // SAFETY: Required by the caller contract and checked for alignment/null.
        let shape = decode_shape(
            // SAFETY: Required by the caller contract and checked for alignment/null.
            unsafe { read_copy(shape, "shape")? },
        )?;
        let config = decode_config(
            // SAFETY: Required by the caller contract and checked for alignment/null.
            unsafe { read_copy(config, "config")? },
        )?;
        let trainer = Trainer::from_shape(shape, config)
            .map_err(|error| FfiError::Model(error.to_string()))?;
        let handle = Box::into_raw(Box::new(Cb2VecTrainer { trainer }));
        // SAFETY: Output storage was validated above and remains exclusively writable.
        unsafe { ptr::write(out_trainer, handle) };
        Ok(())
    })
}

/// Frees a trainer handle. A null handle is a successful no-op.
///
/// # Safety
///
/// A non-null handle must have been returned by a trainer constructor, must
/// not have been freed, and must not be in use by another call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_trainer_free_v1(trainer: *mut Cb2VecTrainer) -> i32 {
    ffi_guard(|| {
        if !trainer.is_null() {
            validate_mut_pointer(trainer, "trainer")?;
            // SAFETY: The caller transfers the unique live Box allocation back
            // exactly once, as required by this function's contract.
            drop(unsafe { Box::from_raw(trainer) });
        }
        Ok(())
    })
}

/// Returns FP32 trainer shape, inference recipe, and progress-independent metadata.
///
/// # Safety
///
/// `trainer` must be a live handle and `out_info` writable for one info struct.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_trainer_get_info_v1(
    trainer: *const Cb2VecTrainer,
    out_info: *mut Cb2VecModelInfoV1,
) -> i32 {
    ffi_guard(|| {
        // SAFETY: Caller supplies output storage; initialize it before work.
        unsafe { write_out(out_info, Cb2VecModelInfoV1::default(), "out_info")? };
        // SAFETY: Caller guarantees a live trainer handle.
        let trainer = unsafe { trainer_ref(trainer)? };
        let shape = trainer
            .trainer
            .weights()
            .validate()
            .map_err(|error| FfiError::Model(error.to_string()))?;
        let info = model_info(
            shape,
            CB2VEC_MODEL_KIND_FP32,
            0,
            trainer.trainer.inference_config(),
            0,
            0,
            0,
        )?;
        // SAFETY: Output storage was validated above.
        unsafe { ptr::write(out_info, info) };
        Ok(())
    })
}

/// Predicts one raw FP32 trainer logit.
///
/// # Safety
///
/// The handle must be live. Every non-empty input buffer must remain readable
/// for the call. `out_score` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_trainer_predict_logit_v1(
    trainer: *const Cb2VecTrainer,
    tokens: *const u16,
    tokens_len: u32,
    site_offsets: *const u32,
    site_groups: *const u32,
    site_count: u32,
    out_score: *mut f32,
) -> i32 {
    ffi_guard(|| {
        // SAFETY: Caller supplies one writable output scalar.
        unsafe { write_out(out_score, 0.0, "out_score")? };
        // SAFETY: Input buffers are copied and validated during this call.
        let input = unsafe {
            grouped_input_from_raw(tokens, tokens_len, site_offsets, site_groups, site_count)?
        };
        // SAFETY: Caller guarantees a live trainer handle.
        let trainer = unsafe { trainer_ref(trainer)? };
        let score = trainer
            .trainer
            .predict_logit(&input)
            .map_err(map_training_error)?;
        // SAFETY: Output storage was validated above.
        unsafe { ptr::write(out_score, score) };
        Ok(())
    })
}

/// Predicts one stable-sigmoid FP32 trainer probability.
///
/// # Safety
///
/// The pointer contract is identical to `cb2vec_trainer_predict_logit_v1`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_trainer_predict_probability_v1(
    trainer: *const Cb2VecTrainer,
    tokens: *const u16,
    tokens_len: u32,
    site_offsets: *const u32,
    site_groups: *const u32,
    site_count: u32,
    out_probability: *mut f32,
) -> i32 {
    ffi_guard(|| {
        // SAFETY: Caller supplies one writable output scalar.
        unsafe { write_out(out_probability, 0.0, "out_probability")? };
        // SAFETY: Input buffers are copied and validated during this call.
        let input = unsafe {
            grouped_input_from_raw(tokens, tokens_len, site_offsets, site_groups, site_count)?
        };
        // SAFETY: Caller guarantees a live trainer handle.
        let trainer = unsafe { trainer_ref(trainer)? };
        let probability = trainer
            .trainer
            .predict_probability(&input)
            .map_err(map_training_error)?;
        // SAFETY: Output storage was validated above.
        unsafe { ptr::write(out_probability, probability) };
        Ok(())
    })
}

/// Restores FP32 source weights from an artifact and starts fresh optimizer state.
///
/// Artifact v1 does not store the inference recipe, Adam moments, shuffle RNG,
/// or epoch progress. The supplied trainer config defines those values.
///
/// # Safety
///
/// `artifact` must be readable for `artifact_len` bytes. `config` must point
/// to a complete config and `out_trainer` to writable pointer storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_trainer_load_artifact_v1(
    artifact: *const u8,
    artifact_len: u32,
    config: *const Cb2VecTrainerConfigV1,
    out_trainer: *mut *mut Cb2VecTrainer,
) -> i32 {
    ffi_guard(|| {
        // SAFETY: Initialize caller output before fallible parsing.
        unsafe { write_out(out_trainer, ptr::null_mut(), "out_trainer")? };
        let config = decode_config(
            // SAFETY: Caller supplies one complete config.
            unsafe { read_copy(config, "config")? },
        )?;
        let bytes =
            // SAFETY: Caller keeps artifact storage alive for this call.
            unsafe { raw_slice(artifact, artifact_len as usize, "artifact")? };
        let artifact = PackedCodebookArtifact::parse(bytes)
            .map_err(|error| FfiError::Artifact(error.to_string()))?;
        let (source_weights, _) = artifact.into_parts();
        let trainer = Trainer::new(source_weights, config)
            .map_err(|error| FfiError::Model(error.to_string()))?;
        let handle = Box::into_raw(Box::new(Cb2VecTrainer { trainer }));
        // SAFETY: Output storage was validated and initialized above.
        unsafe { ptr::write(out_trainer, handle) };
        Ok(())
    })
}

/// Evaluates a flattened weighted dataset without mutating the trainer.
///
/// # Safety
///
/// `trainer` must be live, `batch` and all non-empty buffers it references
/// must remain readable for the call, and `out_metrics` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_trainer_evaluate_v1(
    trainer: *const Cb2VecTrainer,
    batch: *const Cb2VecTrainingBatchV1,
    out_metrics: *mut Cb2VecTrainingMetricsV1,
) -> i32 {
    ffi_guard(|| {
        // SAFETY: Initialize output before any fallible work.
        unsafe {
            write_out(
                out_metrics,
                Cb2VecTrainingMetricsV1::default(),
                "out_metrics",
            )?
        };
        // SAFETY: The complete flattened view is copied before use.
        let samples = unsafe { samples_from_batch(batch)? };
        // SAFETY: Caller guarantees a live trainer handle.
        let trainer = unsafe { trainer_ref(trainer)? };
        let metrics = trainer
            .trainer
            .evaluate(&samples)
            .map_err(map_training_error)?;
        // SAFETY: Output storage was validated above.
        unsafe { ptr::write(out_metrics, metrics.into()) };
        Ok(())
    })
}

/// Applies exactly one Adam update to the supplied flattened batch.
///
/// # Safety
///
/// The trainer must be exclusively owned for this call. Batch buffers must be
/// readable and `out_metrics` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_trainer_train_batch_v1(
    trainer: *mut Cb2VecTrainer,
    batch: *const Cb2VecTrainingBatchV1,
    out_metrics: *mut Cb2VecTrainingMetricsV1,
) -> i32 {
    ffi_guard(|| {
        // SAFETY: Initialize output before decoding or mutation.
        unsafe {
            write_out(
                out_metrics,
                Cb2VecTrainingMetricsV1::default(),
                "out_metrics",
            )?
        };
        // SAFETY: The complete flattened view is copied before trainer mutation.
        let samples = unsafe { samples_from_batch(batch)? };
        // SAFETY: Caller guarantees exclusive access to the live trainer.
        let trainer = unsafe { trainer_mut(trainer)? };
        let metrics = trainer
            .trainer
            .train_batch(&samples)
            .map_err(map_training_error)?;
        // SAFETY: Output storage was validated above.
        unsafe { ptr::write(out_metrics, metrics.into()) };
        Ok(())
    })
}

/// Trains one epoch with the configured batch size and deterministic shuffle.
///
/// If a numerical error occurs in a later mini-batch, updates from earlier
/// mini-batches in the same epoch may remain. Structural input validation is
/// completed before any update.
///
/// # Safety
///
/// The trainer must be exclusively owned for this call. Dataset buffers must
/// be readable and `out_metrics` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_trainer_train_epoch_v1(
    trainer: *mut Cb2VecTrainer,
    dataset: *const Cb2VecTrainingBatchV1,
    out_metrics: *mut Cb2VecTrainingMetricsV1,
) -> i32 {
    ffi_guard(|| {
        // SAFETY: Initialize output before decoding or mutation.
        unsafe {
            write_out(
                out_metrics,
                Cb2VecTrainingMetricsV1::default(),
                "out_metrics",
            )?
        };
        // SAFETY: The complete flattened view is copied before trainer mutation.
        let samples = unsafe { samples_from_batch(dataset)? };
        // SAFETY: Caller guarantees exclusive access to the live trainer.
        let trainer = unsafe { trainer_mut(trainer)? };
        let metrics = trainer
            .trainer
            .train_epoch(&samples)
            .map_err(map_training_error)?;
        // SAFETY: Output storage was validated above.
        unsafe { ptr::write(out_metrics, metrics.into()) };
        Ok(())
    })
}

/// Quantizes the trainer into an independent immutable model handle.
///
/// # Safety
///
/// The trainer and quantization config pointers must be readable.
/// `out_model` must be writable pointer storage and its returned handle must
/// be freed exactly once with `cb2vec_model_free_v1`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_trainer_quantize_v1(
    trainer: *const Cb2VecTrainer,
    quantization: *const Cb2VecQuantizationConfigV1,
    out_model: *mut *mut Cb2VecWeights,
) -> i32 {
    ffi_guard(|| {
        // SAFETY: Initialize output before fallible work.
        unsafe { write_out(out_model, ptr::null_mut(), "out_model")? };
        let quantization = decode_quantization(
            // SAFETY: Caller supplies one complete config.
            unsafe { read_copy(quantization, "quantization")? },
        )?;
        // SAFETY: Caller guarantees a live trainer.
        let trainer = unsafe { trainer_ref(trainer)? };
        let quantized = trainer
            .trainer
            .weights()
            .quantize_i16(
                quantization.embedding_scale,
                quantization.head_scale,
                quantization.factor_scale,
            )
            .map_err(|error| FfiError::Model(error.to_string()))?;
        let weights = Cb2VecWeights {
            payload: Arc::new(PackedQuantizedPayload::Flat(quantized)),
            inference: trainer.trainer.inference_config(),
            original_kind: CB2VEC_MODEL_KIND_FLAT,
            flags: 0,
            artifact_version: u32::from(crate::CB2VEC_ARTIFACT_VERSION),
            metadata: ArtifactMetadata::default(),
        };
        let handle = Box::into_raw(Box::new(weights));
        // SAFETY: Output storage was validated and initialized above.
        unsafe { ptr::write(out_model, handle) };
        Ok(())
    })
}

/// Writes a canonical flat artifact into caller-owned storage.
///
/// A null `out_bytes` with zero capacity is a size probe: the function writes
/// the exact required byte count and returns
/// [`CB2VEC_ERROR_BUFFER_TOO_SMALL`]. No output bytes are written on a short
/// buffer.
///
/// # Safety
///
/// The trainer and quantization config must be readable. `source_sha256_32`
/// must point to 32 readable bytes. `out_required_or_written` must be
/// writable. A non-null output buffer must be writable for at least
/// `out_capacity` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_trainer_write_artifact_v1(
    trainer: *const Cb2VecTrainer,
    quantization: *const Cb2VecQuantizationConfigV1,
    source_sha256_32: *const u8,
    out_bytes: *mut u8,
    out_capacity: u32,
    out_required_or_written: *mut u32,
) -> i32 {
    ffi_guard(|| {
        // SAFETY: Initialize byte count before fallible work.
        unsafe { write_out(out_required_or_written, 0, "out_required_or_written")? };
        let quantization = decode_quantization(
            // SAFETY: Caller supplies one complete config.
            unsafe { read_copy(quantization, "quantization")? },
        )?;
        let digest_bytes =
            // SAFETY: The caller contract requires exactly 32 readable bytes.
            unsafe { raw_slice(source_sha256_32, 32, "source_sha256_32")? };
        let mut digest = [0u8; 32];
        digest.copy_from_slice(digest_bytes);
        // SAFETY: Caller guarantees a live trainer.
        let trainer = unsafe { trainer_ref(trainer)? };
        let quantized = trainer
            .trainer
            .weights()
            .quantize_i16(
                quantization.embedding_scale,
                quantization.head_scale,
                quantization.factor_scale,
            )
            .map_err(|error| FfiError::Model(error.to_string()))?;
        let artifact =
            PackedCodebookArtifact::new_flat(trainer.trainer.weights().clone(), quantized, digest)
                .map_err(|error| FfiError::Artifact(error.to_string()))?;
        let bytes = artifact
            .to_bytes()
            .map_err(|error| FfiError::Artifact(error.to_string()))?;
        let required = u32::try_from(bytes.len())
            .map_err(|_| FfiError::Artifact("artifact exceeds u32 byte length".to_string()))?;
        // SAFETY: Count output was validated above.
        unsafe { ptr::write(out_required_or_written, required) };

        if out_bytes.is_null() {
            if out_capacity == 0 {
                return Err(FfiError::BufferTooSmall {
                    required: bytes.len(),
                    capacity: 0,
                });
            }
            return Err(FfiError::Null("out_bytes"));
        }
        if (out_capacity as usize) < bytes.len() {
            return Err(FfiError::BufferTooSmall {
                required: bytes.len(),
                capacity: out_capacity as usize,
            });
        }
        let output =
            // SAFETY: Caller promised a writable buffer; only required bytes are used.
            unsafe { raw_mut_slice(out_bytes, bytes.len(), "out_bytes")? };
        output.copy_from_slice(&bytes);
        Ok(())
    })
}

/// Parses artifact bytes into an immutable model.
///
/// Version 2 artifacts answer the inference recipe from their own header; a
/// `supplied` recipe is then a cross-check rather than the source of truth.
/// Version 1 artifacts require one. Factored storage is reconstructed once
/// into a flat table for runtime inference speed.
fn load_model(
    bytes: &[u8],
    supplied: Option<InferenceConfig>,
    expected_schema: Option<ArtifactMetadata>,
) -> Result<Cb2VecWeights, FfiError> {
    let artifact = PackedCodebookArtifact::parse(bytes)
        .map_err(|error| FfiError::Artifact(error.to_string()))?;
    let inference = artifact
        .resolve_inference_config(supplied)
        .map_err(|error| FfiError::Artifact(error.to_string()))?;
    if let Some(expected) = expected_schema {
        artifact
            .verify_schema(expected)
            .map_err(|error| FfiError::Artifact(error.to_string()))?;
    }
    let artifact_version = u32::from(artifact.format_version());
    let metadata = artifact.metadata();
    let mut flags = u32::from(artifact.used_legacy_magic()) * CB2VEC_MODEL_FLAG_LEGACY_MAGIC;
    let original_kind = match artifact.kind() {
        crate::PackedCodebookKind::Flat => CB2VEC_MODEL_KIND_FLAT,
        crate::PackedCodebookKind::Factored => CB2VEC_MODEL_KIND_FACTORED,
    };
    let (_, payload) = artifact.into_parts();
    let payload = match payload {
        PackedQuantizedPayload::Flat(weights) => PackedQuantizedPayload::Flat(weights),
        PackedQuantizedPayload::Factored(weights) => {
            flags |= CB2VEC_MODEL_FLAG_FLATTENED_AT_LOAD;
            PackedQuantizedPayload::Flat(weights.reconstruct_flat())
        }
    };
    let model = Cb2VecWeights {
        payload: Arc::new(payload),
        inference,
        original_kind,
        flags,
        artifact_version,
        metadata,
    };
    quantized_info(&model)?;
    Ok(model)
}

/// Loads an immutable quantized model from caller-owned artifact bytes.
///
/// The artifact buffer may be released immediately after the call returns.
/// A version-2 artifact whose stored recipe disagrees with `inference` is
/// rejected; use `cb2vec_model_load_v2` to let the artifact decide.
///
/// # Safety
///
/// Artifact bytes and inference config must be readable. `out_model` must be
/// writable pointer storage. The returned handle must be freed exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_model_load_v1(
    artifact: *const u8,
    artifact_len: u32,
    inference: *const Cb2VecInferenceConfigV1,
    out_model: *mut *mut Cb2VecWeights,
) -> i32 {
    ffi_guard(|| {
        // SAFETY: Initialize output before parsing.
        unsafe { write_out(out_model, ptr::null_mut(), "out_model")? };
        let inference = decode_inference(
            // SAFETY: Caller supplies one complete config.
            unsafe { read_copy(inference, "inference")? },
        )?;
        let bytes =
            // SAFETY: Caller keeps artifact storage live for this call.
            unsafe { raw_slice(artifact, artifact_len as usize, "artifact")? };
        let handle = Box::into_raw(Box::new(load_model(bytes, Some(inference), None)?));
        // SAFETY: Output storage was validated and initialized above.
        unsafe { ptr::write(out_model, handle) };
        Ok(())
    })
}

/// Loads a model, preferring the inference recipe stored in the artifact.
///
/// `inference` may be null. When it is not, a version-2 artifact whose stored
/// recipe disagrees returns [`CB2VEC_ERROR_ARTIFACT`] instead of silently
/// scoring with the wrong activation or pooling. A version-1 artifact still
/// requires a non-null `inference`.
///
/// `expected_schema` may be null. When it is not, an artifact that carries a
/// schema identity must match it exactly. Artifacts that carry none are
/// accepted, so an unlabeled model can still be loaded deliberately.
///
/// # Safety
///
/// Artifact bytes must be readable for `artifact_len` bytes. Non-null
/// `inference` and `expected_schema` must point to complete structs.
/// `out_model` must be writable pointer storage and the returned handle must
/// be freed exactly once with `cb2vec_model_free_v1`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_model_load_v2(
    artifact: *const u8,
    artifact_len: u32,
    inference: *const Cb2VecInferenceConfigV1,
    expected_schema: *const Cb2VecArtifactMetadataV1,
    out_model: *mut *mut Cb2VecWeights,
) -> i32 {
    ffi_guard(|| {
        // SAFETY: Initialize output before parsing.
        unsafe { write_out(out_model, ptr::null_mut(), "out_model")? };
        let supplied = if inference.is_null() {
            None
        } else {
            Some(decode_inference(
                // SAFETY: A non-null pointer must reference one complete config.
                unsafe { read_copy(inference, "inference")? },
            )?)
        };
        let expected = if expected_schema.is_null() {
            None
        } else {
            Some(decode_metadata(
                // SAFETY: A non-null pointer must reference one complete struct.
                unsafe { read_copy(expected_schema, "expected_schema")? },
            )?)
        };
        let bytes =
            // SAFETY: Caller keeps artifact storage live for this call.
            unsafe { raw_slice(artifact, artifact_len as usize, "artifact")? };
        let handle = Box::into_raw(Box::new(load_model(bytes, supplied, expected)?));
        // SAFETY: Output storage was validated and initialized above.
        unsafe { ptr::write(out_model, handle) };
        Ok(())
    })
}

/// Reads artifact metadata without building a model.
///
/// This is the cheap pre-flight check a consumer runs before committing to a
/// download or a load: it reports the format version, shape, quantization
/// scales, whether the file carries its own inference recipe, and the
/// consumer-defined schema identity.
///
/// # Safety
///
/// Artifact bytes must be readable for `artifact_len` bytes and `out_info`
/// writable for one complete [`Cb2VecArtifactInfoV1`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_artifact_probe_v1(
    artifact: *const u8,
    artifact_len: u32,
    out_info: *mut Cb2VecArtifactInfoV1,
) -> i32 {
    ffi_guard(|| {
        // SAFETY: Initialize output before parsing.
        unsafe { write_out(out_info, Cb2VecArtifactInfoV1::default(), "out_info")? };
        let bytes =
            // SAFETY: Caller keeps artifact storage live for this call.
            unsafe { raw_slice(artifact, artifact_len as usize, "artifact")? };
        let artifact = PackedCodebookArtifact::parse(bytes)
            .map_err(|error| FfiError::Artifact(error.to_string()))?;
        let (shape, embedding_scale, head_scale, factor_scale) =
            payload_scales(artifact.quantized())?;
        let metadata = artifact.metadata();
        let info = Cb2VecArtifactInfoV1 {
            artifact_version: u32::from(artifact.format_version()),
            kind: match artifact.kind() {
                crate::PackedCodebookKind::Flat => CB2VEC_MODEL_KIND_FLAT,
                crate::PackedCodebookKind::Factored => CB2VEC_MODEL_KIND_FACTORED,
            },
            token_count: shape_field(shape.token_count(), "token_count")?,
            group_count: shape_field(shape.group_count(), "group_count")?,
            dim: shape_field(shape.dim(), "dim")?,
            fm_rank: shape_field(shape.fm_rank(), "fm_rank")?,
            has_inference_config: u32::from(artifact.inference_config().is_some()),
            activation: artifact
                .inference_config()
                .map_or(CB2VEC_ACTIVATION_IDENTITY, |config| {
                    activation_to_ffi(config.activation)
                }),
            pooling: artifact
                .inference_config()
                .map_or(CB2VEC_POOLING_SUM, |config| pooling_to_ffi(config.pooling)),
            schema_version: metadata.schema_version,
            embedding_scale,
            head_scale,
            factor_scale,
            flags: u32::from(artifact.used_legacy_magic()) * CB2VEC_MODEL_FLAG_LEGACY_MAGIC,
            source_sha256: *artifact.source_sha256(),
            schema_digest: metadata.schema_digest,
            ..Cb2VecArtifactInfoV1::default()
        };
        // SAFETY: Output storage was validated above.
        unsafe { ptr::write(out_info, info) };
        Ok(())
    })
}

/// Returns immutable quantized-model metadata.
///
/// # Safety
///
/// `model` must be live and `out_info` writable for one info struct.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_model_get_info_v1(
    model: *const Cb2VecWeights,
    out_info: *mut Cb2VecModelInfoV1,
) -> i32 {
    ffi_guard(|| {
        // SAFETY: Initialize output before handle validation.
        unsafe { write_out(out_info, Cb2VecModelInfoV1::default(), "out_info")? };
        // SAFETY: Caller guarantees a live immutable model.
        let model = unsafe { weights_ref(model)? };
        let info = quantized_info(model)?;
        // SAFETY: Output storage was validated above.
        unsafe { ptr::write(out_info, info) };
        Ok(())
    })
}

/// Returns the schema identity the model's artifact carried.
///
/// A model loaded from a version-1 artifact, or from a version-2 artifact that
/// recorded no schema, reports version zero and an all-zero digest.
///
/// # Safety
///
/// `model` must be live and `out_metadata` writable for one complete
/// [`Cb2VecArtifactMetadataV1`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_model_get_metadata_v1(
    model: *const Cb2VecWeights,
    out_metadata: *mut Cb2VecArtifactMetadataV1,
) -> i32 {
    ffi_guard(|| {
        // SAFETY: Initialize output before handle validation.
        unsafe {
            write_out(
                out_metadata,
                Cb2VecArtifactMetadataV1::default(),
                "out_metadata",
            )?
        };
        // SAFETY: Caller guarantees a live immutable model.
        let model = unsafe { weights_ref(model)? };
        let metadata = Cb2VecArtifactMetadataV1 {
            schema_version: model.metadata.schema_version,
            schema_digest: model.metadata.schema_digest,
            ..Cb2VecArtifactMetadataV1::default()
        };
        // SAFETY: Output storage was validated above.
        unsafe { ptr::write(out_metadata, metadata) };
        Ok(())
    })
}

/// Predicts one raw quantized-model score.
///
/// # Safety
///
/// The model must be live. Every non-empty input buffer must remain readable
/// for the call. `out_score` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_model_predict_v1(
    model: *const Cb2VecWeights,
    tokens: *const u16,
    tokens_len: u32,
    site_offsets: *const u32,
    site_groups: *const u32,
    site_count: u32,
    out_score: *mut f32,
) -> i32 {
    ffi_guard(|| {
        // SAFETY: Initialize output before fallible work.
        unsafe { write_out(out_score, 0.0, "out_score")? };
        // SAFETY: Input buffers are copied and validated during this call.
        let input = unsafe {
            grouped_input_from_raw(tokens, tokens_len, site_offsets, site_groups, site_count)?
        };
        // SAFETY: Caller guarantees a live immutable model.
        let model = unsafe { weights_ref(model)? };
        let score = predict_payload(&input, model)?;
        // SAFETY: Output storage was validated above.
        unsafe { ptr::write(out_score, score) };
        Ok(())
    })
}

/// Predicts a flattened batch into caller-owned score storage.
///
/// Targets and weights in the batch descriptor are ignored and may be null.
/// Scores are written only after every input has predicted successfully.
///
/// # Safety
///
/// The model must be live. Batch input buffers must be readable.
/// `out_scores` must be writable for `out_scores_len` floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_model_predict_batch_v1(
    model: *const Cb2VecWeights,
    batch: *const Cb2VecTrainingBatchV1,
    out_scores: *mut f32,
    out_scores_len: u32,
) -> i32 {
    ffi_guard(|| {
        // SAFETY: Flattened buffers are copied before prediction.
        let inputs = unsafe { inputs_from_batch(batch)? };
        if out_scores_len as usize != inputs.len() {
            return Err(FfiError::Invalid(format!(
                "out_scores_len is {}, expected {}",
                out_scores_len,
                inputs.len()
            )));
        }
        // SAFETY: Caller guarantees a live immutable model.
        let model = unsafe { weights_ref(model)? };
        let scores = inputs
            .iter()
            .map(|input| predict_payload(input, model))
            .collect::<Result<Vec<_>, _>>()?;
        let output =
            // SAFETY: Caller supplies exactly one writable float per sample.
            unsafe { raw_mut_slice(out_scores, scores.len(), "out_scores")? };
        output.copy_from_slice(&scores);
        Ok(())
    })
}

/// Frees an immutable model handle. A null handle is a successful no-op.
///
/// # Safety
///
/// A non-null handle must have been returned by a model constructor, must not
/// have been freed, and must not be in use by another call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_model_free_v1(model: *mut Cb2VecWeights) -> i32 {
    ffi_guard(|| {
        if !model.is_null() {
            validate_mut_pointer(model, "model")?;
            // SAFETY: The caller transfers the unique live Box allocation back
            // exactly once, as required by this function's contract.
            drop(unsafe { Box::from_raw(model) });
        }
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// Incremental search sessions (ABI 1.1)
// ---------------------------------------------------------------------------

/// Writes the default session configuration.
///
/// # Safety
///
/// `out_config` must be aligned, writable storage for one complete
/// [`Cb2VecSessionConfigV1`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_session_config_default_v1(
    out_config: *mut Cb2VecSessionConfigV1,
) -> i32 {
    ffi_guard(|| {
        // SAFETY: Required by this exported function's caller contract.
        unsafe { write_out(out_config, Cb2VecSessionConfigV1::default(), "out_config") }
    })
}

/// Creates an incremental search session over an immutable model.
///
/// Every buffer the search loop needs is allocated here, sized by `config`.
/// The session takes shared ownership of the model's weights: freeing the
/// model handle first is safe, and any number of sessions may share one model.
/// A session itself is single-owner and must not be used from two threads.
///
/// The session is not scorable until `cb2vec_session_reset_v1` installs a
/// position.
///
/// # Safety
///
/// `model` must be a live handle and `config` must point to one complete
/// [`Cb2VecSessionConfigV1`]. `out_session` must be writable pointer storage,
/// and the returned handle must be freed exactly once with
/// `cb2vec_session_free_v1`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_session_create_v1(
    model: *const Cb2VecWeights,
    config: *const Cb2VecSessionConfigV1,
    out_session: *mut *mut Cb2VecSession,
) -> i32 {
    ffi_guard(|| {
        // SAFETY: Initialize output before fallible work.
        unsafe { write_out(out_session, ptr::null_mut(), "out_session")? };
        let limits = decode_session_config(
            // SAFETY: Caller supplies one complete config.
            unsafe { read_copy(config, "config")? },
        )?;
        // SAFETY: Caller guarantees a live immutable model.
        let model = unsafe { weights_ref(model)? };
        let session = IncrementalSession::new(Arc::clone(&model.payload), model.inference, limits)
            .map_err(map_session_error)?;
        let handle = Box::into_raw(Box::new(Cb2VecSession { session }));
        // SAFETY: Output storage was validated and initialized above.
        unsafe { ptr::write(out_session, handle) };
        Ok(())
    })
}

/// Installs a complete position and discards every pushed frame.
///
/// The layout arguments are the same ragged token view every other CB2Vec
/// entry point takes: `site_offsets` has `site_count + 1` monotonic entries
/// starting at zero and ending at `tokens_len`, and `site_groups` holds one
/// group index per site. Nothing changes unless every check passes.
///
/// This call allocates nothing; it only writes into buffers sized at creation.
///
/// # Safety
///
/// `session` must be a live handle owned exclusively by this thread. Every
/// non-empty input buffer must remain readable for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_session_reset_v1(
    session: *mut Cb2VecSession,
    tokens: *const u16,
    tokens_len: u32,
    site_offsets: *const u32,
    site_groups: *const u32,
    site_count: u32,
) -> i32 {
    ffi_guard(|| {
        let tokens =
            // SAFETY: Forwarded caller buffer; raw_slice performs structural checks.
            unsafe { raw_slice(tokens, tokens_len as usize, "tokens")? };
        let offset_count = (site_count as usize)
            .checked_add(1)
            .ok_or_else(|| FfiError::Invalid("site offset count overflow".to_string()))?;
        let site_offsets =
            // SAFETY: The prefix table has exactly site_count + 1 entries.
            unsafe { raw_slice(site_offsets, offset_count, "site_offsets")? };
        let site_groups =
            // SAFETY: There is exactly one group index per site.
            unsafe { raw_slice(site_groups, site_count as usize, "site_groups")? };
        // SAFETY: Caller guarantees exclusive access to a live session.
        let session = unsafe { session_mut(session)? };
        session
            .session
            .reset(tokens, site_offsets, site_groups)
            .map_err(map_session_error)
    })
}

/// Pushes one search move's token replacements as a single reversible frame.
///
/// Each delta names a site, the lane within that site, the token it expects to
/// find there, and its replacement. Every delta is validated before any state
/// changes, so a rejected frame leaves the session exactly as it was and does
/// not consume depth. A zero-length frame is legal and still pushes a frame,
/// which keeps push/pop balanced for a null move.
///
/// Only the logical token state is updated here; numeric work is deferred to
/// `cb2vec_session_materialize_v1` or `cb2vec_session_predict_v1`, so a branch
/// that is pushed and popped without scoring costs nothing numerically.
///
/// # Safety
///
/// `session` must be a live handle owned exclusively by this thread. When
/// `delta_count` is non-zero, `deltas` must be readable and 4-byte aligned for
/// `delta_count` [`Cb2VecTokenDeltaV1`] values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_session_push_v1(
    session: *mut Cb2VecSession,
    deltas: *const Cb2VecTokenDeltaV1,
    delta_count: u32,
) -> i32 {
    ffi_guard(|| {
        // `Cb2VecTokenDeltaV1` and `SessionDelta` are asserted at compile time
        // to have identical size, alignment, and field offsets, so the caller's
        // array is borrowed directly. That is what keeps this call allocation
        // free.
        let deltas =
            // SAFETY: Reading the layout-identical Rust type from a validated,
            // caller-owned array of the same length; raw_slice checks null,
            // alignment, and representable byte length.
            unsafe { raw_slice(deltas.cast::<SessionDelta>(), delta_count as usize, "deltas")? };
        // SAFETY: Caller guarantees exclusive access to a live session.
        let session = unsafe { session_mut(session)? };
        session.session.push(deltas).map_err(map_session_error)?;
        Ok(())
    })
}

/// Applies every pushed-but-unapplied frame to the numeric accumulators.
///
/// `cb2vec_session_predict_v1` does this implicitly; call this directly only
/// to move the cost to a chosen point.
///
/// # Safety
///
/// `session` must be a live handle owned exclusively by this thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_session_materialize_v1(session: *mut Cb2VecSession) -> i32 {
    ffi_guard(|| {
        // SAFETY: Caller guarantees exclusive access to a live session.
        let session = unsafe { session_mut(session)? };
        session.session.materialize_pending();
        Ok(())
    })
}

/// Materializes pending frames and scores the current position.
///
/// The result is bit-identical to `cb2vec_model_predict_v1` on the same tokens
/// with the same inference recipe.
///
/// # Safety
///
/// `session` must be a live handle owned exclusively by this thread, and
/// `out_score` must be writable for one `float`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_session_predict_v1(
    session: *mut Cb2VecSession,
    out_score: *mut f32,
) -> i32 {
    ffi_guard(|| {
        // SAFETY: Caller supplies one writable output scalar.
        unsafe { write_out(out_score, 0.0, "out_score")? };
        // SAFETY: Caller guarantees exclusive access to a live session.
        let session = unsafe { session_mut(session)? };
        let score = session.session.predict().map_err(map_session_error)?;
        // SAFETY: Output storage was validated above.
        unsafe { ptr::write(out_score, score) };
        Ok(())
    })
}

/// Undoes the most recent frame.
///
/// `out_popped_deltas` may be null. Popping with no frames left returns
/// [`CB2VEC_ERROR_STATE`].
///
/// # Safety
///
/// `session` must be a live handle owned exclusively by this thread. A
/// non-null `out_popped_deltas` must be writable for one `uint32_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_session_pop_v1(
    session: *mut Cb2VecSession,
    out_popped_deltas: *mut u32,
) -> i32 {
    ffi_guard(|| {
        if !out_popped_deltas.is_null() {
            // SAFETY: A non-null output is initialized before fallible work.
            unsafe { write_out(out_popped_deltas, 0, "out_popped_deltas")? };
        }
        // SAFETY: Caller guarantees exclusive access to a live session.
        let session = unsafe { session_mut(session)? };
        let popped = session.session.pop().map_err(map_session_error)?;
        if !out_popped_deltas.is_null() {
            let popped = u32::try_from(popped)
                .map_err(|_| FfiError::Invalid("frame size does not fit u32".to_string()))?;
            // SAFETY: Output storage was validated above.
            unsafe { ptr::write(out_popped_deltas, popped) };
        }
        Ok(())
    })
}

/// Returns session shape, stack depth, and the capacities it was built with.
///
/// # Safety
///
/// `session` must be a live handle and `out_info` writable for one complete
/// [`Cb2VecSessionInfoV1`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_session_get_info_v1(
    session: *const Cb2VecSession,
    out_info: *mut Cb2VecSessionInfoV1,
) -> i32 {
    ffi_guard(|| {
        // SAFETY: Initialize output before handle validation.
        unsafe { write_out(out_info, Cb2VecSessionInfoV1::default(), "out_info")? };
        // SAFETY: Caller guarantees a live session handle.
        let session = unsafe { session_ref(session)? };
        let info = session.session.info();
        let inference = session.session.inference_config();
        let field = |value: usize, name: &'static str| {
            u32::try_from(value).map_err(|_| FfiError::Model(format!("{name} does not fit u32")))
        };
        let info = Cb2VecSessionInfoV1 {
            site_count: field(info.site_count, "site_count")?,
            token_slots: field(info.token_slots, "token_slots")?,
            group_count: field(info.group_count, "group_count")?,
            depth: field(info.depth, "depth")?,
            materialized_depth: field(info.materialized_depth, "materialized_depth")?,
            pending_deltas: field(info.pending_deltas, "pending_deltas")?,
            max_sites: field(info.limits.max_sites, "max_sites")?,
            max_token_slots: field(info.limits.max_token_slots, "max_token_slots")?,
            max_deltas_per_frame: field(info.limits.max_deltas_per_frame, "max_deltas_per_frame")?,
            max_depth: field(info.limits.max_depth, "max_depth")?,
            activation: activation_to_ffi(inference.activation),
            pooling: pooling_to_ffi(inference.pooling),
            ..Cb2VecSessionInfoV1::default()
        };
        // SAFETY: Output storage was validated above.
        unsafe { ptr::write(out_info, info) };
        Ok(())
    })
}

/// Frees a session handle. A null handle is a successful no-op.
///
/// Freeing a session releases its share of the model's weights. The model
/// handle and the session may be freed in either order.
///
/// # Safety
///
/// A non-null handle must have been returned by `cb2vec_session_create_v1`,
/// must not have been freed, and must not be in use by another call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_session_free_v1(session: *mut Cb2VecSession) -> i32 {
    ffi_guard(|| {
        if !session.is_null() {
            validate_mut_pointer(session, "session")?;
            // SAFETY: The caller transfers the unique live Box allocation back
            // exactly once, as required by this function's contract.
            drop(unsafe { Box::from_raw(session) });
        }
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// Trainer checkpoints and artifact v2 (ABI 1.1)
// ---------------------------------------------------------------------------

/// Writes a complete trainer checkpoint into caller-owned storage.
///
/// Unlike an inference artifact, a checkpoint restores Adam moments, the
/// optimizer step, the shuffle RNG, and completed epochs, so a resumed run is
/// bit-identical to an uninterrupted one.
///
/// A null `out_bytes` with zero capacity is a size probe: the function writes
/// the exact required byte count and returns [`CB2VEC_ERROR_BUFFER_TOO_SMALL`].
/// No output bytes are written on a short buffer.
///
/// # Safety
///
/// `trainer` must be a live handle. `out_required_or_written` must be
/// writable. A non-null output buffer must be writable for at least
/// `out_capacity` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_trainer_write_checkpoint_v1(
    trainer: *const Cb2VecTrainer,
    out_bytes: *mut u8,
    out_capacity: u32,
    out_required_or_written: *mut u32,
) -> i32 {
    ffi_guard(|| {
        // SAFETY: Initialize byte count before fallible work.
        unsafe { write_out(out_required_or_written, 0, "out_required_or_written")? };
        // SAFETY: Caller guarantees a live trainer.
        let trainer = unsafe { trainer_ref(trainer)? };
        let bytes = trainer
            .trainer
            .write_checkpoint()
            .map_err(map_checkpoint_error)?;
        let required = u32::try_from(bytes.len())
            .map_err(|_| FfiError::Checkpoint("checkpoint exceeds u32 byte length".to_string()))?;
        // SAFETY: Count output was validated above.
        unsafe { ptr::write(out_required_or_written, required) };

        if out_bytes.is_null() {
            if out_capacity == 0 {
                return Err(FfiError::BufferTooSmall {
                    required: bytes.len(),
                    capacity: 0,
                });
            }
            return Err(FfiError::Null("out_bytes"));
        }
        if (out_capacity as usize) < bytes.len() {
            return Err(FfiError::BufferTooSmall {
                required: bytes.len(),
                capacity: out_capacity as usize,
            });
        }
        let output =
            // SAFETY: Caller promised a writable buffer; only required bytes are used.
            unsafe { raw_mut_slice(out_bytes, bytes.len(), "out_bytes")? };
        output.copy_from_slice(&bytes);
        Ok(())
    })
}

/// Restores a trainer that continues exactly where the checkpoint stopped.
///
/// The trainer configuration comes from the checkpoint itself. Corrupt,
/// truncated, or incompatible files are rejected with
/// [`CB2VEC_ERROR_CHECKPOINT`] and no handle is produced.
///
/// # Safety
///
/// `checkpoint` must be readable for `checkpoint_len` bytes and `out_trainer`
/// must be writable pointer storage. The returned handle must be freed exactly
/// once with `cb2vec_trainer_free_v1`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_trainer_load_checkpoint_v1(
    checkpoint: *const u8,
    checkpoint_len: u32,
    out_trainer: *mut *mut Cb2VecTrainer,
) -> i32 {
    ffi_guard(|| {
        // SAFETY: Initialize caller output before fallible parsing.
        unsafe { write_out(out_trainer, ptr::null_mut(), "out_trainer")? };
        let bytes =
            // SAFETY: Caller keeps checkpoint storage alive for this call.
            unsafe { raw_slice(checkpoint, checkpoint_len as usize, "checkpoint")? };
        let trainer = Trainer::from_checkpoint(bytes).map_err(map_checkpoint_error)?;
        let handle = Box::into_raw(Box::new(Cb2VecTrainer { trainer }));
        // SAFETY: Output storage was validated and initialized above.
        unsafe { ptr::write(out_trainer, handle) };
        Ok(())
    })
}

/// Exact byte length a checkpoint for this trainer will occupy.
///
/// # Safety
///
/// `trainer` must be a live handle and `out_len` writable for one `uint32_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_trainer_checkpoint_len_v1(
    trainer: *const Cb2VecTrainer,
    out_len: *mut u32,
) -> i32 {
    ffi_guard(|| {
        // SAFETY: Initialize output before fallible work.
        unsafe { write_out(out_len, 0, "out_len")? };
        // SAFETY: Caller guarantees a live trainer.
        let trainer = unsafe { trainer_ref(trainer)? };
        let shape = trainer
            .trainer
            .weights()
            .validate()
            .map_err(|error| FfiError::Model(error.to_string()))?;
        let len = TrainerCheckpoint::byte_len(shape).map_err(map_checkpoint_error)?;
        let len = u32::try_from(len)
            .map_err(|_| FfiError::Checkpoint("checkpoint exceeds u32 byte length".to_string()))?;
        // SAFETY: Output storage was validated above.
        unsafe { ptr::write(out_len, len) };
        Ok(())
    })
}

/// Writes a version-2 artifact that carries its own inference recipe.
///
/// The activation and pooling stored in the file come from the trainer, so a
/// consumer loading it through `cb2vec_model_load_v2` cannot score with the
/// wrong recipe. `metadata` may be null, in which case the artifact records no
/// schema identity.
///
/// The buffer protocol matches `cb2vec_trainer_write_artifact_v1`: a null
/// `out_bytes` with zero capacity is a size probe.
///
/// # Safety
///
/// The trainer and quantization config must be readable. `source_sha256_32`
/// must point to 32 readable bytes. A non-null `metadata` must point to one
/// complete struct. `out_required_or_written` must be writable, and a non-null
/// output buffer must be writable for at least `out_capacity` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cb2vec_trainer_write_artifact_v2(
    trainer: *const Cb2VecTrainer,
    quantization: *const Cb2VecQuantizationConfigV1,
    source_sha256_32: *const u8,
    metadata: *const Cb2VecArtifactMetadataV1,
    out_bytes: *mut u8,
    out_capacity: u32,
    out_required_or_written: *mut u32,
) -> i32 {
    ffi_guard(|| {
        // SAFETY: Initialize byte count before fallible work.
        unsafe { write_out(out_required_or_written, 0, "out_required_or_written")? };
        let quantization = decode_quantization(
            // SAFETY: Caller supplies one complete config.
            unsafe { read_copy(quantization, "quantization")? },
        )?;
        let metadata = if metadata.is_null() {
            ArtifactMetadata::default()
        } else {
            decode_metadata(
                // SAFETY: A non-null pointer must reference one complete struct.
                unsafe { read_copy(metadata, "metadata")? },
            )?
        };
        let digest_bytes =
            // SAFETY: The caller contract requires exactly 32 readable bytes.
            unsafe { raw_slice(source_sha256_32, 32, "source_sha256_32")? };
        let mut digest = [0u8; 32];
        digest.copy_from_slice(digest_bytes);
        // SAFETY: Caller guarantees a live trainer.
        let trainer = unsafe { trainer_ref(trainer)? };
        let quantized = trainer
            .trainer
            .weights()
            .quantize_i16(
                quantization.embedding_scale,
                quantization.head_scale,
                quantization.factor_scale,
            )
            .map_err(|error| FfiError::Model(error.to_string()))?;
        let artifact = PackedCodebookArtifact::new_flat_v2(
            trainer.trainer.weights().clone(),
            quantized,
            digest,
            trainer.trainer.inference_config(),
            metadata,
        )
        .map_err(|error| FfiError::Artifact(error.to_string()))?;
        let bytes = artifact
            .to_bytes()
            .map_err(|error| FfiError::Artifact(error.to_string()))?;
        let required = u32::try_from(bytes.len())
            .map_err(|_| FfiError::Artifact("artifact exceeds u32 byte length".to_string()))?;
        // SAFETY: Count output was validated above.
        unsafe { ptr::write(out_required_or_written, required) };

        if out_bytes.is_null() {
            if out_capacity == 0 {
                return Err(FfiError::BufferTooSmall {
                    required: bytes.len(),
                    capacity: 0,
                });
            }
            return Err(FfiError::Null("out_bytes"));
        }
        if (out_capacity as usize) < bytes.len() {
            return Err(FfiError::BufferTooSmall {
                required: bytes.len(),
                capacity: out_capacity as usize,
            });
        }
        let output =
            // SAFETY: Caller promised a writable buffer; only required bytes are used.
            unsafe { raw_mut_slice(out_bytes, bytes.len(), "out_bytes")? };
        output.copy_from_slice(&bytes);
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;
    use std::mem::{offset_of, size_of};
    use std::ptr;

    use super::*;
    use crate::{
        CB2VEC_CHECKPOINT_HEADER_LEN, CodebookWeights, FactoredQuantizedCodebookWeights,
        QuantizedCodebookWeights,
    };

    struct BatchFixture {
        tokens: Vec<u16>,
        site_offsets: Vec<u32>,
        site_groups: Vec<u32>,
        sample_offsets: Vec<u32>,
        targets: Vec<f32>,
        weights: Vec<f32>,
    }

    impl BatchFixture {
        fn unequal_mean() -> Self {
            Self {
                tokens: vec![0, 0, 1, 2, 3, 1],
                site_offsets: vec![0, 2, 3, 3, 4, 6],
                site_groups: vec![0, 0, 1, 0, 1],
                sample_offsets: vec![0, 3, 5],
                targets: vec![0.2, 0.8],
                weights: vec![1.0, 2.0],
            }
        }

        fn view(&self) -> Cb2VecTrainingBatchV1 {
            Cb2VecTrainingBatchV1 {
                struct_size: size_of::<Cb2VecTrainingBatchV1>() as u32,
                flags: 0,
                tokens: self.tokens.as_ptr(),
                site_token_offsets: self.site_offsets.as_ptr(),
                site_groups: self.site_groups.as_ptr(),
                sample_site_offsets: self.sample_offsets.as_ptr(),
                targets: self.targets.as_ptr(),
                weights: self.weights.as_ptr(),
                tokens_len: self.tokens.len() as u32,
                site_count: self.site_groups.len() as u32,
                sample_count: self.targets.len() as u32,
                reserved: 0,
            }
        }

        fn rust_samples(&self) -> Vec<TrainingSample> {
            vec![
                TrainingSample::weighted(
                    GroupedTokens::new(vec![0, 0, 1], vec![0, 2, 3, 3], vec![0, 0, 1]).unwrap(),
                    0.2,
                    1.0,
                ),
                TrainingSample::weighted(
                    GroupedTokens::new(vec![2, 3, 1], vec![0, 1, 3], vec![0, 1]).unwrap(),
                    0.8,
                    2.0,
                ),
            ]
        }
    }

    fn shape_fixture() -> (Cb2VecModelShapeV1, ModelShape) {
        let ffi = Cb2VecModelShapeV1 {
            token_count: 4,
            group_count: 2,
            dim: 3,
            fm_rank: 2,
            ..Cb2VecModelShapeV1::default()
        };
        let rust = ModelShape::new(4, 2, 3, 2).unwrap();
        (ffi, rust)
    }

    fn config_fixture() -> (Cb2VecTrainerConfigV1, TrainerConfig) {
        let ffi = Cb2VecTrainerConfigV1 {
            activation: CB2VEC_ACTIVATION_RELU,
            pooling: CB2VEC_POOLING_MEAN,
            loss: CB2VEC_LOSS_BCE_WITH_LOGITS,
            batch_size: 2,
            shuffle: 0,
            seed: 0x1234_5678_9ABC_DEF0,
            learning_rate: 0.002,
            ..Cb2VecTrainerConfigV1::default()
        };
        let rust = decode_config(ffi).unwrap();
        (ffi, rust)
    }

    fn create_trainer(
        shape: &Cb2VecModelShapeV1,
        config: &Cb2VecTrainerConfigV1,
    ) -> *mut Cb2VecTrainer {
        let mut trainer = ptr::null_mut();
        // SAFETY: All pointers reference complete live test values.
        let status = unsafe { cb2vec_trainer_create_v1(shape, config, &mut trainer) };
        assert_eq!(status, CB2VEC_OK, "{}", last_error_string());
        assert!(!trainer.is_null());
        trainer
    }

    fn last_error_string() -> String {
        // SAFETY: cb2vec_last_error returns a valid thread-local NUL-terminated
        // string until the next status-returning FFI call.
        unsafe {
            CStr::from_ptr(cb2vec_last_error())
                .to_string_lossy()
                .into_owned()
        }
    }

    fn assert_metrics_equal(actual: Cb2VecTrainingMetricsV1, expected: TrainingMetrics) {
        assert_eq!(actual.mean_loss.to_bits(), expected.mean_loss.to_bits());
        assert_eq!(
            actual.total_weight.to_bits(),
            expected.total_weight.to_bits()
        );
        assert_eq!(actual.sample_count, expected.sample_count as u64);
        assert_eq!(actual.batch_count, expected.batch_count as u64);
        assert_eq!(actual.optimizer_step, expected.optimizer_step);
        assert_eq!(actual.completed_epochs, expected.completed_epochs);
    }

    #[test]
    fn abi_layouts_are_fixed() {
        assert_eq!(size_of::<Cb2VecModelShapeV1>(), 32);
        assert_eq!(size_of::<Cb2VecTrainerConfigV1>(), 64);
        assert_eq!(size_of::<Cb2VecQuantizationConfigV1>(), 32);
        assert_eq!(size_of::<Cb2VecInferenceConfigV1>(), 16);
        assert_eq!(size_of::<Cb2VecTrainingMetricsV1>(), 64);
        assert_eq!(size_of::<Cb2VecModelInfoV1>(), 64);
        assert_eq!(
            size_of::<Cb2VecTrainingBatchV1>(),
            if cfg!(target_pointer_width = "64") {
                72
            } else {
                48
            }
        );
        assert_eq!(offset_of!(Cb2VecTrainerConfigV1, seed), 32);
        assert_eq!(offset_of!(Cb2VecTrainerConfigV1, learning_rate), 40);
        assert_eq!(offset_of!(Cb2VecTrainingMetricsV1, total_weight), 16);
        assert_eq!(offset_of!(Cb2VecTrainingMetricsV1, sample_count), 24);
        assert_eq!(offset_of!(Cb2VecModelInfoV1, factor_scale), 52);
        assert_eq!(cb2vec_abi_version(), 0x0001_0001);
        // SAFETY: Version is a process-lifetime NUL-terminated static string.
        let version = unsafe { CStr::from_ptr(cb2vec_library_version()) };
        assert_eq!(version.to_bytes(), env!("CARGO_PKG_VERSION").as_bytes());

        // Structures added in ABI 1.1.
        assert_eq!(size_of::<Cb2VecSessionConfigV1>(), 32);
        assert_eq!(size_of::<Cb2VecTokenDeltaV1>(), 12);
        assert_eq!(size_of::<Cb2VecSessionInfoV1>(), 64);
        assert_eq!(size_of::<Cb2VecArtifactMetadataV1>(), 32);
        assert_eq!(size_of::<Cb2VecArtifactInfoV1>(), 112);
        assert_eq!(offset_of!(Cb2VecTokenDeltaV1, old_token), 8);
        assert_eq!(offset_of!(Cb2VecSessionInfoV1, max_sites), 32);
        assert_eq!(offset_of!(Cb2VecArtifactMetadataV1, schema_digest), 16);
        assert_eq!(offset_of!(Cb2VecArtifactInfoV1, source_sha256), 64);
        assert_eq!(offset_of!(Cb2VecArtifactInfoV1, schema_digest), 96);

        // The delta type crosses the ABI by reinterpretation, not conversion.
        assert_eq!(size_of::<Cb2VecTokenDeltaV1>(), size_of::<SessionDelta>());
        assert_eq!(align_of::<Cb2VecTokenDeltaV1>(), align_of::<SessionDelta>());
        assert_eq!(
            offset_of!(Cb2VecTokenDeltaV1, site),
            offset_of!(SessionDelta, site)
        );
        assert_eq!(
            offset_of!(Cb2VecTokenDeltaV1, new_token),
            offset_of!(SessionDelta, new)
        );
    }

    #[test]
    fn abi_1_0_structs_are_still_accepted() {
        // A consumer compiled against ABI 1.0 keeps working unchanged: it
        // stamps 0x00010000 into every versioned struct it fills in itself.
        let shape = Cb2VecModelShapeV1 {
            abi_version: CB2VEC_ABI_VERSION_1_0,
            token_count: 4,
            group_count: 2,
            dim: 3,
            fm_rank: 1,
            ..Cb2VecModelShapeV1::default()
        };
        let config = Cb2VecTrainerConfigV1 {
            abi_version: CB2VEC_ABI_VERSION_1_0,
            batch_size: 2,
            shuffle: 0,
            ..Cb2VecTrainerConfigV1::default()
        };
        let quantization = Cb2VecQuantizationConfigV1 {
            abi_version: CB2VEC_ABI_VERSION_1_0,
            ..Cb2VecQuantizationConfigV1::default()
        };
        assert!(decode_shape(shape).is_ok());
        assert!(decode_config(config).is_ok());
        assert!(decode_quantization(quantization).is_ok());

        let trainer = create_trainer(&shape, &config);
        let mut model = ptr::null_mut();
        // SAFETY: Live trainer/config and writable handle output.
        unsafe {
            assert_eq!(
                cb2vec_trainer_quantize_v1(trainer, &quantization, &mut model),
                CB2VEC_OK,
                "{}",
                last_error_string()
            );
            assert_eq!(cb2vec_model_free_v1(model), CB2VEC_OK);
            assert_eq!(cb2vec_trainer_free_v1(trainer), CB2VEC_OK);
        }

        // A future minor, and any other major, are rejected.
        for rejected in [0x0001_0002u32, 0x0002_0000, 0x0000_0001] {
            assert!(matches!(
                decode_shape(Cb2VecModelShapeV1 {
                    abi_version: rejected,
                    ..shape
                }),
                Err(FfiError::Abi(_)),
            ));
        }
    }

    #[test]
    fn defaults_create_info_and_free() {
        let mut shape = Cb2VecModelShapeV1 {
            struct_size: 0,
            abi_version: 0,
            token_count: 0,
            group_count: 0,
            dim: 0,
            fm_rank: 0,
            reserved: [9; 2],
        };
        let mut trainer_config = Cb2VecTrainerConfigV1 {
            struct_size: 0,
            abi_version: 0,
            activation: 0,
            pooling: 0,
            loss: 0,
            batch_size: 0,
            shuffle: 0,
            flags: 9,
            seed: 0,
            learning_rate: 0.0,
            beta1: 0.0,
            beta2: 0.0,
            epsilon: 0.0,
            reserved: [9; 2],
        };
        let mut quant = Cb2VecQuantizationConfigV1 {
            struct_size: 0,
            abi_version: 0,
            embedding_scale: 0,
            head_scale: 0,
            factor_scale: 0,
            flags: 9,
            reserved: [9; 2],
        };
        let mut inference = Cb2VecInferenceConfigV1 {
            struct_size: 0,
            activation: 0,
            pooling: 0,
            flags: 9,
        };
        // SAFETY: All outputs are complete writable stack values.
        unsafe {
            assert_eq!(cb2vec_model_shape_default_v1(&mut shape), CB2VEC_OK);
            assert_eq!(
                cb2vec_trainer_config_default_v1(&mut trainer_config),
                CB2VEC_OK
            );
            assert_eq!(cb2vec_quantization_config_default_v1(&mut quant), CB2VEC_OK);
            assert_eq!(
                cb2vec_inference_config_default_v1(&mut inference),
                CB2VEC_OK
            );
        }
        assert_eq!(shape, Cb2VecModelShapeV1::default());
        assert_eq!(trainer_config, Cb2VecTrainerConfigV1::default());
        assert_eq!(quant, Cb2VecQuantizationConfigV1::default());
        assert_eq!(inference, Cb2VecInferenceConfigV1::default());

        let trainer = create_trainer(&shape, &trainer_config);
        let mut info = Cb2VecModelInfoV1::default();
        // SAFETY: The handle is live and info is writable.
        unsafe {
            assert_eq!(cb2vec_trainer_get_info_v1(trainer, &mut info), CB2VEC_OK);
        }
        assert_eq!(info.kind, CB2VEC_MODEL_KIND_FP32);
        assert_eq!(info.token_count, shape.token_count);
        assert_eq!(info.dim, shape.dim);
        // SAFETY: The handle is uniquely owned and freed exactly once; null is
        // explicitly a successful no-op.
        unsafe {
            assert_eq!(cb2vec_trainer_free_v1(trainer), CB2VEC_OK);
            assert_eq!(cb2vec_trainer_free_v1(ptr::null_mut()), CB2VEC_OK);
            assert_eq!(cb2vec_model_free_v1(ptr::null_mut()), CB2VEC_OK);
        }
    }

    #[test]
    fn ffi_evaluate_and_train_match_rust_api_bitwise() {
        let (shape_ffi, shape) = shape_fixture();
        let (config_ffi, config) = config_fixture();
        let fixture = BatchFixture::unequal_mean();
        let batch = fixture.view();
        let samples = fixture.rust_samples();
        let mut rust_trainer = Trainer::from_shape(shape, config).unwrap();
        let trainer = create_trainer(&shape_ffi, &config_ffi);

        let expected_eval = rust_trainer.evaluate(&samples).unwrap();
        let mut actual_eval = Cb2VecTrainingMetricsV1::default();
        // SAFETY: Handle, descriptor, referenced buffers, and output are live.
        unsafe {
            assert_eq!(
                cb2vec_trainer_evaluate_v1(trainer, &batch, &mut actual_eval),
                CB2VEC_OK,
                "{}",
                last_error_string()
            );
        }
        assert_metrics_equal(actual_eval, expected_eval);

        let expected_train = rust_trainer.train_batch(&samples).unwrap();
        let mut actual_train = Cb2VecTrainingMetricsV1::default();
        // SAFETY: The trainer is uniquely accessed and all buffers are live.
        unsafe {
            assert_eq!(
                cb2vec_trainer_train_batch_v1(trainer, &batch, &mut actual_train),
                CB2VEC_OK,
                "{}",
                last_error_string()
            );
        }
        assert_metrics_equal(actual_train, expected_train);

        let input = &samples[0].input;
        let offsets: Vec<u32> = input
            .site_offsets()
            .iter()
            .map(|&value| value as u32)
            .collect();
        let groups: Vec<u32> = input
            .site_groups()
            .iter()
            .map(|&value| value as u32)
            .collect();
        let mut score = 0.0;
        // SAFETY: The handle and all direct input/output buffers are live.
        unsafe {
            assert_eq!(
                cb2vec_trainer_predict_logit_v1(
                    trainer,
                    input.tokens().as_ptr(),
                    input.tokens().len() as u32,
                    offsets.as_ptr(),
                    groups.as_ptr(),
                    groups.len() as u32,
                    &mut score,
                ),
                CB2VEC_OK
            );
        }
        assert_eq!(
            score.to_bits(),
            rust_trainer.predict_logit(input).unwrap().to_bits()
        );

        // SAFETY: The handle is uniquely owned and freed once.
        unsafe { assert_eq!(cb2vec_trainer_free_v1(trainer), CB2VEC_OK) };
    }

    #[test]
    fn quantize_write_load_and_resume_match_rust() {
        let (shape_ffi, shape) = shape_fixture();
        let (config_ffi, config) = config_fixture();
        let fixture = BatchFixture::unequal_mean();
        let batch = fixture.view();
        let samples = fixture.rust_samples();
        let mut rust_trainer = Trainer::from_shape(shape, config).unwrap();
        rust_trainer.train_batch(&samples).unwrap();
        let trainer = create_trainer(&shape_ffi, &config_ffi);
        let mut metrics = Cb2VecTrainingMetricsV1::default();
        // SAFETY: The handle is uniquely accessed and fixture buffers are live.
        unsafe {
            assert_eq!(
                cb2vec_trainer_train_batch_v1(trainer, &batch, &mut metrics),
                CB2VEC_OK
            );
        }

        let quant = Cb2VecQuantizationConfigV1::default();
        let mut model = ptr::null_mut();
        // SAFETY: Live trainer/config and writable handle output.
        unsafe {
            assert_eq!(
                cb2vec_trainer_quantize_v1(trainer, &quant, &mut model),
                CB2VEC_OK,
                "{}",
                last_error_string()
            );
        }
        let expected_quantized = rust_trainer.weights().quantize_i16_s32_s64();
        let input = &samples[0].input;
        let offsets: Vec<u32> = input
            .site_offsets()
            .iter()
            .map(|&value| value as u32)
            .collect();
        let groups: Vec<u32> = input
            .site_groups()
            .iter()
            .map(|&value| value as u32)
            .collect();
        let expected_score =
            predict_quantized(input, &expected_quantized, rust_trainer.inference_config()).unwrap();
        let mut actual_score = 0.0;
        // SAFETY: Live model and direct input/output buffers.
        unsafe {
            assert_eq!(
                cb2vec_model_predict_v1(
                    model,
                    input.tokens().as_ptr(),
                    input.tokens().len() as u32,
                    offsets.as_ptr(),
                    groups.as_ptr(),
                    groups.len() as u32,
                    &mut actual_score,
                ),
                CB2VEC_OK
            );
        }
        assert_eq!(actual_score.to_bits(), expected_score.to_bits());

        let digest = [0x5Au8; 32];
        let expected_artifact = PackedCodebookArtifact::new_flat(
            rust_trainer.weights().clone(),
            expected_quantized,
            digest,
        )
        .unwrap()
        .to_bytes()
        .unwrap();
        let mut required = 0;
        // SAFETY: Live trainer/config/digest and writable byte-count output.
        let probe = unsafe {
            cb2vec_trainer_write_artifact_v1(
                trainer,
                &quant,
                digest.as_ptr(),
                ptr::null_mut(),
                0,
                &mut required,
            )
        };
        assert_eq!(probe, CB2VEC_ERROR_BUFFER_TOO_SMALL);
        assert_eq!(required as usize, expected_artifact.len());
        let mut bytes = vec![0xA5; required as usize + 8];
        // SAFETY: Output has at least required writable bytes; all inputs live.
        unsafe {
            assert_eq!(
                cb2vec_trainer_write_artifact_v1(
                    trainer,
                    &quant,
                    digest.as_ptr(),
                    bytes.as_mut_ptr(),
                    required,
                    &mut required,
                ),
                CB2VEC_OK,
                "{}",
                last_error_string()
            );
        }
        assert_eq!(&bytes[..required as usize], expected_artifact);
        assert_eq!(&bytes[required as usize..], &[0xA5; 8]);

        let inference = Cb2VecInferenceConfigV1 {
            activation: CB2VEC_ACTIVATION_RELU,
            pooling: CB2VEC_POOLING_MEAN,
            ..Cb2VecInferenceConfigV1::default()
        };
        let mut loaded_model = ptr::null_mut();
        // SAFETY: Artifact/config inputs and handle output are live.
        unsafe {
            assert_eq!(
                cb2vec_model_load_v1(bytes.as_ptr(), required, &inference, &mut loaded_model,),
                CB2VEC_OK,
                "{}",
                last_error_string()
            );
        }
        let mut loaded_score = 0.0;
        // SAFETY: Live model and direct input/output buffers.
        unsafe {
            assert_eq!(
                cb2vec_model_predict_v1(
                    loaded_model,
                    input.tokens().as_ptr(),
                    input.tokens().len() as u32,
                    offsets.as_ptr(),
                    groups.as_ptr(),
                    groups.len() as u32,
                    &mut loaded_score,
                ),
                CB2VEC_OK
            );
        }
        assert_eq!(loaded_score.to_bits(), expected_score.to_bits());

        let mut resumed = ptr::null_mut();
        // SAFETY: Artifact/config inputs and trainer output are live.
        unsafe {
            assert_eq!(
                cb2vec_trainer_load_artifact_v1(
                    bytes.as_ptr(),
                    required,
                    &config_ffi,
                    &mut resumed,
                ),
                CB2VEC_OK
            );
        }
        let mut resumed_metrics = Cb2VecTrainingMetricsV1::default();
        // SAFETY: Live trainer, descriptor buffers, and output.
        unsafe {
            assert_eq!(
                cb2vec_trainer_evaluate_v1(resumed, &batch, &mut resumed_metrics),
                CB2VEC_OK
            );
        }
        let resumed_rust = Trainer::new(rust_trainer.weights().clone(), config).unwrap();
        assert_metrics_equal(resumed_metrics, resumed_rust.evaluate(&samples).unwrap());
        assert_eq!(resumed_metrics.optimizer_step, 0);
        assert_eq!(resumed_metrics.completed_epochs, 0);

        // SAFETY: Every live handle is uniquely freed exactly once.
        unsafe {
            assert_eq!(cb2vec_model_free_v1(model), CB2VEC_OK);
            assert_eq!(cb2vec_model_free_v1(loaded_model), CB2VEC_OK);
            assert_eq!(cb2vec_trainer_free_v1(resumed), CB2VEC_OK);
            assert_eq!(cb2vec_trainer_free_v1(trainer), CB2VEC_OK);
        }
    }

    #[test]
    fn invalid_batch_does_not_mutate_trainer() {
        let (shape_ffi, _) = shape_fixture();
        let (config_ffi, _) = config_fixture();
        let mut fixture = BatchFixture::unequal_mean();
        let trainer = create_trainer(&shape_ffi, &config_ffi);
        let input = fixture.rust_samples().remove(0).input;
        let offsets: Vec<u32> = input
            .site_offsets()
            .iter()
            .map(|&value| value as u32)
            .collect();
        let groups: Vec<u32> = input
            .site_groups()
            .iter()
            .map(|&value| value as u32)
            .collect();
        let mut before = 0.0;
        // SAFETY: Live handle and direct input/output buffers.
        unsafe {
            assert_eq!(
                cb2vec_trainer_predict_logit_v1(
                    trainer,
                    input.tokens().as_ptr(),
                    input.tokens().len() as u32,
                    offsets.as_ptr(),
                    groups.as_ptr(),
                    groups.len() as u32,
                    &mut before,
                ),
                CB2VEC_OK
            );
        }

        fixture.targets[1] = 2.0;
        let invalid = fixture.view();
        let mut metrics = Cb2VecTrainingMetricsV1 {
            mean_loss: 99.0,
            ..Cb2VecTrainingMetricsV1::default()
        };
        // SAFETY: Descriptor is structurally readable; invalid target is a
        // validated data error and the trainer remains exclusively accessed.
        let status = unsafe { cb2vec_trainer_train_batch_v1(trainer, &invalid, &mut metrics) };
        assert_eq!(status, CB2VEC_ERROR_INVALID_ARGUMENT);
        assert_eq!(metrics, Cb2VecTrainingMetricsV1::default());

        let mut after = 0.0;
        // SAFETY: Live handle and direct input/output buffers.
        unsafe {
            assert_eq!(
                cb2vec_trainer_predict_logit_v1(
                    trainer,
                    input.tokens().as_ptr(),
                    input.tokens().len() as u32,
                    offsets.as_ptr(),
                    groups.as_ptr(),
                    groups.len() as u32,
                    &mut after,
                ),
                CB2VEC_OK
            );
        }
        assert_eq!(after.to_bits(), before.to_bits());

        let invalid_tokens = [shape_ffi.token_count as u16];
        let invalid_token_offsets = [0, 1];
        let invalid_token_groups = [0];
        // SAFETY: Every pointer references complete live storage. The token is
        // intentionally outside the configured codebook and must be reported
        // as an invalid argument rather than a numerical failure.
        let status = unsafe {
            cb2vec_trainer_predict_logit_v1(
                trainer,
                invalid_tokens.as_ptr(),
                invalid_tokens.len() as u32,
                invalid_token_offsets.as_ptr(),
                invalid_token_groups.as_ptr(),
                invalid_token_groups.len() as u32,
                &mut after,
            )
        };
        assert_eq!(status, CB2VEC_ERROR_INVALID_ARGUMENT);

        fixture.targets[1] = 0.8;
        fixture.site_offsets[2] = 1;
        let invalid_offsets = fixture.view();
        // SAFETY: Descriptor points to live storage; malformed offsets are
        // rejected before any slice indexing or trainer mutation.
        let status =
            unsafe { cb2vec_trainer_train_batch_v1(trainer, &invalid_offsets, &mut metrics) };
        assert_eq!(status, CB2VEC_ERROR_INVALID_ARGUMENT);

        // SAFETY: Handle is uniquely freed once.
        unsafe { assert_eq!(cb2vec_trainer_free_v1(trainer), CB2VEC_OK) };
    }

    #[test]
    fn factored_artifact_is_flattened_once_and_scores_match() {
        let shape = ModelShape::new(4, 2, 3, 2).unwrap();
        let source = CodebookWeights::deterministic(4, 2, 3, 2);
        let flat = source.quantize_i16_s32_s64();
        let factored = FactoredQuantizedCodebookWeights::new(
            shape.dim(),
            shape.fm_rank(),
            flat.embedding_scale,
            flat.head_scale,
            flat.factor_scale,
            (0..shape.token_count() as u8).collect(),
            flat.embeddings.clone(),
            vec![0; flat.embeddings.len()],
            flat.head.clone(),
            flat.factors.clone(),
            flat.bias,
        )
        .unwrap();
        let bytes = PackedCodebookArtifact::new_factored(source, factored, [7; 32])
            .unwrap()
            .to_bytes()
            .unwrap();
        let inference = Cb2VecInferenceConfigV1 {
            activation: CB2VEC_ACTIVATION_RELU,
            pooling: CB2VEC_POOLING_MEAN,
            ..Cb2VecInferenceConfigV1::default()
        };
        let mut model = ptr::null_mut();
        // SAFETY: Artifact/config are readable and model output is writable.
        unsafe {
            assert_eq!(
                cb2vec_model_load_v1(bytes.as_ptr(), bytes.len() as u32, &inference, &mut model,),
                CB2VEC_OK
            );
        }
        let mut info = Cb2VecModelInfoV1::default();
        // SAFETY: Live model and writable info.
        unsafe {
            assert_eq!(cb2vec_model_get_info_v1(model, &mut info), CB2VEC_OK);
        }
        assert_eq!(info.kind, CB2VEC_MODEL_KIND_FACTORED);
        assert_ne!(info.flags & CB2VEC_MODEL_FLAG_FLATTENED_AT_LOAD, 0);

        let input = GroupedTokens::new(vec![0, 1, 2], vec![0, 2, 3], vec![0, 1]).unwrap();
        let offsets = [0u32, 2, 3];
        let groups = [0u32, 1];
        let expected = predict_quantized(
            &input,
            &flat,
            InferenceConfig::new(Activation::Relu, Pooling::Mean),
        )
        .unwrap();
        let mut actual = 0.0;
        // SAFETY: Live model and direct input/output buffers.
        unsafe {
            assert_eq!(
                cb2vec_model_predict_v1(
                    model,
                    input.tokens().as_ptr(),
                    input.tokens().len() as u32,
                    offsets.as_ptr(),
                    groups.as_ptr(),
                    groups.len() as u32,
                    &mut actual,
                ),
                CB2VEC_OK
            );
            assert_eq!(cb2vec_model_free_v1(model), CB2VEC_OK);
        }
        assert_eq!(actual.to_bits(), expected.to_bits());
    }

    #[test]
    fn pointer_errors_short_buffer_and_panic_are_contained() {
        let mut shape = Cb2VecModelShapeV1::default();
        // SAFETY: A null output is intentionally passed to validate fail-closed
        // pointer handling; no dereference occurs.
        let status = unsafe { cb2vec_model_shape_default_v1(ptr::null_mut()) };
        assert_eq!(status, CB2VEC_ERROR_NULL_POINTER);
        assert!(last_error_string().contains("out_shape"));

        let bytes = [0u8; 16];
        // SAFETY: Adding one stays within the allocation and intentionally
        // produces a misaligned u32 pointer; raw_slice rejects before reading.
        let misaligned = unsafe { bytes.as_ptr().add(1).cast::<u32>() };
        // SAFETY: This private helper validates alignment before constructing a slice.
        let error = unsafe { raw_slice(misaligned, 1, "misaligned") }.unwrap_err();
        assert!(matches!(error, FfiError::Invalid(_)));
        // SAFETY: Zero-length paths return an empty slice without constructing
        // a Rust slice from the null pointer.
        assert!(
            unsafe { raw_slice::<u32>(ptr::null(), 0, "empty") }
                .unwrap()
                .is_empty()
        );

        let (shape_ffi, _) = shape_fixture();
        let (config_ffi, _) = config_fixture();
        let trainer = create_trainer(&shape_ffi, &config_ffi);
        let quant = Cb2VecQuantizationConfigV1::default();
        let digest = [0u8; 32];
        let mut required = 0;
        // SAFETY: Live inputs and writable byte count; null/zero is the defined probe.
        unsafe {
            assert_eq!(
                cb2vec_trainer_write_artifact_v1(
                    trainer,
                    &quant,
                    digest.as_ptr(),
                    ptr::null_mut(),
                    0,
                    &mut required,
                ),
                CB2VEC_ERROR_BUFFER_TOO_SMALL
            );
        }
        let mut short = vec![0xCC; required.saturating_sub(1) as usize];
        let short_capacity = short.len() as u32;
        // SAFETY: Short buffer is live; the function detects capacity before writing.
        unsafe {
            assert_eq!(
                cb2vec_trainer_write_artifact_v1(
                    trainer,
                    &quant,
                    digest.as_ptr(),
                    short.as_mut_ptr(),
                    short_capacity,
                    &mut required,
                ),
                CB2VEC_ERROR_BUFFER_TOO_SMALL
            );
        }
        assert!(short.iter().all(|&byte| byte == 0xCC));

        let panic_status = ffi_guard(|| -> Result<(), FfiError> {
            panic!("fixture panic");
        });
        assert_eq!(panic_status, CB2VEC_ERROR_PANIC);
        assert!(last_error_string().contains("fixture panic"));
        // A version accessor does not clear the TLS error.
        assert_eq!(cb2vec_abi_version(), CB2VEC_ABI_VERSION);
        assert!(last_error_string().contains("fixture panic"));
        // A successful status-returning call clears it.
        // SAFETY: `shape` is a complete writable stack value.
        unsafe {
            assert_eq!(cb2vec_model_shape_default_v1(&mut shape), CB2VEC_OK);
        }
        assert!(last_error_string().is_empty());

        // SAFETY: Handle is uniquely freed once.
        unsafe { assert_eq!(cb2vec_trainer_free_v1(trainer), CB2VEC_OK) };
    }

    /// Builds a loaded model handle plus the ragged layout used by the
    /// session tests below.
    struct SessionFixture {
        model: *mut Cb2VecWeights,
        weights: QuantizedCodebookWeights,
        tokens: Vec<u16>,
        offsets: Vec<u32>,
        groups: Vec<u32>,
        inference: InferenceConfig,
    }

    impl SessionFixture {
        fn new() -> Self {
            let source = CodebookWeights::deterministic(9, 3, 4, 2);
            let weights = source.quantize_i16_s32_s64();
            let artifact = PackedCodebookArtifact::new_flat(source, weights.clone(), [0x33; 32])
                .unwrap()
                .to_bytes()
                .unwrap();
            let inference_ffi = Cb2VecInferenceConfigV1 {
                activation: CB2VEC_ACTIVATION_RELU,
                pooling: CB2VEC_POOLING_MEAN,
                ..Cb2VecInferenceConfigV1::default()
            };
            let mut model = ptr::null_mut();
            // SAFETY: Artifact/config are readable and handle output writable.
            unsafe {
                assert_eq!(
                    cb2vec_model_load_v1(
                        artifact.as_ptr(),
                        artifact.len() as u32,
                        &inference_ffi,
                        &mut model,
                    ),
                    CB2VEC_OK,
                    "{}",
                    last_error_string()
                );
            }
            Self {
                model,
                weights,
                tokens: vec![0, 3, 5, 1, 8, 2, 7, 4, 6],
                offsets: vec![0, 3, 5, 7, 8, 9],
                groups: vec![0, 1, 1, 2, 0],
                inference: InferenceConfig::new(Activation::Relu, Pooling::Mean),
            }
        }

        fn create_session(&self, config: &Cb2VecSessionConfigV1) -> *mut Cb2VecSession {
            let mut session = ptr::null_mut();
            // SAFETY: Live model, complete config, writable handle output.
            let status = unsafe { cb2vec_session_create_v1(self.model, config, &mut session) };
            assert_eq!(status, CB2VEC_OK, "{}", last_error_string());
            assert!(!session.is_null());
            session
        }

        fn reset(&self, session: *mut Cb2VecSession) {
            // SAFETY: Live session and complete live layout buffers.
            let status = unsafe {
                cb2vec_session_reset_v1(
                    session,
                    self.tokens.as_ptr(),
                    self.tokens.len() as u32,
                    self.offsets.as_ptr(),
                    self.groups.as_ptr(),
                    self.groups.len() as u32,
                )
            };
            assert_eq!(status, CB2VEC_OK, "{}", last_error_string());
        }

        fn predict(&self, session: *mut Cb2VecSession) -> f32 {
            let mut score = 0.0;
            // SAFETY: Live session and writable output scalar.
            let status = unsafe { cb2vec_session_predict_v1(session, &mut score) };
            assert_eq!(status, CB2VEC_OK, "{}", last_error_string());
            score
        }

        /// Full-rebuild reference score for an arbitrary token vector.
        fn rebuild(&self, tokens: &[u16]) -> f32 {
            let input = GroupedTokens::new(
                tokens.to_vec(),
                self.offsets.iter().map(|&value| value as usize).collect(),
                self.groups.iter().map(|&value| value as usize).collect(),
            )
            .unwrap();
            predict_quantized(&input, &self.weights, self.inference).unwrap()
        }

        /// The same score through the non-incremental C entry point.
        fn rebuild_through_ffi(&self, tokens: &[u16]) -> f32 {
            let mut score = 0.0;
            // SAFETY: Live model and complete live input/output buffers.
            let status = unsafe {
                cb2vec_model_predict_v1(
                    self.model,
                    tokens.as_ptr(),
                    tokens.len() as u32,
                    self.offsets.as_ptr(),
                    self.groups.as_ptr(),
                    self.groups.len() as u32,
                    &mut score,
                )
            };
            assert_eq!(status, CB2VEC_OK, "{}", last_error_string());
            score
        }
    }

    impl Drop for SessionFixture {
        fn drop(&mut self) {
            // SAFETY: The handle is uniquely owned and freed exactly once.
            unsafe { assert_eq!(cb2vec_model_free_v1(self.model), CB2VEC_OK) };
        }
    }

    fn session_config() -> Cb2VecSessionConfigV1 {
        Cb2VecSessionConfigV1 {
            max_sites: 8,
            max_token_slots: 16,
            max_deltas_per_frame: 4,
            max_depth: 6,
            ..Cb2VecSessionConfigV1::default()
        }
    }

    #[test]
    fn session_defaults_create_reset_and_report_info() {
        let mut config = Cb2VecSessionConfigV1 {
            struct_size: 0,
            abi_version: 0,
            max_sites: 0,
            max_token_slots: 0,
            max_deltas_per_frame: 0,
            max_depth: 0,
            flags: 9,
            reserved: [9; 1],
        };
        // SAFETY: Output is a complete writable stack value.
        unsafe { assert_eq!(cb2vec_session_config_default_v1(&mut config), CB2VEC_OK) };
        assert_eq!(config, Cb2VecSessionConfigV1::default());

        let fixture = SessionFixture::new();
        let config = session_config();
        let session = fixture.create_session(&config);

        let mut info = Cb2VecSessionInfoV1::default();
        // SAFETY: Live session and writable info.
        unsafe { assert_eq!(cb2vec_session_get_info_v1(session, &mut info), CB2VEC_OK) };
        assert_eq!(info.site_count, 0);
        assert_eq!(info.max_depth, config.max_depth);
        assert_eq!(info.activation, CB2VEC_ACTIVATION_RELU);
        assert_eq!(info.pooling, CB2VEC_POOLING_MEAN);

        // Scoring before a reset is a state error, not a crash.
        let mut score = 1.0;
        // SAFETY: Live session and writable output.
        let status = unsafe { cb2vec_session_predict_v1(session, &mut score) };
        assert_eq!(status, CB2VEC_ERROR_STATE);
        assert_eq!(score, 0.0);

        fixture.reset(session);
        // SAFETY: Live session and writable info.
        unsafe { assert_eq!(cb2vec_session_get_info_v1(session, &mut info), CB2VEC_OK) };
        assert_eq!(info.site_count, 5);
        assert_eq!(info.token_slots, 9);
        assert_eq!(info.group_count, 3);
        assert_eq!(info.depth, 0);

        // SAFETY: The handle is uniquely owned and freed exactly once; null is
        // explicitly a successful no-op.
        unsafe {
            assert_eq!(cb2vec_session_free_v1(session), CB2VEC_OK);
            assert_eq!(cb2vec_session_free_v1(ptr::null_mut()), CB2VEC_OK);
        }
    }

    #[test]
    fn session_scores_match_full_rebuild_through_the_abi() {
        let fixture = SessionFixture::new();
        let session = fixture.create_session(&session_config());
        fixture.reset(session);

        assert_eq!(
            fixture.predict(session).to_bits(),
            fixture.rebuild(&fixture.tokens).to_bits()
        );
        assert_eq!(
            fixture.predict(session).to_bits(),
            fixture.rebuild_through_ffi(&fixture.tokens).to_bits()
        );

        let frame = [
            Cb2VecTokenDeltaV1 {
                site: 0,
                lane: 0,
                old_token: 0,
                new_token: 6,
            },
            Cb2VecTokenDeltaV1 {
                site: 2,
                lane: 1,
                old_token: 7,
                new_token: 0,
            },
        ];
        // SAFETY: Live session and a complete live delta array.
        unsafe {
            assert_eq!(
                cb2vec_session_push_v1(session, frame.as_ptr(), frame.len() as u32),
                CB2VEC_OK,
                "{}",
                last_error_string()
            );
        }
        let expected = [6, 3, 5, 1, 8, 2, 0, 4, 6];
        assert_eq!(
            fixture.predict(session).to_bits(),
            fixture.rebuild_through_ffi(&expected).to_bits()
        );

        // Explicit materialization is idempotent and does not shift the score.
        // SAFETY: Live session handle.
        unsafe {
            assert_eq!(cb2vec_session_materialize_v1(session), CB2VEC_OK);
            assert_eq!(cb2vec_session_materialize_v1(session), CB2VEC_OK);
        }
        assert_eq!(
            fixture.predict(session).to_bits(),
            fixture.rebuild_through_ffi(&expected).to_bits()
        );

        let mut popped = 0;
        // SAFETY: Live session and writable count output.
        unsafe {
            assert_eq!(cb2vec_session_pop_v1(session, &mut popped), CB2VEC_OK);
        }
        assert_eq!(popped, 2);
        assert_eq!(
            fixture.predict(session).to_bits(),
            fixture.rebuild_through_ffi(&fixture.tokens).to_bits()
        );

        // Popping an empty stack reports state, and a null count output is fine.
        // SAFETY: Live session; a null count output is an explicit opt-out.
        unsafe {
            assert_eq!(
                cb2vec_session_pop_v1(session, ptr::null_mut()),
                CB2VEC_ERROR_STATE
            );
            assert_eq!(cb2vec_session_free_v1(session), CB2VEC_OK);
        }
    }

    #[test]
    fn session_rejects_bad_deltas_and_limits_without_corrupting_state() {
        let fixture = SessionFixture::new();
        let config = session_config();
        let session = fixture.create_session(&config);
        fixture.reset(session);
        let before = fixture.predict(session).to_bits();

        let push = |deltas: &[Cb2VecTokenDeltaV1]| {
            // SAFETY: Live session and a complete live delta array.
            unsafe { cb2vec_session_push_v1(session, deltas.as_ptr(), deltas.len() as u32) }
        };
        let delta = |site, lane, old_token, new_token| Cb2VecTokenDeltaV1 {
            site,
            lane,
            old_token,
            new_token,
        };

        // Wrong expected old token.
        assert_eq!(push(&[delta(0, 0, 4, 1)]), CB2VEC_ERROR_INVALID_ARGUMENT);
        // Site and lane out of range.
        assert_eq!(push(&[delta(99, 0, 0, 1)]), CB2VEC_ERROR_INVALID_ARGUMENT);
        assert_eq!(push(&[delta(3, 5, 4, 1)]), CB2VEC_ERROR_INVALID_ARGUMENT);
        // Replacement token outside the codebook.
        assert_eq!(push(&[delta(0, 0, 0, 500)]), CB2VEC_ERROR_INVALID_ARGUMENT);
        // The same slot twice in one frame.
        assert_eq!(
            push(&[delta(0, 0, 0, 1), delta(0, 0, 0, 2)]),
            CB2VEC_ERROR_INVALID_ARGUMENT
        );
        // A valid delta followed by an invalid one must not partially apply.
        assert_eq!(
            push(&[delta(1, 0, 1, 2), delta(1, 7, 0, 2)]),
            CB2VEC_ERROR_INVALID_ARGUMENT
        );

        let mut info = Cb2VecSessionInfoV1::default();
        // SAFETY: Live session and writable info.
        unsafe { assert_eq!(cb2vec_session_get_info_v1(session, &mut info), CB2VEC_OK) };
        assert_eq!(info.depth, 0);
        assert_eq!(info.pending_deltas, 0);
        assert_eq!(fixture.predict(session).to_bits(), before);

        // Too many deltas in one frame.
        let oversized: Vec<Cb2VecTokenDeltaV1> = (0..config.max_deltas_per_frame + 1)
            .map(|lane| delta(0, lane % 3, 0, 1))
            .collect();
        // SAFETY: Live session and a complete live delta array.
        let status =
            unsafe { cb2vec_session_push_v1(session, oversized.as_ptr(), oversized.len() as u32) };
        assert_eq!(status, CB2VEC_ERROR_LIMIT_EXCEEDED);

        // Exhaust depth, then confirm the ceiling reports distinctly.
        for step in 0..config.max_depth {
            let old = if step == 0 { 0 } else { (step - 1) as u16 % 9 };
            assert_eq!(push(&[delta(0, 0, old, (step % 9) as u16)]), CB2VEC_OK);
        }
        let last = (config.max_depth - 1) as u16 % 9;
        assert_eq!(push(&[delta(0, 0, last, 1)]), CB2VEC_ERROR_LIMIT_EXCEEDED);

        // Everything unwinds cleanly back to the reset position.
        for _ in 0..config.max_depth {
            // SAFETY: Live session; the count output is not needed here.
            unsafe { assert_eq!(cb2vec_session_pop_v1(session, ptr::null_mut()), CB2VEC_OK) };
        }
        assert_eq!(fixture.predict(session).to_bits(), before);

        // SAFETY: The handle is uniquely owned and freed exactly once.
        unsafe { assert_eq!(cb2vec_session_free_v1(session), CB2VEC_OK) };
    }

    #[test]
    fn sessions_share_a_model_and_outlive_its_handle() {
        let source = CodebookWeights::deterministic(9, 3, 4, 2);
        let weights = source.quantize_i16_s32_s64();
        let artifact = PackedCodebookArtifact::new_flat(source, weights.clone(), [1; 32])
            .unwrap()
            .to_bytes()
            .unwrap();
        let inference_ffi = Cb2VecInferenceConfigV1 {
            activation: CB2VEC_ACTIVATION_RELU,
            pooling: CB2VEC_POOLING_MEAN,
            ..Cb2VecInferenceConfigV1::default()
        };
        let mut model = ptr::null_mut();
        // SAFETY: Artifact/config are readable and handle output writable.
        unsafe {
            assert_eq!(
                cb2vec_model_load_v1(
                    artifact.as_ptr(),
                    artifact.len() as u32,
                    &inference_ffi,
                    &mut model,
                ),
                CB2VEC_OK
            );
        }

        let tokens = [0u16, 3, 5, 1, 8, 2, 7, 4, 6];
        let offsets = [0u32, 3, 5, 7, 8, 9];
        let groups = [0u32, 1, 1, 2, 0];
        let config = session_config();
        let sessions: Vec<*mut Cb2VecSession> = (0..4)
            .map(|index| {
                let mut session = ptr::null_mut();
                // SAFETY: Live model, complete config, writable output.
                unsafe {
                    assert_eq!(
                        cb2vec_session_create_v1(model, &config, &mut session),
                        CB2VEC_OK
                    );
                    assert_eq!(
                        cb2vec_session_reset_v1(
                            session,
                            tokens.as_ptr(),
                            tokens.len() as u32,
                            offsets.as_ptr(),
                            groups.as_ptr(),
                            groups.len() as u32,
                        ),
                        CB2VEC_OK
                    );
                    let delta = Cb2VecTokenDeltaV1 {
                        site: 0,
                        lane: 0,
                        old_token: 0,
                        new_token: index as u16 + 1,
                    };
                    assert_eq!(cb2vec_session_push_v1(session, &delta, 1), CB2VEC_OK);
                }
                session
            })
            .collect();

        // Free the model first: sessions hold a share of the weights, so this
        // is defined behavior rather than a dangling reference.
        // SAFETY: The model handle is uniquely owned and freed exactly once.
        unsafe { assert_eq!(cb2vec_model_free_v1(model), CB2VEC_OK) };

        let inference = InferenceConfig::new(Activation::Relu, Pooling::Mean);
        for (index, &session) in sessions.iter().enumerate() {
            let mut expected_tokens = tokens;
            expected_tokens[0] = index as u16 + 1;
            let expected = predict_quantized(
                &GroupedTokens::new(
                    expected_tokens.to_vec(),
                    offsets.iter().map(|&value| value as usize).collect(),
                    groups.iter().map(|&value| value as usize).collect(),
                )
                .unwrap(),
                &weights,
                inference,
            )
            .unwrap();
            let mut score = 0.0;
            // SAFETY: Every session handle is still live and exclusively used.
            unsafe {
                assert_eq!(cb2vec_session_predict_v1(session, &mut score), CB2VEC_OK);
            }
            assert_eq!(
                score.to_bits(),
                expected.to_bits(),
                "session {index} observed another session's state"
            );
        }
        for &session in &sessions {
            // SAFETY: Each handle is uniquely owned and freed exactly once.
            unsafe { assert_eq!(cb2vec_session_free_v1(session), CB2VEC_OK) };
        }
    }

    #[test]
    fn the_c_session_loop_does_not_allocate() {
        let fixture = SessionFixture::new();
        let session = fixture.create_session(&session_config());
        fixture.reset(session);
        let frame = [
            Cb2VecTokenDeltaV1 {
                site: 0,
                lane: 0,
                old_token: 0,
                new_token: 4,
            },
            Cb2VecTokenDeltaV1 {
                site: 2,
                lane: 0,
                old_token: 2,
                new_token: 7,
            },
        ];
        let mut score = 0.0;

        // SAFETY: Live session and complete live delta/score storage. The warm
        // up pass materializes anything lazily initialized on first use.
        unsafe {
            for _ in 0..4 {
                assert_eq!(
                    cb2vec_session_push_v1(session, frame.as_ptr(), 2),
                    CB2VEC_OK
                );
                assert_eq!(cb2vec_session_predict_v1(session, &mut score), CB2VEC_OK);
                assert_eq!(cb2vec_session_pop_v1(session, ptr::null_mut()), CB2VEC_OK);
            }

            let guard = crate::testing::AllocationGuard::new();
            for _ in 0..1_000 {
                assert_eq!(
                    cb2vec_session_push_v1(session, frame.as_ptr(), 2),
                    CB2VEC_OK
                );
                assert_eq!(cb2vec_session_predict_v1(session, &mut score), CB2VEC_OK);
                assert_eq!(cb2vec_session_pop_v1(session, ptr::null_mut()), CB2VEC_OK);
            }
            guard.assert_no_allocations("C ABI session loop");
            assert_eq!(cb2vec_session_free_v1(session), CB2VEC_OK);
        }
        assert!(score.is_finite());
    }

    #[test]
    fn checkpoints_resume_training_exactly_through_the_abi() {
        let (shape_ffi, _) = shape_fixture();
        let (config_ffi, _) = config_fixture();
        let fixture = BatchFixture::unequal_mean();
        let batch = fixture.view();
        let trainer = create_trainer(&shape_ffi, &config_ffi);
        let mut metrics = Cb2VecTrainingMetricsV1::default();
        // SAFETY: Live trainer, descriptor buffers, and output.
        unsafe {
            for _ in 0..3 {
                assert_eq!(
                    cb2vec_trainer_train_epoch_v1(trainer, &batch, &mut metrics),
                    CB2VEC_OK
                );
            }
        }

        let mut declared = 0;
        let mut required = 0;
        // SAFETY: Live trainer and writable count outputs; null/zero is the
        // defined size probe.
        unsafe {
            assert_eq!(
                cb2vec_trainer_checkpoint_len_v1(trainer, &mut declared),
                CB2VEC_OK
            );
            assert_eq!(
                cb2vec_trainer_write_checkpoint_v1(trainer, ptr::null_mut(), 0, &mut required),
                CB2VEC_ERROR_BUFFER_TOO_SMALL
            );
        }
        assert_eq!(declared, required);

        let mut short = vec![0xCCu8; required as usize - 1];
        let short_capacity = short.len() as u32;
        // SAFETY: Short buffer is live; capacity is checked before writing.
        unsafe {
            assert_eq!(
                cb2vec_trainer_write_checkpoint_v1(
                    trainer,
                    short.as_mut_ptr(),
                    short_capacity,
                    &mut required,
                ),
                CB2VEC_ERROR_BUFFER_TOO_SMALL
            );
        }
        assert!(short.iter().all(|&byte| byte == 0xCC));

        let mut bytes = vec![0u8; required as usize];
        // SAFETY: Output has exactly the required writable bytes.
        unsafe {
            assert_eq!(
                cb2vec_trainer_write_checkpoint_v1(
                    trainer,
                    bytes.as_mut_ptr(),
                    required,
                    &mut required,
                ),
                CB2VEC_OK,
                "{}",
                last_error_string()
            );
        }

        let mut resumed = ptr::null_mut();
        // SAFETY: Checkpoint bytes are live and the handle output is writable.
        unsafe {
            assert_eq!(
                cb2vec_trainer_load_checkpoint_v1(bytes.as_ptr(), required, &mut resumed),
                CB2VEC_OK,
                "{}",
                last_error_string()
            );
        }

        // Continue both and require identical reports every epoch.
        // SAFETY: Both trainers are live and exclusively accessed.
        unsafe {
            for epoch in 0..4 {
                let mut original = Cb2VecTrainingMetricsV1::default();
                let mut restored = Cb2VecTrainingMetricsV1::default();
                assert_eq!(
                    cb2vec_trainer_train_epoch_v1(trainer, &batch, &mut original),
                    CB2VEC_OK
                );
                assert_eq!(
                    cb2vec_trainer_train_epoch_v1(resumed, &batch, &mut restored),
                    CB2VEC_OK
                );
                assert_eq!(original, restored, "epoch {epoch} diverged after resume");
            }
        }

        // Corruption is rejected without producing a handle.
        let mut corrupted = bytes;
        corrupted[CB2VEC_CHECKPOINT_HEADER_LEN + 8] ^= 0x40;
        let mut rejected = ptr::null_mut();
        // SAFETY: Bytes are live; the call must fail before constructing a handle.
        unsafe {
            assert_eq!(
                cb2vec_trainer_load_checkpoint_v1(
                    corrupted.as_ptr(),
                    corrupted.len() as u32,
                    &mut rejected,
                ),
                CB2VEC_ERROR_CHECKPOINT
            );
        }
        assert!(rejected.is_null());

        // SAFETY: Each handle is uniquely owned and freed exactly once.
        unsafe {
            assert_eq!(cb2vec_trainer_free_v1(resumed), CB2VEC_OK);
            assert_eq!(cb2vec_trainer_free_v1(trainer), CB2VEC_OK);
        }
    }

    #[test]
    fn artifact_v2_carries_its_inference_recipe_and_schema() {
        let (shape_ffi, _) = shape_fixture();
        let (config_ffi, _) = config_fixture();
        let trainer = create_trainer(&shape_ffi, &config_ffi);
        let quantization = Cb2VecQuantizationConfigV1::default();
        let digest = [0x77u8; 32];
        let metadata = Cb2VecArtifactMetadataV1 {
            schema_version: 42,
            schema_digest: [0xAB; 16],
            ..Cb2VecArtifactMetadataV1::default()
        };

        let mut required = 0;
        // SAFETY: Live inputs and writable count; null/zero is the size probe.
        unsafe {
            assert_eq!(
                cb2vec_trainer_write_artifact_v2(
                    trainer,
                    &quantization,
                    digest.as_ptr(),
                    &metadata,
                    ptr::null_mut(),
                    0,
                    &mut required,
                ),
                CB2VEC_ERROR_BUFFER_TOO_SMALL
            );
        }
        let mut bytes = vec![0u8; required as usize];
        // SAFETY: Output has exactly the required writable bytes.
        unsafe {
            assert_eq!(
                cb2vec_trainer_write_artifact_v2(
                    trainer,
                    &quantization,
                    digest.as_ptr(),
                    &metadata,
                    bytes.as_mut_ptr(),
                    required,
                    &mut required,
                ),
                CB2VEC_OK,
                "{}",
                last_error_string()
            );
        }

        let mut info = Cb2VecArtifactInfoV1::default();
        // SAFETY: Artifact bytes are live and info output is writable.
        unsafe {
            assert_eq!(
                cb2vec_artifact_probe_v1(bytes.as_ptr(), required, &mut info),
                CB2VEC_OK,
                "{}",
                last_error_string()
            );
        }
        assert_eq!(info.artifact_version, 2);
        assert_eq!(info.has_inference_config, 1);
        assert_eq!(info.activation, config_ffi.activation);
        assert_eq!(info.pooling, config_ffi.pooling);
        assert_eq!(info.schema_version, 42);
        assert_eq!(info.schema_digest, [0xAB; 16]);
        assert_eq!(info.source_sha256, digest);
        assert_eq!(info.token_count, shape_ffi.token_count);

        // Loading with no explicit recipe uses the stored one.
        let mut model = ptr::null_mut();
        // SAFETY: Artifact bytes live; null recipe/schema are explicit opt-outs.
        unsafe {
            assert_eq!(
                cb2vec_model_load_v2(
                    bytes.as_ptr(),
                    required,
                    ptr::null(),
                    ptr::null(),
                    &mut model,
                ),
                CB2VEC_OK,
                "{}",
                last_error_string()
            );
        }
        let mut model_info = Cb2VecModelInfoV1::default();
        let mut model_metadata = Cb2VecArtifactMetadataV1::default();
        // SAFETY: Live model and writable outputs.
        unsafe {
            assert_eq!(cb2vec_model_get_info_v1(model, &mut model_info), CB2VEC_OK);
            assert_eq!(
                cb2vec_model_get_metadata_v1(model, &mut model_metadata),
                CB2VEC_OK
            );
        }
        assert_eq!(model_info.artifact_version, 2);
        assert_eq!(model_info.activation, config_ffi.activation);
        assert_eq!(model_info.pooling, config_ffi.pooling);
        assert_eq!(model_metadata.schema_version, 42);

        // A conflicting recipe is refused rather than silently overridden.
        let wrong = Cb2VecInferenceConfigV1 {
            activation: CB2VEC_ACTIVATION_IDENTITY,
            pooling: CB2VEC_POOLING_SUM,
            ..Cb2VecInferenceConfigV1::default()
        };
        let mut rejected = ptr::null_mut();
        // SAFETY: Artifact and config are live; the load must fail closed.
        unsafe {
            assert_eq!(
                cb2vec_model_load_v2(bytes.as_ptr(), required, &wrong, ptr::null(), &mut rejected,),
                CB2VEC_ERROR_ARTIFACT
            );
            assert!(rejected.is_null());
            // The v1 entry point applies the same check.
            assert_eq!(
                cb2vec_model_load_v1(bytes.as_ptr(), required, &wrong, &mut rejected),
                CB2VEC_ERROR_ARTIFACT
            );
            assert!(rejected.is_null());
        }

        // A schema mismatch is refused too.
        let other_schema = Cb2VecArtifactMetadataV1 {
            schema_version: 43,
            schema_digest: [0xAB; 16],
            ..Cb2VecArtifactMetadataV1::default()
        };
        // SAFETY: All inputs are live; the load must fail closed.
        unsafe {
            assert_eq!(
                cb2vec_model_load_v2(
                    bytes.as_ptr(),
                    required,
                    ptr::null(),
                    &other_schema,
                    &mut rejected,
                ),
                CB2VEC_ERROR_ARTIFACT
            );
            assert!(rejected.is_null());
            assert_eq!(
                cb2vec_model_load_v2(
                    bytes.as_ptr(),
                    required,
                    ptr::null(),
                    &metadata,
                    &mut rejected,
                ),
                CB2VEC_OK
            );
            assert_eq!(cb2vec_model_free_v1(rejected), CB2VEC_OK);
        }

        // A v1 artifact still probes cleanly and still needs a recipe.
        let mut v1_required = 0;
        // SAFETY: Live inputs and writable count; null/zero is the size probe.
        unsafe {
            assert_eq!(
                cb2vec_trainer_write_artifact_v1(
                    trainer,
                    &quantization,
                    digest.as_ptr(),
                    ptr::null_mut(),
                    0,
                    &mut v1_required,
                ),
                CB2VEC_ERROR_BUFFER_TOO_SMALL
            );
        }
        let mut v1_bytes = vec![0u8; v1_required as usize];
        // SAFETY: Output has exactly the required writable bytes.
        unsafe {
            assert_eq!(
                cb2vec_trainer_write_artifact_v1(
                    trainer,
                    &quantization,
                    digest.as_ptr(),
                    v1_bytes.as_mut_ptr(),
                    v1_required,
                    &mut v1_required,
                ),
                CB2VEC_OK
            );
            let mut v1_info = Cb2VecArtifactInfoV1::default();
            assert_eq!(
                cb2vec_artifact_probe_v1(v1_bytes.as_ptr(), v1_required, &mut v1_info),
                CB2VEC_OK
            );
            assert_eq!(v1_info.artifact_version, 1);
            assert_eq!(v1_info.has_inference_config, 0);
            assert_eq!(v1_info.schema_version, 0);
            // Without a recipe there is nothing to load with.
            assert_eq!(
                cb2vec_model_load_v2(
                    v1_bytes.as_ptr(),
                    v1_required,
                    ptr::null(),
                    ptr::null(),
                    &mut rejected,
                ),
                CB2VEC_ERROR_ARTIFACT
            );
            assert!(rejected.is_null());
            assert_eq!(cb2vec_model_free_v1(model), CB2VEC_OK);
            assert_eq!(cb2vec_trainer_free_v1(trainer), CB2VEC_OK);
        }
    }

    #[test]
    fn batch_prediction_is_all_or_nothing() {
        let source = CodebookWeights::deterministic(4, 2, 3, 2);
        let flat: QuantizedCodebookWeights = source.quantize_i16_s32_s64();
        let artifact = PackedCodebookArtifact::new_flat(source, flat, [0; 32])
            .unwrap()
            .to_bytes()
            .unwrap();
        let inference = Cb2VecInferenceConfigV1 {
            activation: CB2VEC_ACTIVATION_RELU,
            pooling: CB2VEC_POOLING_MEAN,
            ..Cb2VecInferenceConfigV1::default()
        };
        let mut model = ptr::null_mut();
        // SAFETY: Artifact/config are readable and handle output is writable.
        unsafe {
            assert_eq!(
                cb2vec_model_load_v1(
                    artifact.as_ptr(),
                    artifact.len() as u32,
                    &inference,
                    &mut model,
                ),
                CB2VEC_OK
            );
        }
        let fixture = BatchFixture::unequal_mean();
        let batch = fixture.view();
        let mut scores = [0.0f32; 2];
        // SAFETY: Live model, batch buffers, and exact output span.
        unsafe {
            assert_eq!(
                cb2vec_model_predict_batch_v1(model, &batch, scores.as_mut_ptr(), 2),
                CB2VEC_OK
            );
        }
        assert!(scores.iter().all(|score| score.is_finite()));

        let mut canary = [123.0f32; 2];
        // SAFETY: Output is live but the declared length is intentionally wrong;
        // the function rejects it before writing.
        let status =
            unsafe { cb2vec_model_predict_batch_v1(model, &batch, canary.as_mut_ptr(), 1) };
        assert_eq!(status, CB2VEC_ERROR_INVALID_ARGUMENT);
        assert_eq!(canary, [123.0; 2]);
        // SAFETY: Handle is uniquely freed once.
        unsafe { assert_eq!(cb2vec_model_free_v1(model), CB2VEC_OK) };
    }
}
