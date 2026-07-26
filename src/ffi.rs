//! Stable C ABI for native and Unity deployments.
//!
//! The numerical core remains safe Rust. This module is the only place where
//! raw foreign pointers are converted into Rust values. See
//! `include/cb2vec.h` for the complete ownership and thread-safety contract.

#![deny(unsafe_op_in_unsafe_fn)]

use std::cell::RefCell;
use std::ffi::{CString, c_char};
use std::fmt;
use std::mem::{align_of, size_of};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::slice;

use crate::{
    Activation, AdamConfig, GroupedTokens, InferenceConfig, Loss, ModelShape,
    PackedCodebookArtifact, PackedQuantizedPayload, Pooling, Trainer, TrainerConfig, TrainingError,
    TrainingMetrics, TrainingSample, predict_quantized,
};

/// ABI major 1, minor 0.
pub const CB2VEC_ABI_VERSION: u32 = 0x0001_0000;

pub const CB2VEC_OK: i32 = 0;
pub const CB2VEC_ERROR_NULL_POINTER: i32 = -1;
pub const CB2VEC_ERROR_INVALID_ARGUMENT: i32 = -2;
pub const CB2VEC_ERROR_ABI_MISMATCH: i32 = -3;
pub const CB2VEC_ERROR_ARTIFACT: i32 = -4;
pub const CB2VEC_ERROR_MODEL: i32 = -5;
pub const CB2VEC_ERROR_NUMERIC: i32 = -6;
pub const CB2VEC_ERROR_BUFFER_TOO_SMALL: i32 = -7;
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
    static LAST_ERROR: RefCell<CString> =
        RefCell::new(CString::new(Vec::new()).expect("empty CString"));
}

/// Opaque FP32 trainer handle.
pub struct Cb2VecTrainer {
    trainer: Trainer,
}

/// Opaque immutable quantized-model handle.
pub struct Cb2VecWeights {
    payload: PackedQuantizedPayload,
    inference: InferenceConfig,
    original_kind: u32,
    flags: u32,
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

const _: [(); 32] = [(); size_of::<Cb2VecModelShapeV1>()];
const _: [(); 64] = [(); size_of::<Cb2VecTrainerConfigV1>()];
const _: [(); 32] = [(); size_of::<Cb2VecQuantizationConfigV1>()];
const _: [(); 16] = [(); size_of::<Cb2VecInferenceConfigV1>()];
const _: [(); 64] = [(); size_of::<Cb2VecTrainingMetricsV1>()];
const _: [(); 64] = [(); size_of::<Cb2VecModelInfoV1>()];
#[cfg(target_pointer_width = "64")]
const _: [(); 72] = [(); size_of::<Cb2VecTrainingBatchV1>()];
#[cfg(target_pointer_width = "32")]
const _: [(); 48] = [(); size_of::<Cb2VecTrainingBatchV1>()];

#[derive(Debug)]
enum FfiError {
    Null(&'static str),
    Invalid(String),
    Abi(String),
    Artifact(String),
    Model(String),
    Numeric(String),
    BufferTooSmall { required: usize, capacity: usize },
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
            | Self::Numeric(message) => f.write_str(message),
            Self::BufferTooSmall { required, capacity } => write!(
                f,
                "output buffer is too small: capacity {capacity}, required {required}"
            ),
        }
    }
}

fn clear_last_error() {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = CString::new(Vec::new()).expect("empty CString");
    });
}

fn set_last_error(message: impl fmt::Display) {
    let sanitized = message.to_string().replace('\0', "\\0");
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() =
            CString::new(sanitized).expect("NUL bytes were replaced before CString creation");
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

fn decode_shape(shape: Cb2VecModelShapeV1) -> Result<ModelShape, FfiError> {
    if shape.struct_size != size_of::<Cb2VecModelShapeV1>() as u32 {
        return Err(FfiError::Abi(format!(
            "model shape size is {}, expected {}",
            shape.struct_size,
            size_of::<Cb2VecModelShapeV1>()
        )));
    }
    if shape.abi_version != CB2VEC_ABI_VERSION {
        return Err(FfiError::Abi(format!(
            "model shape ABI is 0x{:08x}, expected 0x{CB2VEC_ABI_VERSION:08x}",
            shape.abi_version
        )));
    }
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
    if config.abi_version != CB2VEC_ABI_VERSION {
        return Err(FfiError::Abi(format!(
            "trainer config ABI is 0x{:08x}, expected 0x{CB2VEC_ABI_VERSION:08x}",
            config.abi_version
        )));
    }
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
    if config.abi_version != CB2VEC_ABI_VERSION {
        return Err(FfiError::Abi(format!(
            "quantization config ABI is 0x{:08x}, expected 0x{CB2VEC_ABI_VERSION:08x}",
            config.abi_version
        )));
    }
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

fn quantized_info(weights: &Cb2VecWeights) -> Result<Cb2VecModelInfoV1, FfiError> {
    let (shape, embedding_scale, head_scale, factor_scale) = match &weights.payload {
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
    };
    model_info(
        shape,
        weights.original_kind,
        weights.flags,
        weights.inference,
        embedding_scale,
        head_scale,
        factor_scale,
    )
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
        token_count: u32::try_from(shape.token_count())
            .map_err(|_| FfiError::Model("token_count does not fit u32".to_string()))?,
        group_count: u32::try_from(shape.group_count())
            .map_err(|_| FfiError::Model("group_count does not fit u32".to_string()))?,
        dim: u32::try_from(shape.dim())
            .map_err(|_| FfiError::Model("dim does not fit u32".to_string()))?,
        fm_rank: u32::try_from(shape.fm_rank())
            .map_err(|_| FfiError::Model("fm_rank does not fit u32".to_string()))?,
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

fn predict_payload(input: &GroupedTokens, weights: &Cb2VecWeights) -> Result<f32, FfiError> {
    match &weights.payload {
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
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
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
            payload: PackedQuantizedPayload::Flat(quantized),
            inference: trainer.trainer.inference_config(),
            original_kind: CB2VEC_MODEL_KIND_FLAT,
            flags: 0,
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

/// Loads an immutable quantized model from caller-owned artifact bytes.
///
/// Factored storage is reconstructed once into a flat table for runtime
/// inference speed. The artifact buffer may be released immediately after the
/// call returns.
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
        let artifact = PackedCodebookArtifact::parse(bytes)
            .map_err(|error| FfiError::Artifact(error.to_string()))?;
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
            payload,
            inference,
            original_kind,
            flags,
        };
        quantized_info(&model)?;
        let handle = Box::into_raw(Box::new(model));
        // SAFETY: Output storage was validated and initialized above.
        unsafe { ptr::write(out_model, handle) };
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

#[cfg(test)]
mod tests {
    use std::ffi::CStr;
    use std::mem::{offset_of, size_of};
    use std::ptr;

    use super::*;
    use crate::{CodebookWeights, FactoredQuantizedCodebookWeights, QuantizedCodebookWeights};

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
        assert_eq!(cb2vec_abi_version(), 0x0001_0000);
        // SAFETY: Version is a process-lifetime NUL-terminated static string.
        let version = unsafe { CStr::from_ptr(cb2vec_library_version()) };
        assert_eq!(version.to_bytes(), env!("CARGO_PKG_VERSION").as_bytes());
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
