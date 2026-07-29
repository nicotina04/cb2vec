//! Complete trainer checkpoint for interrupted long training runs.
//!
//! An inference artifact ([`PackedCodebookArtifact`]) is a deployment format:
//! it carries weights so a game can score positions, and restoring one starts
//! optimization over with fresh Adam moments, a fresh shuffle stream, and zero
//! completed epochs. That is the right trade for shipping a model and the
//! wrong one for resuming training.
//!
//! A checkpoint is the other half. It stores everything
//! [`Trainer::train_epoch`] reads or writes, so a run split across process
//! restarts produces bit-identical weights to an uninterrupted run:
//!
//! * FP32 weights — embeddings, head, FM factors, and bias;
//! * Adam first and second moments for every parameter, plus both bias
//!   correction powers and the optimizer step counter;
//! * the shuffle RNG state, including its buffered Box-Muller normal;
//! * completed epochs and the full [`TrainerConfig`].
//!
//! # Layout
//!
//! Little-endian, `CB2VECCK` magic, 144-byte header, then the payload as raw
//! `f32` bit patterns in a fixed order. A CRC-32 over everything after the
//! checksum field rejects corruption, and the header's shape must agree with
//! both the payload length and the weights it decodes.
//!
//! [`PackedCodebookArtifact`]: crate::PackedCodebookArtifact

use std::error::Error;
use std::fmt;

use crate::trainer::{AdamState, Rng64};
use crate::{
    Activation, AdamConfig, CodebookWeights, Loss, ModelError, ModelShape, Pooling, Trainer,
    TrainerConfig, TrainingError,
};

pub const CB2VEC_CHECKPOINT_MAGIC: [u8; 8] = *b"CB2VECCK";
pub const CB2VEC_CHECKPOINT_VERSION: u16 = 1;
pub const CB2VEC_CHECKPOINT_HEADER_LEN: usize = 144;

/// Error returned when writing or restoring a trainer checkpoint.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CheckpointError {
    Truncated {
        actual: usize,
        minimum: usize,
    },
    InvalidMagic,
    UnsupportedVersion(u16),
    UnsupportedHeaderLength(u32),
    NonZeroReserved,
    ChecksumMismatch {
        stored: u32,
        computed: u32,
    },
    LengthMismatch {
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    InvalidShape(String),
    InvalidConfig(String),
    NonFinite(&'static str),
    ValueOutOfRange(&'static str),
    UnsupportedActivation(u32),
    UnsupportedPooling(u32),
    UnsupportedLoss(u32),
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { actual, minimum } => write!(
                f,
                "checkpoint is truncated: got {actual} bytes, need {minimum}"
            ),
            Self::InvalidMagic => write!(f, "invalid CB2Vec checkpoint magic"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported CB2Vec checkpoint version {version}")
            }
            Self::UnsupportedHeaderLength(len) => {
                write!(f, "unsupported CB2Vec checkpoint header length {len}")
            }
            Self::NonZeroReserved => write!(f, "reserved checkpoint bytes must be zero"),
            Self::ChecksumMismatch { stored, computed } => write!(
                f,
                "checkpoint checksum is 0x{stored:08x}, but the content hashes to 0x{computed:08x}"
            ),
            Self::LengthMismatch {
                field,
                actual,
                expected,
            } => write!(
                f,
                "checkpoint {field} length mismatch: got {actual}, expected {expected}"
            ),
            Self::InvalidShape(message) => write!(f, "invalid checkpoint shape: {message}"),
            Self::InvalidConfig(message) => write!(f, "invalid checkpoint config: {message}"),
            Self::NonFinite(field) => write!(f, "checkpoint contains a non-finite {field}"),
            Self::ValueOutOfRange(field) => write!(f, "checkpoint {field} is out of range"),
            Self::UnsupportedActivation(value) => {
                write!(f, "unsupported checkpoint activation {value}")
            }
            Self::UnsupportedPooling(value) => write!(f, "unsupported checkpoint pooling {value}"),
            Self::UnsupportedLoss(value) => write!(f, "unsupported checkpoint loss {value}"),
        }
    }
}

impl Error for CheckpointError {}

impl From<ModelError> for CheckpointError {
    fn from(error: ModelError) -> Self {
        Self::InvalidShape(error.to_string())
    }
}

impl From<TrainingError> for CheckpointError {
    fn from(error: TrainingError) -> Self {
        Self::InvalidConfig(error.to_string())
    }
}

/// Serializes and restores complete [`Trainer`] state.
///
/// This type is a namespace: checkpoints are always produced from and consumed
/// into a live trainer.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TrainerCheckpoint;

impl TrainerCheckpoint {
    /// Exact byte length a checkpoint for `shape` will occupy.
    pub fn byte_len(shape: ModelShape) -> Result<usize, CheckpointError> {
        let scalars = payload_scalars(shape)?;
        scalars
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_add(CB2VEC_CHECKPOINT_HEADER_LEN))
            .ok_or(CheckpointError::ValueOutOfRange("payload length"))
    }
}

impl Trainer {
    /// Serializes complete trainer state, including optimizer and RNG.
    pub fn write_checkpoint(&self) -> Result<Vec<u8>, CheckpointError> {
        let shape = self.weights.validate()?;
        self.adam.check_shape(&self.weights)?;
        let payload_len = payload_scalars(shape)? * 4;

        let mut bytes = vec![0u8; CB2VEC_CHECKPOINT_HEADER_LEN];
        bytes[..8].copy_from_slice(&CB2VEC_CHECKPOINT_MAGIC);
        put_u16(&mut bytes, 8, CB2VEC_CHECKPOINT_VERSION);
        put_u16(&mut bytes, 10, 0);
        put_u32(&mut bytes, 12, CB2VEC_CHECKPOINT_HEADER_LEN as u32);
        put_u64(&mut bytes, 16, payload_len as u64);
        // 24..28 is the checksum, filled in once the payload exists.
        put_u32(&mut bytes, 28, 0);
        put_u32(
            &mut bytes,
            32,
            u32_value(shape.token_count(), "token_count")?,
        );
        put_u32(
            &mut bytes,
            36,
            u32_value(shape.group_count(), "group_count")?,
        );
        put_u32(&mut bytes, 40, u32_value(shape.dim(), "dim")?);
        put_u32(&mut bytes, 44, u32_value(shape.fm_rank(), "fm_rank")?);
        put_u32(&mut bytes, 48, activation_to_u32(self.config.activation));
        put_u32(&mut bytes, 52, pooling_to_u32(self.config.pooling));
        put_u32(&mut bytes, 56, loss_to_u32(self.config.loss));
        put_u32(
            &mut bytes,
            60,
            u32_value(self.config.batch_size, "batch_size")?,
        );
        put_u32(&mut bytes, 64, u32::from(self.config.shuffle));
        put_u32(&mut bytes, 68, 0);
        put_u64(&mut bytes, 72, self.config.seed);
        put_f32(&mut bytes, 80, self.config.adam.learning_rate);
        put_f32(&mut bytes, 84, self.config.adam.beta1);
        put_f32(&mut bytes, 88, self.config.adam.beta2);
        put_f32(&mut bytes, 92, self.config.adam.epsilon);
        put_u64(&mut bytes, 96, self.adam.step);
        put_u64(&mut bytes, 104, self.completed_epochs);
        put_f32(&mut bytes, 112, self.adam.beta1_power);
        put_f32(&mut bytes, 116, self.adam.beta2_power);
        put_u64(&mut bytes, 120, self.shuffle_rng.state);
        put_u32(
            &mut bytes,
            128,
            u32::from(self.shuffle_rng.spare_normal.is_some()),
        );
        put_f32(
            &mut bytes,
            132,
            self.shuffle_rng.spare_normal.unwrap_or(0.0),
        );
        put_u64(&mut bytes, 136, 0);

        bytes.reserve_exact(payload_len);
        append_f32(&mut bytes, &self.weights.embeddings);
        append_f32(&mut bytes, &self.weights.head);
        append_f32(&mut bytes, &self.weights.factors);
        append_f32(&mut bytes, &[self.weights.bias]);
        append_f32(&mut bytes, &self.adam.embedding_m);
        append_f32(&mut bytes, &self.adam.embedding_v);
        append_f32(&mut bytes, &self.adam.head_m);
        append_f32(&mut bytes, &self.adam.head_v);
        append_f32(&mut bytes, &self.adam.factor_m);
        append_f32(&mut bytes, &self.adam.factor_v);
        append_f32(&mut bytes, &[self.adam.bias_m, self.adam.bias_v]);
        debug_assert_eq!(bytes.len(), CB2VEC_CHECKPOINT_HEADER_LEN + payload_len);

        let checksum = crc32(&bytes[28..]);
        put_u32(&mut bytes, 24, checksum);
        Ok(bytes)
    }

    /// Restores a trainer that continues exactly where the writer stopped.
    ///
    /// Corrupted, truncated, or incompatible checkpoints are rejected without
    /// constructing a trainer.
    pub fn from_checkpoint(bytes: &[u8]) -> Result<Self, CheckpointError> {
        if bytes.len() < CB2VEC_CHECKPOINT_HEADER_LEN {
            return Err(CheckpointError::Truncated {
                actual: bytes.len(),
                minimum: CB2VEC_CHECKPOINT_HEADER_LEN,
            });
        }
        if bytes[..8] != CB2VEC_CHECKPOINT_MAGIC {
            return Err(CheckpointError::InvalidMagic);
        }
        let version = read_u16(bytes, 8);
        if version != CB2VEC_CHECKPOINT_VERSION {
            return Err(CheckpointError::UnsupportedVersion(version));
        }
        if read_u16(bytes, 10) != 0 || read_u32(bytes, 28) != 0 || read_u64(bytes, 136) != 0 {
            return Err(CheckpointError::NonZeroReserved);
        }
        let header_len = read_u32(bytes, 12);
        if header_len as usize != CB2VEC_CHECKPOINT_HEADER_LEN {
            return Err(CheckpointError::UnsupportedHeaderLength(header_len));
        }
        let stored_checksum = read_u32(bytes, 24);
        let computed = crc32(&bytes[28..]);
        if stored_checksum != computed {
            return Err(CheckpointError::ChecksumMismatch {
                stored: stored_checksum,
                computed,
            });
        }

        let payload_len = usize::try_from(read_u64(bytes, 16))
            .map_err(|_| CheckpointError::ValueOutOfRange("payload length"))?;
        let expected_total = payload_len
            .checked_add(CB2VEC_CHECKPOINT_HEADER_LEN)
            .ok_or(CheckpointError::ValueOutOfRange("payload length"))?;
        if bytes.len() != expected_total {
            return Err(CheckpointError::LengthMismatch {
                field: "checkpoint bytes",
                actual: bytes.len(),
                expected: expected_total,
            });
        }

        let shape = ModelShape::new(
            read_u32(bytes, 32) as usize,
            read_u32(bytes, 36) as usize,
            read_u32(bytes, 40) as usize,
            read_u32(bytes, 44) as usize,
        )?;
        if payload_scalars(shape)? * 4 != payload_len {
            return Err(CheckpointError::LengthMismatch {
                field: "payload",
                actual: payload_len,
                expected: payload_scalars(shape)? * 4,
            });
        }
        if read_u32(bytes, 68) != 0 {
            return Err(CheckpointError::NonZeroReserved);
        }
        let shuffle = match read_u32(bytes, 64) {
            0 => false,
            1 => true,
            _ => return Err(CheckpointError::ValueOutOfRange("shuffle")),
        };
        let config = TrainerConfig {
            activation: activation_from_u32(read_u32(bytes, 48))?,
            pooling: pooling_from_u32(read_u32(bytes, 52))?,
            loss: loss_from_u32(read_u32(bytes, 56))?,
            adam: AdamConfig {
                learning_rate: read_f32(bytes, 80),
                beta1: read_f32(bytes, 84),
                beta2: read_f32(bytes, 88),
                epsilon: read_f32(bytes, 92),
            },
            batch_size: read_u32(bytes, 60) as usize,
            shuffle,
            seed: read_u64(bytes, 72),
        };

        let beta1_power = read_f32(bytes, 112);
        let beta2_power = read_f32(bytes, 116);
        if !beta1_power.is_finite() || !beta2_power.is_finite() {
            return Err(CheckpointError::NonFinite("Adam bias correction power"));
        }
        let spare_normal = match read_u32(bytes, 128) {
            0 => None,
            1 => {
                let value = read_f32(bytes, 132);
                if !value.is_finite() {
                    return Err(CheckpointError::NonFinite("buffered normal"));
                }
                Some(value)
            }
            _ => return Err(CheckpointError::ValueOutOfRange("buffered normal flag")),
        };

        let embedding_len = shape.embedding_len()?;
        let feature_len = shape.feature_len()?;
        let factor_len = shape.factor_len()?;
        let mut cursor = Cursor::new(&bytes[CB2VEC_CHECKPOINT_HEADER_LEN..]);
        let weights = CodebookWeights {
            dim: shape.dim(),
            fm_rank: shape.fm_rank(),
            embeddings: cursor.read_f32_vec(embedding_len),
            head: cursor.read_f32_vec(feature_len),
            factors: cursor.read_f32_vec(factor_len),
            bias: cursor.read_f32(),
        };
        // `validate` rejects non-finite weights, so corrupted-but-checksummed
        // files that still decode cannot produce a poisoned trainer.
        if weights.validate()? != shape {
            return Err(CheckpointError::InvalidShape(
                "decoded weights do not match the header shape".to_string(),
            ));
        }
        let adam = AdamState {
            embedding_m: cursor.read_f32_vec(embedding_len),
            embedding_v: cursor.read_f32_vec(embedding_len),
            head_m: cursor.read_f32_vec(feature_len),
            head_v: cursor.read_f32_vec(feature_len),
            factor_m: cursor.read_f32_vec(factor_len),
            factor_v: cursor.read_f32_vec(factor_len),
            bias_m: cursor.read_f32(),
            bias_v: cursor.read_f32(),
            beta1_power,
            beta2_power,
            step: read_u64(bytes, 96),
        };
        debug_assert!(cursor.is_finished());
        adam.check_finite()?;

        // Reuse the constructor's config validation, then install the restored
        // optimizer and shuffle state over its fresh defaults.
        let mut trainer = Trainer::new(weights, config)?;
        trainer.adam = adam;
        trainer.shuffle_rng = Rng64 {
            state: read_u64(bytes, 120),
            spare_normal,
        };
        trainer.completed_epochs = read_u64(bytes, 104);
        Ok(trainer)
    }
}

impl AdamState {
    fn check_shape(&self, weights: &CodebookWeights) -> Result<(), CheckpointError> {
        for (field, actual, expected) in [
            (
                "embedding_m",
                self.embedding_m.len(),
                weights.embeddings.len(),
            ),
            (
                "embedding_v",
                self.embedding_v.len(),
                weights.embeddings.len(),
            ),
            ("head_m", self.head_m.len(), weights.head.len()),
            ("head_v", self.head_v.len(), weights.head.len()),
            ("factor_m", self.factor_m.len(), weights.factors.len()),
            ("factor_v", self.factor_v.len(), weights.factors.len()),
        ] {
            if actual != expected {
                return Err(CheckpointError::LengthMismatch {
                    field,
                    actual,
                    expected,
                });
            }
        }
        Ok(())
    }

    fn check_finite(&self) -> Result<(), CheckpointError> {
        let finite = self.bias_m.is_finite()
            && self.bias_v.is_finite()
            && self.embedding_m.iter().all(|value| value.is_finite())
            && self.embedding_v.iter().all(|value| value.is_finite())
            && self.head_m.iter().all(|value| value.is_finite())
            && self.head_v.iter().all(|value| value.is_finite())
            && self.factor_m.iter().all(|value| value.is_finite())
            && self.factor_v.iter().all(|value| value.is_finite());
        if !finite {
            return Err(CheckpointError::NonFinite("Adam moment"));
        }
        Ok(())
    }
}

/// `f32` values in a checkpoint payload: weights, then Adam moments.
fn payload_scalars(shape: ModelShape) -> Result<usize, CheckpointError> {
    let embedding_len = shape.embedding_len()?;
    let feature_len = shape.feature_len()?;
    let factor_len = shape.factor_len()?;
    // Three copies of every parameter (value, first moment, second moment),
    // and three scalars for the bias.
    [embedding_len, feature_len, factor_len]
        .into_iter()
        .try_fold(3usize, |total, len| {
            len.checked_mul(3)
                .and_then(|scaled| total.checked_add(scaled))
        })
        .ok_or(CheckpointError::ValueOutOfRange("payload length"))
}

const fn crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut index = 0usize;
    while index < 256 {
        let mut value = index as u32;
        let mut bit = 0;
        while bit < 8 {
            value = if value & 1 == 1 {
                0xEDB8_8320 ^ (value >> 1)
            } else {
                value >> 1
            };
            bit += 1;
        }
        table[index] = value;
        index += 1;
    }
    table
}

static CRC32_TABLE: [u32; 256] = crc32_table();

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in bytes {
        crc = CRC32_TABLE[((crc ^ u32::from(byte)) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

const fn activation_to_u32(activation: Activation) -> u32 {
    match activation {
        Activation::Identity => 0,
        Activation::Relu => 1,
    }
}

fn activation_from_u32(value: u32) -> Result<Activation, CheckpointError> {
    match value {
        0 => Ok(Activation::Identity),
        1 => Ok(Activation::Relu),
        value => Err(CheckpointError::UnsupportedActivation(value)),
    }
}

const fn pooling_to_u32(pooling: Pooling) -> u32 {
    match pooling {
        Pooling::Sum => 0,
        Pooling::Mean => 1,
    }
}

fn pooling_from_u32(value: u32) -> Result<Pooling, CheckpointError> {
    match value {
        0 => Ok(Pooling::Sum),
        1 => Ok(Pooling::Mean),
        value => Err(CheckpointError::UnsupportedPooling(value)),
    }
}

const fn loss_to_u32(loss: Loss) -> u32 {
    match loss {
        Loss::BinaryCrossEntropyWithLogits => 0,
        Loss::MeanSquaredError => 1,
    }
}

fn loss_from_u32(value: u32) -> Result<Loss, CheckpointError> {
    match value {
        0 => Ok(Loss::BinaryCrossEntropyWithLogits),
        1 => Ok(Loss::MeanSquaredError),
        value => Err(CheckpointError::UnsupportedLoss(value)),
    }
}

fn u32_value(value: usize, field: &'static str) -> Result<u32, CheckpointError> {
    u32::try_from(value).map_err(|_| CheckpointError::ValueOutOfRange(field))
}

fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn put_f32(bytes: &mut [u8], offset: usize, value: f32) {
    put_u32(bytes, offset, value.to_bits());
}

fn append_f32(bytes: &mut Vec<u8>, values: &[f32]) {
    for &value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().expect("two bytes"))
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("four bytes"))
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("eight bytes"))
}

fn read_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_bits(read_u32(bytes, offset))
}

/// Payload reader. Every read is in range because the header's shape has
/// already been checked against the payload length.
struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn read_f32(&mut self) -> f32 {
        let value = read_f32(self.bytes, self.offset);
        self.offset += 4;
        value
    }

    fn read_f32_vec(&mut self, count: usize) -> Vec<f32> {
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.read_f32());
        }
        values
    }

    fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GroupedTokens, TrainingSample};

    fn config() -> TrainerConfig {
        TrainerConfig {
            activation: Activation::Relu,
            pooling: Pooling::Mean,
            loss: Loss::MeanSquaredError,
            adam: AdamConfig {
                learning_rate: 0.02,
                ..AdamConfig::default()
            },
            batch_size: 2,
            // Shuffling on is the interesting case: resuming must reproduce
            // the same permutation stream.
            shuffle: true,
            seed: 0x0BAD_C0DE_1234_5678,
        }
    }

    fn samples() -> Vec<TrainingSample> {
        let inputs = [
            (vec![0u16, 1], vec![0usize, 1, 2], vec![0usize, 1], 0.25f32),
            (vec![2, 0, 1], vec![0, 2, 3], vec![1, 0], -0.5),
            (vec![1], vec![0, 0, 1], vec![0, 1], 0.75),
            (vec![2, 2], vec![0, 1, 2], vec![1, 0], 0.1),
            (vec![0], vec![0, 1, 1], vec![0, 1], -0.2),
        ];
        inputs
            .into_iter()
            .map(|(tokens, offsets, groups, target)| {
                TrainingSample::new(GroupedTokens::new(tokens, offsets, groups).unwrap(), target)
            })
            .collect()
    }

    fn shape() -> ModelShape {
        ModelShape::new(3, 2, 3, 2).unwrap()
    }

    fn assert_same_state(left: &Trainer, right: &Trainer) {
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&left.weights.embeddings),
            bits(&right.weights.embeddings)
        );
        assert_eq!(bits(&left.weights.head), bits(&right.weights.head));
        assert_eq!(bits(&left.weights.factors), bits(&right.weights.factors));
        assert_eq!(left.weights.bias.to_bits(), right.weights.bias.to_bits());
        assert_eq!(bits(&left.adam.embedding_m), bits(&right.adam.embedding_m));
        assert_eq!(bits(&left.adam.embedding_v), bits(&right.adam.embedding_v));
        assert_eq!(bits(&left.adam.head_m), bits(&right.adam.head_m));
        assert_eq!(bits(&left.adam.head_v), bits(&right.adam.head_v));
        assert_eq!(bits(&left.adam.factor_m), bits(&right.adam.factor_m));
        assert_eq!(bits(&left.adam.factor_v), bits(&right.adam.factor_v));
        assert_eq!(left.adam.bias_m.to_bits(), right.adam.bias_m.to_bits());
        assert_eq!(left.adam.bias_v.to_bits(), right.adam.bias_v.to_bits());
        assert_eq!(
            left.adam.beta1_power.to_bits(),
            right.adam.beta1_power.to_bits()
        );
        assert_eq!(
            left.adam.beta2_power.to_bits(),
            right.adam.beta2_power.to_bits()
        );
        assert_eq!(left.adam.step, right.adam.step);
        assert_eq!(left.completed_epochs, right.completed_epochs);
        assert_eq!(left.shuffle_rng.state, right.shuffle_rng.state);
        assert_eq!(
            left.shuffle_rng.spare_normal.map(f32::to_bits),
            right.shuffle_rng.spare_normal.map(f32::to_bits)
        );
        assert_eq!(left.config, right.config);
    }

    #[test]
    fn resumed_training_matches_an_uninterrupted_run_bitwise() {
        let samples = samples();
        let mut uninterrupted = Trainer::from_shape(shape(), config()).unwrap();
        uninterrupted.train_epochs(&samples, 5).unwrap();

        let bytes = uninterrupted.write_checkpoint().unwrap();
        let mut resumed = Trainer::from_checkpoint(&bytes).unwrap();
        assert_same_state(&uninterrupted, &resumed);

        // Continue both for the same number of epochs and batches.
        let continued = uninterrupted.train_epochs(&samples, 7).unwrap();
        let restored = resumed.train_epochs(&samples, 7).unwrap();
        assert_eq!(continued, restored);
        assert_same_state(&uninterrupted, &resumed);
        assert_eq!(uninterrupted.completed_epochs(), 12);

        // Chained checkpoints stay exact too.
        let mut chained = Trainer::from_checkpoint(&bytes).unwrap();
        for _ in 0..7 {
            let bytes = chained.write_checkpoint().unwrap();
            chained = Trainer::from_checkpoint(&bytes).unwrap();
            chained.train_epochs(&samples, 1).unwrap();
        }
        assert_same_state(&uninterrupted, &chained);
    }

    #[test]
    fn checkpoint_captures_the_buffered_normal_of_the_shuffle_rng() {
        // `train_batch` never draws from the shuffle stream, so build a state
        // with a live spare by construction and confirm it survives.
        let mut trainer = Trainer::from_shape(shape(), config()).unwrap();
        trainer.shuffle_rng.spare_normal = Some(0.123_456_79);
        trainer.shuffle_rng.state = 0xDEAD_BEEF_CAFE_F00D;
        let restored = Trainer::from_checkpoint(&trainer.write_checkpoint().unwrap()).unwrap();
        assert_eq!(
            restored.shuffle_rng.spare_normal.map(f32::to_bits),
            Some(0.123_456_79f32.to_bits())
        );
        assert_eq!(restored.shuffle_rng.state, 0xDEAD_BEEF_CAFE_F00D);

        trainer.shuffle_rng.spare_normal = None;
        let restored = Trainer::from_checkpoint(&trainer.write_checkpoint().unwrap()).unwrap();
        assert_eq!(restored.shuffle_rng.spare_normal, None);
    }

    #[test]
    fn every_config_variant_round_trips() {
        for activation in [Activation::Identity, Activation::Relu] {
            for pooling in [Pooling::Sum, Pooling::Mean] {
                for loss in [Loss::BinaryCrossEntropyWithLogits, Loss::MeanSquaredError] {
                    for shuffle in [false, true] {
                        let config = TrainerConfig {
                            activation,
                            pooling,
                            loss,
                            shuffle,
                            ..config()
                        };
                        let trainer = Trainer::from_shape(shape(), config).unwrap();
                        let restored =
                            Trainer::from_checkpoint(&trainer.write_checkpoint().unwrap()).unwrap();
                        assert_eq!(restored.config(), config);
                    }
                }
            }
        }
    }

    #[test]
    fn byte_length_is_predictable_and_shape_dependent() {
        let trainer = Trainer::from_shape(shape(), config()).unwrap();
        let bytes = trainer.write_checkpoint().unwrap();
        assert_eq!(bytes.len(), TrainerCheckpoint::byte_len(shape()).unwrap());
        assert_eq!(&bytes[..8], &CB2VEC_CHECKPOINT_MAGIC);
        // 9 embeddings + 6 head + 12 factors = 27 parameters, times three
        // copies, plus three bias scalars.
        assert_eq!(bytes.len(), CB2VEC_CHECKPOINT_HEADER_LEN + (27 * 3 + 3) * 4);
    }

    #[test]
    fn corrupted_and_incompatible_checkpoints_are_rejected() {
        let trainer = Trainer::from_shape(shape(), config()).unwrap();
        let good = trainer.write_checkpoint().unwrap();

        assert_eq!(
            Trainer::from_checkpoint(&good[..CB2VEC_CHECKPOINT_HEADER_LEN - 1]).err(),
            Some(CheckpointError::Truncated {
                actual: CB2VEC_CHECKPOINT_HEADER_LEN - 1,
                minimum: CB2VEC_CHECKPOINT_HEADER_LEN,
            })
        );

        let mut wrong_magic = good.clone();
        wrong_magic[0] = b'X';
        assert_eq!(
            Trainer::from_checkpoint(&wrong_magic).err(),
            Some(CheckpointError::InvalidMagic)
        );

        let mut future = good.clone();
        put_u16(&mut future, 8, 2);
        assert_eq!(
            Trainer::from_checkpoint(&future).err(),
            Some(CheckpointError::UnsupportedVersion(2))
        );

        // Every single-byte payload corruption must be caught by the checksum.
        for index in [
            CB2VEC_CHECKPOINT_HEADER_LEN,
            CB2VEC_CHECKPOINT_HEADER_LEN + 17,
            good.len() - 1,
            32,
            96,
        ] {
            let mut corrupted = good.clone();
            corrupted[index] ^= 0x01;
            assert!(
                matches!(
                    Trainer::from_checkpoint(&corrupted),
                    Err(CheckpointError::ChecksumMismatch { .. })
                ),
                "byte {index} was not detected"
            );
        }

        // A truncated payload with a consistent checksum is still rejected.
        let mut short = good[..good.len() - 4].to_vec();
        let checksum = crc32(&short[28..]);
        put_u32(&mut short, 24, checksum);
        assert!(matches!(
            Trainer::from_checkpoint(&short),
            Err(CheckpointError::LengthMismatch { .. })
        ));

        // A header claiming a different shape than the payload is rejected.
        let mut wrong_shape = good.clone();
        put_u32(&mut wrong_shape, 40, 4);
        let checksum = crc32(&wrong_shape[28..]);
        put_u32(&mut wrong_shape, 24, checksum);
        assert!(matches!(
            Trainer::from_checkpoint(&wrong_shape),
            Err(CheckpointError::LengthMismatch {
                field: "payload",
                ..
            })
        ));

        // Unknown enum values are rejected rather than silently defaulted.
        let mut bad_loss = good.clone();
        put_u32(&mut bad_loss, 56, 9);
        let checksum = crc32(&bad_loss[28..]);
        put_u32(&mut bad_loss, 24, checksum);
        assert_eq!(
            Trainer::from_checkpoint(&bad_loss).err(),
            Some(CheckpointError::UnsupportedLoss(9))
        );

        // A structurally valid file whose weights decode to NaN is rejected.
        let mut poisoned = good;
        put_u32(&mut poisoned, CB2VEC_CHECKPOINT_HEADER_LEN, 0x7FC0_0000);
        let checksum = crc32(&poisoned[28..]);
        put_u32(&mut poisoned, 24, checksum);
        assert!(matches!(
            Trainer::from_checkpoint(&poisoned),
            Err(CheckpointError::InvalidShape(_))
        ));
    }

    #[test]
    fn crc32_matches_known_vectors() {
        assert_eq!(crc32(b""), 0x0000_0000);
        assert_eq!(crc32(b"a"), 0xE8B7_BE43);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(
            crc32(b"The quick brown fox jumps over the lazy dog"),
            0x414F_A339
        );
    }
}
