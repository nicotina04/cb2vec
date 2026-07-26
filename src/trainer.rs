use std::error::Error;
use std::fmt;

use crate::{
    CodebookWeights, ModelError, ModelShape, QuantizedCodebookAccess, score_quantized_grouped,
};

/// Activation applied after summing every token embedding at a site.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Activation {
    /// Preserve the signed embedding sum.
    Identity,
    /// Clamp negative components to zero.
    Relu,
}

impl Activation {
    #[inline]
    fn apply(self, value: f32) -> f32 {
        match self {
            Self::Identity => value,
            Self::Relu => value.max(0.0),
        }
    }

    #[inline]
    fn derivative(self, preactivation: f32) -> f32 {
        match self {
            Self::Identity => 1.0,
            Self::Relu => f32::from(preactivation > 0.0),
        }
    }
}

/// Reduction used to combine activated sites in the same group.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Pooling {
    /// Add every activated site.
    Sum,
    /// Divide each group sum by the number of sites assigned to that group.
    ///
    /// Every model group must have at least one site.
    Mean,
}

/// Activation and pooling recipe required to reproduce trained inference.
///
/// Artifact version 1 stores weights and quantization metadata only. Keep this
/// small recipe beside that artifact or pass it explicitly to a deployment
/// adapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InferenceConfig {
    pub activation: Activation,
    pub pooling: Pooling,
}

impl InferenceConfig {
    #[inline]
    pub const fn new(activation: Activation, pooling: Pooling) -> Self {
        Self {
            activation,
            pooling,
        }
    }
}

/// Objective optimized by [`Trainer`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Loss {
    /// Binary cross entropy evaluated directly from a logit.
    ///
    /// Targets must be in the inclusive range `0.0..=1.0`.
    BinaryCrossEntropyWithLogits,
    /// Squared error on the raw model score: `(score - target)^2`.
    MeanSquaredError,
}

/// A compact site-preserving token input.
///
/// Site `s` owns `tokens[site_offsets[s]..site_offsets[s + 1]]` and contributes
/// to `site_groups[s]`. Keeping site boundaries is necessary because CB2Vec
/// sums the token embeddings at a site before applying the activation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GroupedTokens {
    tokens: Vec<u16>,
    site_offsets: Vec<usize>,
    site_groups: Vec<usize>,
}

impl GroupedTokens {
    /// Builds a compact token input and validates its offset table.
    pub fn new(
        tokens: Vec<u16>,
        site_offsets: Vec<usize>,
        site_groups: Vec<usize>,
    ) -> Result<Self, SampleIssue> {
        let input = Self {
            tokens,
            site_offsets,
            site_groups,
        };
        input.validate_structure()?;
        Ok(input)
    }

    /// Flat token storage shared by all sites.
    #[inline]
    pub fn tokens(&self) -> &[u16] {
        &self.tokens
    }

    /// Prefix offsets delimiting the token slice for each site.
    #[inline]
    pub fn site_offsets(&self) -> &[usize] {
        &self.site_offsets
    }

    /// Group index for each site.
    #[inline]
    pub fn site_groups(&self) -> &[usize] {
        &self.site_groups
    }

    /// Number of sites in this input.
    #[inline]
    pub fn site_count(&self) -> usize {
        self.site_groups.len()
    }

    fn validate_structure(&self) -> Result<(), SampleIssue> {
        let expected = self
            .site_groups
            .len()
            .checked_add(1)
            .ok_or(SampleIssue::OffsetCount {
                actual: self.site_offsets.len(),
                expected: usize::MAX,
            })?;
        if self.site_offsets.len() != expected {
            return Err(SampleIssue::OffsetCount {
                actual: self.site_offsets.len(),
                expected,
            });
        }
        if self.site_offsets.first().copied() != Some(0) {
            return Err(SampleIssue::FirstOffset);
        }
        if self.site_offsets.last().copied() != Some(self.tokens.len()) {
            return Err(SampleIssue::LastOffset {
                actual: self.site_offsets.last().copied().unwrap_or(0),
                expected: self.tokens.len(),
            });
        }
        for (site, offsets) in self.site_offsets.windows(2).enumerate() {
            if offsets[0] > offsets[1] {
                return Err(SampleIssue::NonMonotonicOffsets { site });
            }
        }
        Ok(())
    }

    fn validate_for(&self, shape: ModelShape, pooling: Pooling) -> Result<Vec<usize>, SampleIssue> {
        self.validate_structure()?;
        let mut group_counts = vec![0usize; shape.group_count()];
        for (site, &group) in self.site_groups.iter().enumerate() {
            if group >= shape.group_count() {
                return Err(SampleIssue::GroupOutOfRange {
                    site,
                    group,
                    group_count: shape.group_count(),
                });
            }
            group_counts[group] += 1;
            for &token in &self.tokens[self.site_offsets[site]..self.site_offsets[site + 1]] {
                if usize::from(token) >= shape.token_count() {
                    return Err(SampleIssue::TokenOutOfRange {
                        site,
                        token,
                        token_count: shape.token_count(),
                    });
                }
            }
        }
        if pooling == Pooling::Mean {
            if let Some(group) = group_counts.iter().position(|&count| count == 0) {
                return Err(SampleIssue::EmptyMeanGroup { group });
            }
        }
        Ok(group_counts)
    }
}

/// One supervised value-training example.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingSample {
    /// Site-preserving categorical input.
    pub input: GroupedTokens,
    /// BCE probability target or raw regression target, depending on the loss.
    pub target: f32,
    /// Positive finite contribution to the batch mean.
    pub weight: f32,
}

impl TrainingSample {
    /// Creates a unit-weight sample.
    #[inline]
    pub fn new(input: GroupedTokens, target: f32) -> Self {
        Self {
            input,
            target,
            weight: 1.0,
        }
    }

    /// Creates a sample with an explicit positive weight.
    #[inline]
    pub fn weighted(input: GroupedTokens, target: f32, weight: f32) -> Self {
        Self {
            input,
            target,
            weight,
        }
    }
}

/// Adam hyperparameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamConfig {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1.0e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1.0e-8,
        }
    }
}

/// Complete deterministic trainer configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainerConfig {
    pub activation: Activation,
    pub pooling: Pooling,
    pub loss: Loss,
    pub adam: AdamConfig,
    pub batch_size: usize,
    pub shuffle: bool,
    pub seed: u64,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            activation: Activation::Relu,
            pooling: Pooling::Mean,
            loss: Loss::BinaryCrossEntropyWithLogits,
            adam: AdamConfig::default(),
            batch_size: 32,
            shuffle: true,
            seed: 0xCB2C_B2EC_0200_0001,
        }
    }
}

impl TrainerConfig {
    fn validate(self) -> Result<(), TrainingError> {
        if self.batch_size == 0 {
            return Err(TrainingError::InvalidConfig("batch_size must be non-zero"));
        }
        if !self.adam.learning_rate.is_finite() || self.adam.learning_rate <= 0.0 {
            return Err(TrainingError::InvalidConfig(
                "Adam learning_rate must be positive and finite",
            ));
        }
        if !self.adam.beta1.is_finite() || !(0.0..1.0).contains(&self.adam.beta1) {
            return Err(TrainingError::InvalidConfig(
                "Adam beta1 must be finite and in 0.0..1.0",
            ));
        }
        if !self.adam.beta2.is_finite() || !(0.0..1.0).contains(&self.adam.beta2) {
            return Err(TrainingError::InvalidConfig(
                "Adam beta2 must be finite and in 0.0..1.0",
            ));
        }
        if !self.adam.epsilon.is_finite() || self.adam.epsilon <= 0.0 {
            return Err(TrainingError::InvalidConfig(
                "Adam epsilon must be positive and finite",
            ));
        }
        Ok(())
    }
}

/// Structural or range error in a [`GroupedTokens`] input.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SampleIssue {
    OffsetCount {
        actual: usize,
        expected: usize,
    },
    FirstOffset,
    LastOffset {
        actual: usize,
        expected: usize,
    },
    NonMonotonicOffsets {
        site: usize,
    },
    TokenOutOfRange {
        site: usize,
        token: u16,
        token_count: usize,
    },
    GroupOutOfRange {
        site: usize,
        group: usize,
        group_count: usize,
    },
    EmptyMeanGroup {
        group: usize,
    },
}

impl fmt::Display for SampleIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OffsetCount { actual, expected } => {
                write!(f, "site_offsets has {actual} entries, expected {expected}")
            }
            Self::FirstOffset => write!(f, "site_offsets must start at zero"),
            Self::LastOffset { actual, expected } => write!(
                f,
                "last site offset is {actual}, expected token length {expected}"
            ),
            Self::NonMonotonicOffsets { site } => {
                write!(f, "site offsets decrease at site {site}")
            }
            Self::TokenOutOfRange {
                site,
                token,
                token_count,
            } => write!(
                f,
                "site {site} token {token} is outside codebook size {token_count}"
            ),
            Self::GroupOutOfRange {
                site,
                group,
                group_count,
            } => write!(
                f,
                "site {site} group {group} is outside group count {group_count}"
            ),
            Self::EmptyMeanGroup { group } => {
                write!(f, "mean pooling group {group} has no sites")
            }
        }
    }
}

impl Error for SampleIssue {}

/// Error returned by training, evaluation, or feature materialization.
#[derive(Debug)]
#[non_exhaustive]
pub enum TrainingError {
    Model(ModelError),
    InvalidConfig(&'static str),
    InvalidInput(SampleIssue),
    InvalidSample {
        index: usize,
        issue: SampleIssue,
    },
    InvalidTarget {
        index: usize,
        target: f32,
        loss: Loss,
    },
    InvalidWeight {
        index: usize,
        weight: f32,
    },
    EmptyBatch,
    EmptyDataset,
    OutputLength {
        actual: usize,
        expected: usize,
    },
    NonFiniteComputation(&'static str),
}

impl fmt::Display for TrainingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(error) => write!(f, "invalid model: {error}"),
            Self::InvalidConfig(message) => write!(f, "invalid trainer config: {message}"),
            Self::InvalidInput(issue) => write!(f, "invalid token input: {issue}"),
            Self::InvalidSample { index, issue } => {
                write!(f, "invalid training sample {index}: {issue}")
            }
            Self::InvalidTarget {
                index,
                target,
                loss,
            } => write!(f, "sample {index} target {target} is invalid for {loss:?}"),
            Self::InvalidWeight { index, weight } => {
                write!(
                    f,
                    "sample {index} weight {weight} must be positive and finite"
                )
            }
            Self::EmptyBatch => write!(f, "training batch must not be empty"),
            Self::EmptyDataset => write!(f, "training dataset must not be empty"),
            Self::OutputLength { actual, expected } => {
                write!(f, "feature output has length {actual}, expected {expected}")
            }
            Self::NonFiniteComputation(field) => {
                write!(f, "training produced a non-finite {field}")
            }
        }
    }
}

impl Error for TrainingError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Model(error) => Some(error),
            Self::InvalidInput(issue) | Self::InvalidSample { issue, .. } => Some(issue),
            _ => None,
        }
    }
}

impl From<ModelError> for TrainingError {
    fn from(error: ModelError) -> Self {
        Self::Model(error)
    }
}

/// Loss summary returned after a batch, epoch, or evaluation pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainingMetrics {
    pub mean_loss: f32,
    pub sample_count: usize,
    pub total_weight: f64,
    pub batch_count: usize,
    pub optimizer_step: u64,
    pub completed_epochs: u64,
}

/// Integer grouped sums and the pooling divisor for each group.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuantizedGroupedFeatures {
    values: Vec<i32>,
    group_divisors: Vec<usize>,
}

impl QuantizedGroupedFeatures {
    #[inline]
    pub fn values(&self) -> &[i32] {
        &self.values
    }

    #[inline]
    pub fn group_divisors(&self) -> &[usize] {
        &self.group_divisors
    }

    #[inline]
    pub fn into_parts(self) -> (Vec<i32>, Vec<usize>) {
        (self.values, self.group_divisors)
    }
}

/// Materializes activated, pooled floating-point features.
pub fn materialize_features_f32(
    input: &GroupedTokens,
    weights: &CodebookWeights,
    activation: Activation,
    pooling: Pooling,
) -> Result<Vec<f32>, TrainingError> {
    let shape = weights.validate()?;
    let mut output = vec![0.0; shape.feature_len()?];
    materialize_features_f32_into(input, weights, activation, pooling, &mut output)?;
    Ok(output)
}

/// Materializes activated, pooled floating-point features into caller storage.
pub fn materialize_features_f32_into(
    input: &GroupedTokens,
    weights: &CodebookWeights,
    activation: Activation,
    pooling: Pooling,
    output: &mut [f32],
) -> Result<(), TrainingError> {
    let shape = weights.validate()?;
    let expected = shape.feature_len()?;
    if output.len() != expected {
        return Err(TrainingError::OutputLength {
            actual: output.len(),
            expected,
        });
    }
    let group_counts = input
        .validate_for(shape, pooling)
        .map_err(TrainingError::InvalidInput)?;
    output.fill(0.0);
    let dim = shape.dim();
    let mut site_sum = vec![0.0; dim];
    for site in 0..input.site_count() {
        site_sum.fill(0.0);
        for &token in &input.tokens[input.site_offsets[site]..input.site_offsets[site + 1]] {
            let start = usize::from(token) * dim;
            for (sum, &embedding) in site_sum
                .iter_mut()
                .zip(&weights.embeddings[start..start + dim])
            {
                *sum += embedding;
            }
        }
        let group_start = input.site_groups[site] * dim;
        for component in 0..dim {
            output[group_start + component] += activation.apply(site_sum[component]);
        }
    }
    if pooling == Pooling::Mean {
        for (group, &count) in group_counts.iter().enumerate() {
            let divisor = count as f32;
            for component in 0..dim {
                output[group * dim + component] /= divisor;
            }
        }
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(TrainingError::NonFiniteComputation("materialized features"));
    }
    Ok(())
}

/// Materializes exact integer embedding sums for quantized grouped scoring.
pub fn materialize_features_quantized<W: QuantizedCodebookAccess>(
    input: &GroupedTokens,
    weights: &W,
    config: InferenceConfig,
) -> Result<QuantizedGroupedFeatures, TrainingError> {
    let shape = validate_quantized_shape(weights)?;
    let group_counts = input
        .validate_for(shape, config.pooling)
        .map_err(TrainingError::InvalidInput)?;
    let dim = shape.dim();
    let mut values = vec![0i32; shape.feature_len()?];
    let mut site_sum = vec![0i32; dim];
    for site in 0..input.site_count() {
        site_sum.fill(0);
        for &token in &input.tokens[input.site_offsets[site]..input.site_offsets[site + 1]] {
            for (component, sum) in site_sum.iter_mut().enumerate() {
                *sum = sum
                    .checked_add(i32::from(weights.embedding(token, component)))
                    .ok_or(ModelError::ArithmeticOverflow("quantized site sum"))?;
            }
        }
        let group_start = input.site_groups[site] * dim;
        for (component, &sum) in site_sum.iter().enumerate() {
            let activated = match config.activation {
                Activation::Identity => sum,
                Activation::Relu => sum.max(0),
            };
            values[group_start + component] = values[group_start + component]
                .checked_add(activated)
                .ok_or(ModelError::ArithmeticOverflow("quantized group sum"))?;
        }
    }
    let group_divisors = match config.pooling {
        Pooling::Sum => vec![1; shape.group_count()],
        Pooling::Mean => group_counts,
    };
    Ok(QuantizedGroupedFeatures {
        values,
        group_divisors,
    })
}

/// Runs the complete checked token-to-score path with quantized weights.
pub fn predict_quantized<W: QuantizedCodebookAccess>(
    input: &GroupedTokens,
    weights: &W,
    config: InferenceConfig,
) -> Result<f32, TrainingError> {
    let features = materialize_features_quantized(input, weights, config)?;
    Ok(score_quantized_grouped(
        features.values(),
        weights,
        features.group_divisors(),
    )?)
}

fn validate_quantized_shape<W: QuantizedCodebookAccess>(
    weights: &W,
) -> Result<ModelShape, TrainingError> {
    if weights.dim() == 0 {
        return Err(ModelError::ZeroDimension("dim").into());
    }
    if weights.embedding_scale() <= 0 {
        return Err(ModelError::NonPositiveScale("embedding_scale").into());
    }
    if weights.head_scale() <= 0 {
        return Err(ModelError::NonPositiveScale("head_scale").into());
    }
    if weights.factor_scale() <= 0 {
        return Err(ModelError::NonPositiveScale("factor_scale").into());
    }
    if weights.head().len() % weights.dim() != 0 {
        return Err(ModelError::LengthMismatch {
            field: "head",
            actual: weights.head().len(),
            expected: (weights.head().len() / weights.dim()) * weights.dim(),
        }
        .into());
    }
    let shape = ModelShape::new(
        weights.token_count(),
        weights.head().len() / weights.dim(),
        weights.dim(),
        weights.fm_rank(),
    )?;
    let expected_factors = shape.factor_len()?;
    if weights.factors().len() != expected_factors {
        return Err(ModelError::LengthMismatch {
            field: "factors",
            actual: weights.factors().len(),
            expected: expected_factors,
        }
        .into());
    }
    if !weights.bias().is_finite() {
        return Err(ModelError::NonFinite("bias").into());
    }
    Ok(shape)
}

/// Pure-Rust FP32 trainer for the CB2Vec embedding and linear/FM value head.
///
/// Gradients are accumulated with frozen batch weights, divided by the sum of
/// sample weights, and applied with one dense Adam step per batch.
#[derive(Clone, Debug)]
pub struct Trainer {
    weights: CodebookWeights,
    config: TrainerConfig,
    adam: AdamState,
    shuffle_rng: Rng64,
    completed_epochs: u64,
}

impl Trainer {
    /// Creates a trainer from existing floating-point weights.
    pub fn new(weights: CodebookWeights, config: TrainerConfig) -> Result<Self, TrainingError> {
        config.validate()?;
        weights.validate()?;
        let adam = AdamState::new(&weights);
        Ok(Self {
            weights,
            config,
            adam,
            shuffle_rng: Rng64::new(config.seed ^ 0x5348_5546_464C_4501),
            completed_epochs: 0,
        })
    }

    /// Creates deterministically initialized weights and their trainer.
    pub fn from_shape(shape: ModelShape, config: TrainerConfig) -> Result<Self, TrainingError> {
        config.validate()?;
        let weights = initialized_weights(shape, config.seed)?;
        Self::new(weights, config)
    }

    /// Current immutable FP32 model.
    #[inline]
    pub fn weights(&self) -> &CodebookWeights {
        &self.weights
    }

    /// Consumes the trainer and returns its FP32 model.
    ///
    /// Call [`Trainer::inference_config`] first if the consumer does not
    /// already know the activation and pooling recipe.
    #[inline]
    pub fn into_weights(self) -> CodebookWeights {
        self.weights
    }

    /// Consumes the trainer and preserves both weights and inference recipe.
    #[inline]
    pub fn into_parts(self) -> (CodebookWeights, InferenceConfig) {
        let inference = self.inference_config();
        (self.weights, inference)
    }

    /// Trainer configuration.
    #[inline]
    pub fn config(&self) -> TrainerConfig {
        self.config
    }

    /// Activation and pooling recipe needed by floating or quantized inference.
    #[inline]
    pub fn inference_config(&self) -> InferenceConfig {
        InferenceConfig::new(self.config.activation, self.config.pooling)
    }

    /// Number of Adam updates already applied.
    #[inline]
    pub fn optimizer_step(&self) -> u64 {
        self.adam.step
    }

    /// Number of successful calls to [`Trainer::train_epoch`].
    #[inline]
    pub fn completed_epochs(&self) -> u64 {
        self.completed_epochs
    }

    /// Returns the raw linear/FM score for one input.
    pub fn predict_logit(&self, input: &GroupedTokens) -> Result<f32, TrainingError> {
        Ok(forward_cache(
            input,
            &self.weights,
            self.config.activation,
            self.config.pooling,
        )?
        .logit)
    }

    /// Returns a numerically stable sigmoid of [`Trainer::predict_logit`].
    pub fn predict_probability(&self, input: &GroupedTokens) -> Result<f32, TrainingError> {
        Ok(sigmoid(self.predict_logit(input)?))
    }

    /// Computes a weighted mean loss without changing weights or optimizer state.
    pub fn evaluate(&self, samples: &[TrainingSample]) -> Result<TrainingMetrics, TrainingError> {
        if samples.is_empty() {
            return Err(TrainingError::EmptyDataset);
        }
        let shape = self.validate_samples(samples)?;
        let mut weighted_loss = 0.0f64;
        let mut total_weight = 0.0f64;
        for sample in samples {
            let logit = forward_cache_validated(
                &sample.input,
                &self.weights,
                shape,
                self.config.activation,
                self.config.pooling,
            )
            .logit;
            let (loss, _) = loss_and_delta(self.config.loss, logit, sample.target);
            if !loss.is_finite() {
                return Err(TrainingError::NonFiniteComputation("evaluation loss"));
            }
            weighted_loss += f64::from(loss) * f64::from(sample.weight);
            total_weight += f64::from(sample.weight);
        }
        let mean_loss = (weighted_loss / total_weight) as f32;
        if !mean_loss.is_finite() {
            return Err(TrainingError::NonFiniteComputation("evaluation loss"));
        }
        Ok(TrainingMetrics {
            mean_loss,
            sample_count: samples.len(),
            total_weight,
            batch_count: 0,
            optimizer_step: self.adam.step,
            completed_epochs: self.completed_epochs,
        })
    }

    /// Trains one explicit batch and applies exactly one Adam update.
    pub fn train_batch(
        &mut self,
        samples: &[TrainingSample],
    ) -> Result<TrainingMetrics, TrainingError> {
        if samples.is_empty() {
            return Err(TrainingError::EmptyBatch);
        }
        let shape = self.validate_samples(samples)?;
        let indices: Vec<usize> = (0..samples.len()).collect();
        let (weighted_loss, total_weight) = self.train_indices(samples, &indices, shape)?;
        Ok(TrainingMetrics {
            mean_loss: (weighted_loss / total_weight) as f32,
            sample_count: samples.len(),
            total_weight,
            batch_count: 1,
            optimizer_step: self.adam.step,
            completed_epochs: self.completed_epochs,
        })
    }

    /// Trains one epoch using configured batching and deterministic shuffling.
    ///
    /// `mean_loss` is the weighted mean of each batch's pre-update loss.
    pub fn train_epoch(
        &mut self,
        samples: &[TrainingSample],
    ) -> Result<TrainingMetrics, TrainingError> {
        if samples.is_empty() {
            return Err(TrainingError::EmptyDataset);
        }
        let shape = self.validate_samples(samples)?;
        let mut order: Vec<usize> = (0..samples.len()).collect();
        if self.config.shuffle {
            shuffle(&mut order, &mut self.shuffle_rng);
        }
        let mut weighted_loss = 0.0f64;
        let mut total_weight = 0.0f64;
        let mut batch_count = 0usize;
        for indices in order.chunks(self.config.batch_size) {
            let (batch_loss, batch_weight) = self.train_indices(samples, indices, shape)?;
            weighted_loss += batch_loss;
            total_weight += batch_weight;
            batch_count += 1;
        }
        self.completed_epochs += 1;
        let mean_loss = (weighted_loss / total_weight) as f32;
        if !mean_loss.is_finite() {
            return Err(TrainingError::NonFiniteComputation("epoch loss"));
        }
        Ok(TrainingMetrics {
            mean_loss,
            sample_count: samples.len(),
            total_weight,
            batch_count,
            optimizer_step: self.adam.step,
            completed_epochs: self.completed_epochs,
        })
    }

    /// Trains multiple epochs and returns one report per epoch.
    pub fn train_epochs(
        &mut self,
        samples: &[TrainingSample],
        epochs: usize,
    ) -> Result<Vec<TrainingMetrics>, TrainingError> {
        let mut reports = Vec::with_capacity(epochs);
        for _ in 0..epochs {
            reports.push(self.train_epoch(samples)?);
        }
        Ok(reports)
    }

    fn validate_samples(&self, samples: &[TrainingSample]) -> Result<ModelShape, TrainingError> {
        let shape = self.weights.validate()?;
        for (index, sample) in samples.iter().enumerate() {
            sample
                .input
                .validate_for(shape, self.config.pooling)
                .map_err(|issue| TrainingError::InvalidSample { index, issue })?;
            let valid_target = sample.target.is_finite()
                && (self.config.loss != Loss::BinaryCrossEntropyWithLogits
                    || (0.0..=1.0).contains(&sample.target));
            if !valid_target {
                return Err(TrainingError::InvalidTarget {
                    index,
                    target: sample.target,
                    loss: self.config.loss,
                });
            }
            if !sample.weight.is_finite() || sample.weight <= 0.0 {
                return Err(TrainingError::InvalidWeight {
                    index,
                    weight: sample.weight,
                });
            }
        }
        Ok(shape)
    }

    fn train_indices(
        &mut self,
        samples: &[TrainingSample],
        indices: &[usize],
        shape: ModelShape,
    ) -> Result<(f64, f64), TrainingError> {
        let mut gradients = Gradients::zeroed(&self.weights);
        let mut weighted_loss = 0.0f64;
        let total_weight = indices
            .iter()
            .map(|&index| f64::from(samples[index].weight))
            .sum::<f64>();
        if !total_weight.is_finite() || total_weight > f64::from(f32::MAX) {
            return Err(TrainingError::NonFiniteComputation("total sample weight"));
        }
        for &index in indices {
            let sample = &samples[index];
            let cache = forward_cache_validated(
                &sample.input,
                &self.weights,
                shape,
                self.config.activation,
                self.config.pooling,
            );
            let (loss, delta) = loss_and_delta(self.config.loss, cache.logit, sample.target);
            if !loss.is_finite() || !delta.is_finite() {
                return Err(TrainingError::NonFiniteComputation("loss gradient"));
            }
            weighted_loss += f64::from(loss) * f64::from(sample.weight);
            accumulate_sample_gradient(
                &mut gradients,
                &sample.input,
                &self.weights,
                &cache,
                self.config.activation,
                self.config.pooling,
                delta * sample.weight,
            );
        }
        gradients.scale((1.0 / total_weight) as f32);
        if !gradients.is_finite() {
            return Err(TrainingError::NonFiniteComputation("gradient"));
        }
        self.adam
            .apply(&mut self.weights, &gradients, self.config.adam)?;
        Ok((weighted_loss, total_weight))
    }
}

#[derive(Clone, Debug)]
struct ForwardCache {
    preactivations: Vec<f32>,
    features: Vec<f32>,
    group_counts: Vec<usize>,
    factor_sums: Vec<f32>,
    logit: f32,
}

fn forward_cache(
    input: &GroupedTokens,
    weights: &CodebookWeights,
    activation: Activation,
    pooling: Pooling,
) -> Result<ForwardCache, TrainingError> {
    let shape = weights.validate()?;
    input
        .validate_for(shape, pooling)
        .map_err(TrainingError::InvalidInput)?;
    let cache = forward_cache_validated(input, weights, shape, activation, pooling);
    if !cache.logit.is_finite() {
        return Err(TrainingError::NonFiniteComputation("score"));
    }
    Ok(cache)
}

fn forward_cache_validated(
    input: &GroupedTokens,
    weights: &CodebookWeights,
    shape: ModelShape,
    activation: Activation,
    pooling: Pooling,
) -> ForwardCache {
    let mut group_counts = vec![0usize; shape.group_count()];
    for &group in &input.site_groups {
        group_counts[group] += 1;
    }
    let dim = shape.dim();
    let mut preactivations = vec![0.0; input.site_count() * dim];
    let mut features = vec![0.0; weights.head.len()];
    for site in 0..input.site_count() {
        let site_start = site * dim;
        for &token in &input.tokens[input.site_offsets[site]..input.site_offsets[site + 1]] {
            let embedding_start = usize::from(token) * dim;
            for component in 0..dim {
                preactivations[site_start + component] +=
                    weights.embeddings[embedding_start + component];
            }
        }
        let group_start = input.site_groups[site] * dim;
        for component in 0..dim {
            features[group_start + component] +=
                activation.apply(preactivations[site_start + component]);
        }
    }
    if pooling == Pooling::Mean {
        for (group, &count) in group_counts.iter().enumerate() {
            let divisor = count as f32;
            for component in 0..dim {
                features[group * dim + component] /= divisor;
            }
        }
    }
    let mut factor_sums = vec![0.0; shape.fm_rank()];
    let mut factor_square_sums = vec![0.0; shape.fm_rank()];
    for (feature, &value) in features.iter().enumerate() {
        let factor_start = feature * shape.fm_rank();
        for ((sum, square_sum), &factor) in factor_sums
            .iter_mut()
            .zip(&mut factor_square_sums)
            .zip(&weights.factors[factor_start..factor_start + shape.fm_rank()])
        {
            let product = factor * value;
            *sum += product;
            *square_sum += product * product;
        }
    }
    let mut logit = weights.bias;
    for (&feature, &head) in features.iter().zip(&weights.head) {
        logit += feature * head;
    }
    for (&sum, &square_sum) in factor_sums.iter().zip(&factor_square_sums) {
        logit += 0.5 * (sum * sum - square_sum);
    }
    ForwardCache {
        preactivations,
        features,
        group_counts,
        factor_sums,
        logit,
    }
}

#[derive(Clone, Debug)]
struct Gradients {
    embeddings: Vec<f32>,
    head: Vec<f32>,
    factors: Vec<f32>,
    bias: f32,
}

impl Gradients {
    fn zeroed(weights: &CodebookWeights) -> Self {
        Self {
            embeddings: vec![0.0; weights.embeddings.len()],
            head: vec![0.0; weights.head.len()],
            factors: vec![0.0; weights.factors.len()],
            bias: 0.0,
        }
    }

    fn scale(&mut self, scale: f32) {
        for value in &mut self.embeddings {
            *value *= scale;
        }
        for value in &mut self.head {
            *value *= scale;
        }
        for value in &mut self.factors {
            *value *= scale;
        }
        self.bias *= scale;
    }

    fn is_finite(&self) -> bool {
        self.bias.is_finite()
            && self.embeddings.iter().all(|value| value.is_finite())
            && self.head.iter().all(|value| value.is_finite())
            && self.factors.iter().all(|value| value.is_finite())
    }
}

fn accumulate_sample_gradient(
    gradients: &mut Gradients,
    input: &GroupedTokens,
    weights: &CodebookWeights,
    cache: &ForwardCache,
    activation: Activation,
    pooling: Pooling,
    output_delta: f32,
) {
    let dim = weights.dim;
    let rank_count = weights.fm_rank;
    let mut feature_gradient = vec![0.0; cache.features.len()];
    for (feature, &x) in cache.features.iter().enumerate() {
        gradients.head[feature] += output_delta * x;
        let factor_start = feature * rank_count;
        let mut derivative = weights.head[feature];
        for rank in 0..rank_count {
            let factor = weights.factors[factor_start + rank];
            let without_self = cache.factor_sums[rank] - factor * x;
            gradients.factors[factor_start + rank] += output_delta * x * without_self;
            derivative += factor * without_self;
        }
        feature_gradient[feature] = output_delta * derivative;
    }
    gradients.bias += output_delta;

    for site in 0..input.site_count() {
        let group = input.site_groups[site];
        let divisor = match pooling {
            Pooling::Sum => 1.0,
            Pooling::Mean => cache.group_counts[group] as f32,
        };
        let site_start = site * dim;
        for component in 0..dim {
            let feature = group * dim + component;
            let site_gradient = feature_gradient[feature]
                * activation.derivative(cache.preactivations[site_start + component])
                / divisor;
            for &token in &input.tokens[input.site_offsets[site]..input.site_offsets[site + 1]] {
                gradients.embeddings[usize::from(token) * dim + component] += site_gradient;
            }
        }
    }
}

#[derive(Clone, Debug)]
struct AdamState {
    embedding_m: Vec<f32>,
    embedding_v: Vec<f32>,
    head_m: Vec<f32>,
    head_v: Vec<f32>,
    factor_m: Vec<f32>,
    factor_v: Vec<f32>,
    bias_m: f32,
    bias_v: f32,
    beta1_power: f32,
    beta2_power: f32,
    step: u64,
}

impl AdamState {
    fn new(weights: &CodebookWeights) -> Self {
        Self {
            embedding_m: vec![0.0; weights.embeddings.len()],
            embedding_v: vec![0.0; weights.embeddings.len()],
            head_m: vec![0.0; weights.head.len()],
            head_v: vec![0.0; weights.head.len()],
            factor_m: vec![0.0; weights.factors.len()],
            factor_v: vec![0.0; weights.factors.len()],
            bias_m: 0.0,
            bias_v: 0.0,
            beta1_power: 1.0,
            beta2_power: 1.0,
            step: 0,
        }
    }

    fn apply(
        &mut self,
        weights: &mut CodebookWeights,
        gradients: &Gradients,
        config: AdamConfig,
    ) -> Result<(), TrainingError> {
        let next_step = self
            .step
            .checked_add(1)
            .ok_or(TrainingError::NonFiniteComputation(
                "optimizer step counter",
            ))?;
        let next_beta1_power = self.beta1_power * config.beta1;
        let next_beta2_power = self.beta2_power * config.beta2;
        let correction1 = 1.0 - next_beta1_power;
        let correction2 = 1.0 - next_beta2_power;
        let valid = adam_slice_candidates_are_finite(
            &weights.embeddings,
            &self.embedding_m,
            &self.embedding_v,
            &gradients.embeddings,
            config,
            correction1,
            correction2,
        ) && adam_slice_candidates_are_finite(
            &weights.head,
            &self.head_m,
            &self.head_v,
            &gradients.head,
            config,
            correction1,
            correction2,
        ) && adam_slice_candidates_are_finite(
            &weights.factors,
            &self.factor_m,
            &self.factor_v,
            &gradients.factors,
            config,
            correction1,
            correction2,
        ) && adam_candidate(
            weights.bias,
            self.bias_m,
            self.bias_v,
            gradients.bias,
            config,
            correction1,
            correction2,
        )
        .is_some();
        if !valid {
            return Err(TrainingError::NonFiniteComputation("Adam update"));
        }

        self.step = next_step;
        self.beta1_power = next_beta1_power;
        self.beta2_power = next_beta2_power;
        adam_update_slice(
            &mut weights.embeddings,
            &mut self.embedding_m,
            &mut self.embedding_v,
            &gradients.embeddings,
            config,
            correction1,
            correction2,
        );
        adam_update_slice(
            &mut weights.head,
            &mut self.head_m,
            &mut self.head_v,
            &gradients.head,
            config,
            correction1,
            correction2,
        );
        adam_update_slice(
            &mut weights.factors,
            &mut self.factor_m,
            &mut self.factor_v,
            &gradients.factors,
            config,
            correction1,
            correction2,
        );
        adam_update_value(
            &mut weights.bias,
            &mut self.bias_m,
            &mut self.bias_v,
            gradients.bias,
            config,
            correction1,
            correction2,
        );
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn adam_slice_candidates_are_finite(
    parameters: &[f32],
    first_moment: &[f32],
    second_moment: &[f32],
    gradients: &[f32],
    config: AdamConfig,
    correction1: f32,
    correction2: f32,
) -> bool {
    parameters
        .iter()
        .zip(first_moment)
        .zip(second_moment)
        .zip(gradients)
        .all(|(((&parameter, &moment1), &moment2), &gradient)| {
            adam_candidate(
                parameter,
                moment1,
                moment2,
                gradient,
                config,
                correction1,
                correction2,
            )
            .is_some()
        })
}

#[allow(clippy::too_many_arguments)]
fn adam_update_slice(
    parameters: &mut [f32],
    first_moment: &mut [f32],
    second_moment: &mut [f32],
    gradients: &[f32],
    config: AdamConfig,
    correction1: f32,
    correction2: f32,
) {
    for (((parameter, moment1), moment2), &gradient) in parameters
        .iter_mut()
        .zip(first_moment)
        .zip(second_moment)
        .zip(gradients)
    {
        adam_update_value(
            parameter,
            moment1,
            moment2,
            gradient,
            config,
            correction1,
            correction2,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn adam_update_value(
    parameter: &mut f32,
    first_moment: &mut f32,
    second_moment: &mut f32,
    gradient: f32,
    config: AdamConfig,
    correction1: f32,
    correction2: f32,
) {
    let (next_parameter, next_first, next_second) = adam_candidate(
        *parameter,
        *first_moment,
        *second_moment,
        gradient,
        config,
        correction1,
        correction2,
    )
    .expect("Adam update was preflighted");
    *parameter = next_parameter;
    *first_moment = next_first;
    *second_moment = next_second;
}

#[allow(clippy::too_many_arguments)]
fn adam_candidate(
    parameter: f32,
    first_moment: f32,
    second_moment: f32,
    gradient: f32,
    config: AdamConfig,
    correction1: f32,
    correction2: f32,
) -> Option<(f32, f32, f32)> {
    let next_first = config.beta1 * first_moment + (1.0 - config.beta1) * gradient;
    let next_second = config.beta2 * second_moment + (1.0 - config.beta2) * gradient * gradient;
    let corrected_first = next_first / correction1;
    let corrected_second = next_second / correction2;
    let next_parameter = parameter
        - config.learning_rate * corrected_first / (corrected_second.sqrt() + config.epsilon);
    if next_parameter.is_finite()
        && next_first.is_finite()
        && next_second.is_finite()
        && next_second >= 0.0
    {
        Some((next_parameter, next_first, next_second))
    } else {
        None
    }
}

fn initialized_weights(shape: ModelShape, seed: u64) -> Result<CodebookWeights, TrainingError> {
    let mut rng = Rng64::new(seed ^ 0x494E_4954_5745_4947);
    let embedding_stddev = (2.0 / shape.dim() as f32).sqrt() * 0.02;
    let feature_len = shape.feature_len()?;
    let head_stddev = (2.0 / feature_len as f32).sqrt() * 0.02;
    let factor_denominator = feature_len
        .checked_add(shape.fm_rank())
        .ok_or(ModelError::ArithmeticOverflow("factor initialization"))?;
    let factor_stddev = (2.0 / factor_denominator as f32).sqrt() * 0.02;
    CodebookWeights::new(
        shape,
        random_normal_vec(&mut rng, shape.embedding_len()?, embedding_stddev),
        random_normal_vec(&mut rng, feature_len, head_stddev),
        random_normal_vec(&mut rng, shape.factor_len()?, factor_stddev),
        0.0,
    )
    .map_err(TrainingError::Model)
}

fn random_normal_vec(rng: &mut Rng64, len: usize, stddev: f32) -> Vec<f32> {
    (0..len).map(|_| rng.standard_normal() * stddev).collect()
}

#[derive(Clone, Debug)]
struct Rng64 {
    state: u64,
    spare_normal: Option<f32>,
}

impl Rng64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
            spare_normal: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        value
    }

    fn unit_f64(&mut self) -> f64 {
        let mantissa = (self.next_u64() >> 11) + 1;
        mantissa as f64 / ((1u64 << 53) + 1) as f64
    }

    fn standard_normal(&mut self) -> f32 {
        if let Some(value) = self.spare_normal.take() {
            return value;
        }
        let radius = (-2.0 * self.unit_f64().ln()).sqrt();
        let angle = std::f64::consts::TAU * self.unit_f64();
        let first = (radius * angle.cos()) as f32;
        self.spare_normal = Some((radius * angle.sin()) as f32);
        first
    }
}

fn shuffle(order: &mut [usize], rng: &mut Rng64) {
    for upper in (1..order.len()).rev() {
        let index = (rng.next_u64() % (upper as u64 + 1)) as usize;
        order.swap(upper, index);
    }
}

fn loss_and_delta(loss: Loss, logit: f32, target: f32) -> (f32, f32) {
    match loss {
        Loss::BinaryCrossEntropyWithLogits => {
            let value = logit.max(0.0) - logit * target + (-logit.abs()).exp().ln_1p();
            (value, sigmoid(logit) - target)
        }
        Loss::MeanSquaredError => {
            let difference = logit - target;
            (difference * difference, 2.0 * difference)
        }
    }
}

#[inline]
fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PackedCodebookArtifact, add_embedding_to, score_f32, score_quantized_uniform};

    fn input(tokens: &[u16], offsets: &[usize], groups: &[usize]) -> GroupedTokens {
        GroupedTokens::new(tokens.to_vec(), offsets.to_vec(), groups.to_vec()).unwrap()
    }

    fn test_weights(rank: usize) -> CodebookWeights {
        let shape = ModelShape::new(3, 2, 2, rank).unwrap();
        let factors = if rank == 0 {
            Vec::new()
        } else {
            vec![0.12, -0.08, 0.05, 0.09]
        };
        CodebookWeights::new(
            shape,
            vec![0.20, 0.30, 0.10, 0.25, 0.35, 0.15],
            vec![0.40, -0.20, 0.15, 0.30],
            factors,
            -0.07,
        )
        .unwrap()
    }

    fn config(loss: Loss) -> TrainerConfig {
        TrainerConfig {
            activation: Activation::Relu,
            pooling: Pooling::Mean,
            loss,
            adam: AdamConfig {
                learning_rate: 0.01,
                ..AdamConfig::default()
            },
            batch_size: 8,
            shuffle: false,
            seed: 7,
        }
    }

    fn sample_loss(
        weights: &CodebookWeights,
        sample: &TrainingSample,
        config: TrainerConfig,
    ) -> f32 {
        let cache =
            forward_cache(&sample.input, weights, config.activation, config.pooling).unwrap();
        loss_and_delta(config.loss, cache.logit, sample.target).0
    }

    fn analytic_gradient(
        weights: &CodebookWeights,
        sample: &TrainingSample,
        config: TrainerConfig,
    ) -> Gradients {
        let cache =
            forward_cache(&sample.input, weights, config.activation, config.pooling).unwrap();
        let (_, delta) = loss_and_delta(config.loss, cache.logit, sample.target);
        let mut gradients = Gradients::zeroed(weights);
        accumulate_sample_gradient(
            &mut gradients,
            &sample.input,
            weights,
            &cache,
            config.activation,
            config.pooling,
            delta,
        );
        gradients
    }

    fn assert_close(left: f32, right: f32, tolerance: f32) {
        assert!(
            (left - right).abs() <= tolerance,
            "{left} differs from {right} by more than {tolerance}"
        );
    }

    #[test]
    fn materialized_features_feed_the_existing_scorer_exactly() {
        let weights = test_weights(1);
        let input = input(&[0, 1, 1, 2], &[0, 2, 3, 4], &[0, 1, 1]);
        let features =
            materialize_features_f32(&input, &weights, Activation::Relu, Pooling::Mean).unwrap();
        assert_eq!(features, vec![0.3, 0.55, 0.225, 0.2]);
        let trainer = Trainer::new(weights.clone(), config(Loss::MeanSquaredError)).unwrap();
        assert_eq!(
            trainer.predict_logit(&input).unwrap().to_bits(),
            score_f32(&features, &weights).unwrap().to_bits()
        );
    }

    #[test]
    fn bce_and_mse_gradients_match_central_differences() {
        for loss in [Loss::BinaryCrossEntropyWithLogits, Loss::MeanSquaredError] {
            let weights = test_weights(1);
            let sample = TrainingSample::new(input(&[0, 1, 1, 2], &[0, 2, 4], &[0, 1]), 0.35);
            let config = config(loss);
            let analytic = analytic_gradient(&weights, &sample, config);
            let epsilon = 1.0e-3;

            for index in 0..weights.embeddings.len() {
                let mut positive = weights.clone();
                positive.embeddings[index] += epsilon;
                let mut negative = weights.clone();
                negative.embeddings[index] -= epsilon;
                let numeric = (sample_loss(&positive, &sample, config)
                    - sample_loss(&negative, &sample, config))
                    / (2.0 * epsilon);
                assert_close(analytic.embeddings[index], numeric, 3.0e-4);
            }
            for index in 0..weights.head.len() {
                let mut positive = weights.clone();
                positive.head[index] += epsilon;
                let mut negative = weights.clone();
                negative.head[index] -= epsilon;
                let numeric = (sample_loss(&positive, &sample, config)
                    - sample_loss(&negative, &sample, config))
                    / (2.0 * epsilon);
                assert_close(analytic.head[index], numeric, 3.0e-4);
            }
            for index in 0..weights.factors.len() {
                let mut positive = weights.clone();
                positive.factors[index] += epsilon;
                let mut negative = weights.clone();
                negative.factors[index] -= epsilon;
                let numeric = (sample_loss(&positive, &sample, config)
                    - sample_loss(&negative, &sample, config))
                    / (2.0 * epsilon);
                assert_close(analytic.factors[index], numeric, 3.0e-4);
            }
            let mut positive = weights.clone();
            positive.bias += epsilon;
            let mut negative = weights.clone();
            negative.bias -= epsilon;
            let numeric = (sample_loss(&positive, &sample, config)
                - sample_loss(&negative, &sample, config))
                / (2.0 * epsilon);
            assert_close(analytic.bias, numeric, 3.0e-4);
        }
    }

    #[test]
    fn repeated_tokens_accumulate_embedding_gradient() {
        let shape = ModelShape::new(1, 1, 1, 0).unwrap();
        let weights = CodebookWeights::new(shape, vec![0.0], vec![1.0], vec![], 0.0).unwrap();
        let config = TrainerConfig {
            activation: Activation::Identity,
            pooling: Pooling::Sum,
            loss: Loss::MeanSquaredError,
            ..config(Loss::MeanSquaredError)
        };
        let single = TrainingSample::new(input(&[0], &[0, 1], &[0]), 1.0);
        let repeated = TrainingSample::new(input(&[0, 0], &[0, 2], &[0]), 1.0);
        let single_gradient = analytic_gradient(&weights, &single, config);
        let repeated_gradient = analytic_gradient(&weights, &repeated, config);
        assert_eq!(
            repeated_gradient.embeddings[0].to_bits(),
            (2.0 * single_gradient.embeddings[0]).to_bits()
        );
    }

    #[test]
    fn relu_blocks_negative_and_zero_site_gradients() {
        let shape = ModelShape::new(1, 1, 1, 0).unwrap();
        let config = TrainerConfig {
            activation: Activation::Relu,
            pooling: Pooling::Sum,
            loss: Loss::MeanSquaredError,
            ..config(Loss::MeanSquaredError)
        };
        let sample = TrainingSample::new(input(&[0], &[0, 1], &[0]), 1.0);
        for embedding in [-1.0, 0.0] {
            let weights =
                CodebookWeights::new(shape, vec![embedding], vec![1.0], vec![], 0.0).unwrap();
            assert_eq!(
                analytic_gradient(&weights, &sample, config).embeddings[0],
                0.0
            );
        }
    }

    #[test]
    fn linear_only_rank_zero_trains_without_factors() {
        let shape = ModelShape::new(2, 1, 1, 0).unwrap();
        let mut trainer = Trainer::from_shape(
            shape,
            TrainerConfig {
                activation: Activation::Identity,
                pooling: Pooling::Sum,
                loss: Loss::BinaryCrossEntropyWithLogits,
                batch_size: 2,
                shuffle: false,
                ..TrainerConfig::default()
            },
        )
        .unwrap();
        let samples = [
            TrainingSample::new(input(&[0], &[0, 1], &[0]), 0.0),
            TrainingSample::new(input(&[1], &[0, 1], &[0]), 1.0),
        ];
        trainer.train_epoch(&samples).unwrap();
        assert!(trainer.weights().factors.is_empty());
    }

    #[test]
    fn bce_and_mse_losses_converge_on_bias_only_targets() {
        for (loss, target, maximum) in [
            (Loss::BinaryCrossEntropyWithLogits, 1.0, 0.08),
            (Loss::MeanSquaredError, 2.0, 0.001),
        ] {
            let shape = ModelShape::new(1, 1, 1, 0).unwrap();
            let weights = CodebookWeights::new(shape, vec![0.0], vec![0.0], vec![], 0.0).unwrap();
            let mut trainer = Trainer::new(
                weights,
                TrainerConfig {
                    activation: Activation::Identity,
                    pooling: Pooling::Sum,
                    loss,
                    adam: AdamConfig {
                        learning_rate: 0.1,
                        ..AdamConfig::default()
                    },
                    batch_size: 2,
                    shuffle: false,
                    seed: 11,
                },
            )
            .unwrap();
            let samples = [
                TrainingSample::new(input(&[0], &[0, 1], &[0]), target),
                TrainingSample::new(input(&[0], &[0, 1], &[0]), target),
            ];
            let initial = trainer.evaluate(&samples).unwrap().mean_loss;
            trainer.train_epochs(&samples, 100).unwrap();
            let final_loss = trainer.evaluate(&samples).unwrap().mean_loss;
            assert!(final_loss < initial);
            assert!(final_loss < maximum, "{loss:?} ended at {final_loss}");
        }
    }

    #[test]
    fn initialization_shuffle_and_training_are_reproducible() {
        let shape = ModelShape::new(3, 1, 2, 1).unwrap();
        let config = TrainerConfig {
            pooling: Pooling::Sum,
            batch_size: 2,
            seed: 0x1234,
            ..TrainerConfig::default()
        };
        let samples = [
            TrainingSample::new(input(&[0], &[0, 1], &[0]), 0.0),
            TrainingSample::new(input(&[1], &[0, 1], &[0]), 1.0),
            TrainingSample::new(input(&[2], &[0, 1], &[0]), 0.5),
        ];
        let mut left = Trainer::from_shape(shape, config).unwrap();
        let mut right = Trainer::from_shape(shape, config).unwrap();
        let left_reports = left.train_epochs(&samples, 4).unwrap();
        let right_reports = right.train_epochs(&samples, 4).unwrap();
        assert_eq!(left_reports, right_reports);
        assert_eq!(
            left.weights()
                .embeddings
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            right
                .weights()
                .embeddings
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            left.weights()
                .head
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            right
                .weights()
                .head
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            left.weights()
                .factors
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            right
                .weights()
                .factors
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            left.weights().bias.to_bits(),
            right.weights().bias.to_bits()
        );
    }

    #[test]
    fn invalid_batch_does_not_change_weights_or_optimizer() {
        let mut trainer = Trainer::new(test_weights(1), config(Loss::MeanSquaredError)).unwrap();
        let before = trainer.weights().clone();
        let bad = TrainingSample::new(input(&[9], &[0, 1], &[0]), 0.0);
        assert!(matches!(
            trainer.train_batch(&[bad]),
            Err(TrainingError::InvalidSample {
                issue: SampleIssue::TokenOutOfRange { .. },
                ..
            })
        ));
        assert_eq!(trainer.optimizer_step(), 0);
        assert_eq!(
            trainer
                .weights()
                .embeddings
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            before
                .embeddings
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(trainer.weights().bias.to_bits(), before.bias.to_bits());
    }

    #[test]
    fn final_partial_batch_uses_one_optimizer_step() {
        let shape = ModelShape::new(1, 1, 1, 0).unwrap();
        let mut trainer = Trainer::from_shape(
            shape,
            TrainerConfig {
                pooling: Pooling::Sum,
                batch_size: 2,
                shuffle: false,
                ..TrainerConfig::default()
            },
        )
        .unwrap();
        let samples = vec![
            TrainingSample::new(input(&[0], &[0, 1], &[0]), 0.0),
            TrainingSample::new(input(&[0], &[0, 1], &[0]), 1.0),
            TrainingSample::new(input(&[0], &[0, 1], &[0]), 0.5),
        ];
        let report = trainer.train_epoch(&samples).unwrap();
        assert_eq!(report.batch_count, 2);
        assert_eq!(report.optimizer_step, 2);
    }

    #[test]
    fn trained_model_quantizes_and_round_trips_through_artifact() {
        let shape = ModelShape::new(2, 1, 1, 0).unwrap();
        let mut trainer = Trainer::from_shape(
            shape,
            TrainerConfig {
                activation: Activation::Identity,
                pooling: Pooling::Sum,
                loss: Loss::MeanSquaredError,
                adam: AdamConfig {
                    learning_rate: 0.03,
                    ..AdamConfig::default()
                },
                batch_size: 2,
                shuffle: false,
                seed: 99,
            },
        )
        .unwrap();
        let samples = [
            TrainingSample::new(input(&[0], &[0, 1], &[0]), -0.5),
            TrainingSample::new(input(&[1], &[0, 1], &[0]), 0.75),
        ];
        trainer.train_epochs(&samples, 50).unwrap();
        let source = trainer.into_weights();
        let quantized = source.quantize_i16_s32_s64();
        let mut integer_features = vec![0i32; quantized.dim];
        add_embedding_to(&quantized, 1, &mut integer_features).unwrap();
        let quantized_score = score_quantized_uniform(&integer_features, &quantized, 1).unwrap();
        let artifact = PackedCodebookArtifact::new_flat(source, quantized, [0xA5; 32]).unwrap();
        let bytes = artifact.to_bytes().unwrap();
        let parsed = PackedCodebookArtifact::parse(&bytes).unwrap();
        let parsed_quantized = parsed.flat_quantized().unwrap();
        assert_eq!(
            quantized_score.to_bits(),
            score_quantized_uniform(&integer_features, parsed_quantized, 1)
                .unwrap()
                .to_bits()
        );
    }

    #[test]
    fn quantized_grouped_path_matches_fp32_for_relu_and_unequal_means() {
        let shape = ModelShape::new(3, 2, 2, 1).unwrap();
        let source = CodebookWeights::new(
            shape,
            vec![0.25, -0.5, -0.125, 0.25, 0.375, 0.125],
            vec![0.25, -0.125, 0.0625, 0.375],
            vec![0.125, -0.25, 0.1875, 0.0625],
            0.03125,
        )
        .unwrap();
        let quantized = source.quantize_i16_s32_s64();
        let exactly_dequantized = quantized.dequantized();
        let state = input(
            &[0, 1, 2, 1, 1, 0, 2],
            &[0, 2, 3, 5, 6, 7],
            &[0, 0, 1, 1, 1],
        );
        let inference = InferenceConfig::new(Activation::Relu, Pooling::Mean);
        let float_features = materialize_features_f32(
            &state,
            &exactly_dequantized,
            inference.activation,
            inference.pooling,
        )
        .unwrap();
        let float_score = score_f32(&float_features, &exactly_dequantized).unwrap();
        let integer_score = predict_quantized(&state, &quantized, inference).unwrap();
        assert_close(float_score, integer_score, 1.0e-6);

        let artifact = PackedCodebookArtifact::new_flat(source, quantized, [0x5A; 32]).unwrap();
        let bytes = artifact.to_bytes().unwrap();
        let parsed = PackedCodebookArtifact::parse(&bytes).unwrap();
        assert_eq!(
            integer_score.to_bits(),
            predict_quantized(&state, parsed.flat_quantized().unwrap(), inference)
                .unwrap()
                .to_bits()
        );
    }

    #[test]
    fn trainer_preserves_the_inference_recipe_when_consumed() {
        let shape = ModelShape::new(1, 1, 1, 0).unwrap();
        let trainer = Trainer::from_shape(
            shape,
            TrainerConfig {
                activation: Activation::Identity,
                pooling: Pooling::Sum,
                ..TrainerConfig::default()
            },
        )
        .unwrap();
        let (_, inference) = trainer.into_parts();
        assert_eq!(
            inference,
            InferenceConfig::new(Activation::Identity, Pooling::Sum)
        );
    }

    #[test]
    fn extreme_total_weight_and_feature_overflow_fail_closed() {
        let shape = ModelShape::new(1, 1, 1, 0).unwrap();
        let weights = CodebookWeights::new(shape, vec![0.0], vec![0.0], vec![], 0.0).unwrap();
        let mut trainer = Trainer::new(
            weights,
            TrainerConfig {
                activation: Activation::Identity,
                pooling: Pooling::Sum,
                batch_size: 2,
                shuffle: false,
                ..TrainerConfig::default()
            },
        )
        .unwrap();
        let samples = [
            TrainingSample::weighted(input(&[0], &[0, 1], &[0]), 1.0, f32::MAX),
            TrainingSample::weighted(input(&[0], &[0, 1], &[0]), 1.0, f32::MAX),
        ];
        assert_close(
            trainer.evaluate(&samples).unwrap().mean_loss,
            std::f32::consts::LN_2,
            1.0e-6,
        );
        assert!(matches!(
            trainer.train_batch(&samples),
            Err(TrainingError::NonFiniteComputation("total sample weight"))
        ));
        assert_eq!(trainer.optimizer_step(), 0);

        let overflow_weights =
            CodebookWeights::new(shape, vec![f32::MAX], vec![0.0], vec![], 0.0).unwrap();
        let overflow_input = input(&[0, 0], &[0, 2], &[0]);
        assert!(matches!(
            materialize_features_f32(
                &overflow_input,
                &overflow_weights,
                Activation::Identity,
                Pooling::Sum,
            ),
            Err(TrainingError::NonFiniteComputation("materialized features"))
        ));
    }

    #[test]
    fn adam_numeric_failure_is_transactional() {
        let mut weights = test_weights(1);
        let before_weights = weights.clone();
        let mut state = AdamState::new(&weights);
        let before_state = state.clone();
        let mut gradients = Gradients::zeroed(&weights);
        gradients.embeddings[0] = 1.0e21;
        assert!(matches!(
            state.apply(&mut weights, &gradients, AdamConfig::default()),
            Err(TrainingError::NonFiniteComputation("Adam update"))
        ));
        assert_eq!(state.step, before_state.step);
        assert_eq!(
            state.beta1_power.to_bits(),
            before_state.beta1_power.to_bits()
        );
        assert_eq!(
            state.beta2_power.to_bits(),
            before_state.beta2_power.to_bits()
        );
        assert_eq!(
            state
                .embedding_m
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            before_state
                .embedding_m
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            weights
                .embeddings
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            before_weights
                .embeddings
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(weights.bias.to_bits(), before_weights.bias.to_bits());
    }

    #[test]
    fn invalid_configs_and_mean_inputs_fail_closed() {
        let shape = ModelShape::new(1, 2, 1, 0).unwrap();
        let bad_config = TrainerConfig {
            batch_size: 0,
            ..TrainerConfig::default()
        };
        assert!(matches!(
            Trainer::from_shape(shape, bad_config),
            Err(TrainingError::InvalidConfig(_))
        ));

        let trainer = Trainer::from_shape(shape, TrainerConfig::default()).unwrap();
        let one_group = input(&[0], &[0, 1], &[0]);
        assert!(matches!(
            trainer.predict_logit(&one_group),
            Err(TrainingError::InvalidInput(SampleIssue::EmptyMeanGroup {
                group: 1
            }))
        ));
    }
}
