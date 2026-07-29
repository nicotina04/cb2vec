//! Interrupting and resuming a training run without changing its outcome.
//!
//! An inference artifact restores weights only, so resuming from one restarts
//! Adam with zero moments and reshuffles from the seed. A checkpoint restores
//! the optimizer, the shuffle stream, and epoch progress too, so the resumed
//! run is bit-identical to the one that never stopped.
//!
//! Run with `cargo run --example resume_training`.

use cb2vec::{
    Activation, AdamConfig, GroupedTokens, Loss, ModelShape, PackedCodebookArtifact, Pooling,
    Trainer, TrainerConfig, TrainingSample,
};

fn sample(tokens: &[u16], offsets: &[usize], groups: &[usize], target: f32) -> TrainingSample {
    TrainingSample::new(
        GroupedTokens::new(tokens.to_vec(), offsets.to_vec(), groups.to_vec()).unwrap(),
        target,
    )
}

fn main() {
    let shape = ModelShape::new(4, 2, 3, 1).unwrap();
    let config = TrainerConfig {
        activation: Activation::Relu,
        pooling: Pooling::Mean,
        loss: Loss::MeanSquaredError,
        adam: AdamConfig {
            learning_rate: 0.02,
            ..AdamConfig::default()
        },
        batch_size: 2,
        // Shuffling on is the interesting case: the permutation stream has to
        // survive the interruption as well.
        shuffle: true,
        seed: 0x0BAD_C0DE_1234_5678,
    };
    let samples = [
        sample(&[0, 1], &[0, 1, 2], &[0, 1], 0.25),
        sample(&[2, 0, 1], &[0, 2, 3], &[1, 0], -0.5),
        sample(&[1], &[0, 0, 1], &[0, 1], 0.75),
        sample(&[3, 2], &[0, 1, 2], &[1, 0], 0.1),
        sample(&[0], &[0, 1, 1], &[0, 1], -0.2),
    ];

    let mut uninterrupted = Trainer::from_shape(shape, config).unwrap();
    uninterrupted.train_epochs(&samples, 20).unwrap();

    let checkpoint = uninterrupted.write_checkpoint().unwrap();
    println!(
        "checkpoint after {} epochs / {} optimizer steps: {} bytes",
        uninterrupted.completed_epochs(),
        uninterrupted.optimizer_step(),
        checkpoint.len()
    );

    // Simulate a restart, then run both forward the same distance.
    let mut resumed = Trainer::from_checkpoint(&checkpoint).unwrap();
    let expected = uninterrupted.train_epochs(&samples, 20).unwrap();
    let actual = resumed.train_epochs(&samples, 20).unwrap();
    assert_eq!(expected, actual, "resumed metrics must match exactly");
    assert_eq!(
        uninterrupted
            .weights()
            .embeddings
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        resumed
            .weights()
            .embeddings
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        "resumed weights must match bit for bit"
    );
    println!(
        "resumed run matched bitwise through epoch {} (loss {:.6})",
        resumed.completed_epochs(),
        actual.last().unwrap().mean_loss
    );

    // Contrast: restoring from an inference artifact keeps the weights but
    // deliberately drops optimizer and epoch state.
    let quantized = uninterrupted.weights().quantize_i16_s32_s64();
    let artifact =
        PackedCodebookArtifact::new_flat(uninterrupted.weights().clone(), quantized, [0; 32])
            .unwrap()
            .to_bytes()
            .unwrap();
    let (weights, _) = PackedCodebookArtifact::parse(&artifact)
        .unwrap()
        .into_parts();
    let restarted = Trainer::new(weights, config).unwrap();
    println!(
        "artifact restore: same weights, but optimizer step {} and epoch {}",
        restarted.optimizer_step(),
        restarted.completed_epochs()
    );
    assert_eq!(restarted.optimizer_step(), 0);

    // A corrupted checkpoint is refused rather than silently resumed.
    let mut corrupted = checkpoint;
    let last = corrupted.len() - 1;
    corrupted[last] ^= 0x01;
    match Trainer::from_checkpoint(&corrupted) {
        Ok(_) => panic!("corruption must not be accepted"),
        Err(error) => println!("corrupted checkpoint refused: {error}"),
    }
}
