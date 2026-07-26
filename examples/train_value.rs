use cb2vec::{
    Activation, AdamConfig, GroupedTokens, Loss, ModelShape, PackedCodebookArtifact, Pooling,
    Trainer, TrainerConfig, TrainingSample, predict_quantized,
};

fn state(token: u16) -> Result<GroupedTokens, cb2vec::SampleIssue> {
    GroupedTokens::new(vec![token], vec![0, 1], vec![0])
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let samples = vec![
        TrainingSample::new(state(0)?, 0.0),
        TrainingSample::new(state(1)?, 1.0),
        TrainingSample::new(state(0)?, 0.0),
        TrainingSample::new(state(1)?, 1.0),
    ];
    let shape = ModelShape::new(2, 1, 4, 2)?;
    let mut trainer = Trainer::from_shape(
        shape,
        TrainerConfig {
            activation: Activation::Relu,
            pooling: Pooling::Sum,
            loss: Loss::BinaryCrossEntropyWithLogits,
            adam: AdamConfig {
                learning_rate: 0.03,
                ..AdamConfig::default()
            },
            batch_size: 4,
            shuffle: true,
            seed: 42,
        },
    )?;

    let initial_loss = trainer.evaluate(&samples)?.mean_loss;
    trainer.train_epochs(&samples, 100)?;
    let final_loss = trainer.evaluate(&samples)?.mean_loss;
    assert!(final_loss < initial_loss);

    let probability_zero = trainer.predict_probability(&samples[0].input)?;
    let probability_one = trainer.predict_probability(&samples[1].input)?;
    let (source, inference) = trainer.into_parts();
    let quantized = source.quantize_i16_s32_s64();
    let quantized_score = predict_quantized(&samples[1].input, &quantized, inference)?;
    let artifact = PackedCodebookArtifact::new_flat(source, quantized, [0; 32])?;
    let artifact_bytes = artifact.to_bytes()?;

    println!(
        "loss {initial_loss:.4} -> {final_loss:.4}; p(0)={probability_zero:.3}, \
         p(1)={probability_one:.3}; quantized score={quantized_score:.3}; \
         artifact={} bytes",
        artifact_bytes.len()
    );
    Ok(())
}
