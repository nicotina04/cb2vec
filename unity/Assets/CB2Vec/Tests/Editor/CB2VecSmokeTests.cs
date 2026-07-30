using NUnit.Framework;

namespace CB2Vec.Tests
{
    /// <summary>
    /// Proves the native library loads and the whole pipeline works inside the
    /// Editor: create, train, save, load, and incremental search.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These run against the Windows Editor plug-in, so a failure here is
    /// almost always an install problem rather than a model problem: a missing
    /// <c>cb2vec.dll</c>, a plug-in disabled for the Editor, or an ABI mismatch
    /// between the binding and the built library.
    /// </para>
    /// <para>
    /// The model is deliberately tiny: two tokens, one group, one site per
    /// sample. Token 0 targets 0.0 and token 1 targets 1.0, which is separable
    /// and so converges in a fraction of a second.
    /// </para>
    /// </remarks>
    [TestFixture]
    public sealed class CB2VecSmokeTests
    {
        private const int TrainingEpochs = 200;

        /// <summary>One site holding one token in group 0.</summary>
        private static Cb2VecInput SingleSite(ushort token)
        {
            return new Cb2VecInput(
                new ushort[] { token },
                new uint[] { 0, 1 },
                new uint[] { 0 });
        }

        /// <summary>Four single-site samples alternating token 0 and token 1.</summary>
        private static Cb2VecTrainingBatch Dataset()
        {
            return new Cb2VecTrainingBatch(
                new ushort[] { 0, 1, 0, 1 },
                new uint[] { 0, 1, 2, 3, 4 },
                new uint[] { 0, 0, 0, 0 },
                new uint[] { 0, 1, 2, 3, 4 },
                new float[] { 0f, 1f, 0f, 1f });
        }

        private static Cb2VecModelShapeV1 Shape()
        {
            Cb2VecModelShapeV1 shape = Cb2VecNative.DefaultShape();
            shape.TokenCount = 2;
            shape.GroupCount = 1;
            shape.Dim = 4;
            shape.FmRank = 2;
            return shape;
        }

        private static Cb2VecTrainerConfigV1 TrainerConfig()
        {
            Cb2VecTrainerConfigV1 config = Cb2VecNative.DefaultTrainerConfig();
            config.LearningRate = 0.05f;
            config.BatchSize = 4;
            config.Seed = 42;
            return config;
        }

        [Test]
        public void NativeLibraryLoadsWithMatchingAbi()
        {
            Assert.AreEqual(
                Cb2VecNative.AbiVersion >> 16,
                Cb2VecNative.NativeAbiVersion >> 16,
                "Native ABI major version disagrees with the binding.");
            Assert.IsNotEmpty(Cb2VecNative.LibraryVersion);
        }

        [Test]
        public void TrainingReducesLoss()
        {
            using (Cb2VecTrainer trainer = Cb2VecTrainer.Create(Shape(), TrainerConfig()))
            {
                Cb2VecTrainingBatch dataset = Dataset();
                float before = trainer.Evaluate(dataset).MeanLoss;
                for (int epoch = 0; epoch < TrainingEpochs; ++epoch)
                    trainer.TrainEpoch(dataset);
                float after = trainer.Evaluate(dataset).MeanLoss;

                Assert.Less(after, before, "Training did not reduce mean loss.");
                Assert.Less(
                    trainer.PredictProbability(SingleSite(0)),
                    trainer.PredictProbability(SingleSite(1)),
                    "The trained model does not separate the two tokens.");
            }
        }

        [Test]
        public void ArtifactRoundTripsThroughModelAndSession()
        {
            byte[] artifact = TrainAndWriteArtifact();

            // A version-2 artifact carries its own inference recipe, so loading
            // it cannot pick the wrong activation or pooling.
            using (Cb2VecModel model = Cb2VecModel.Load(artifact))
            {
                float modelScore = model.Predict(SingleSite(1));

                Cb2VecSessionConfigV1 config = Cb2VecNative.DefaultSessionConfig();
                config.MaxSites = 4;
                config.MaxTokenSlots = 8;
                config.MaxDeltasPerFrame = 2;
                config.MaxDepth = 4;

                using (Cb2VecSession session = model.CreateSession(config))
                {
                    session.Reset(SingleSite(1));
                    Assert.AreEqual(
                        modelScore,
                        session.Predict(),
                        "A session must score a position identically to the model.");
                }
            }
        }

        [Test]
        public void SessionPushAndPopRestoreTheOriginalScore()
        {
            byte[] artifact = TrainAndWriteArtifact();

            using (Cb2VecModel model = Cb2VecModel.Load(artifact))
            {
                Cb2VecSessionConfigV1 config = Cb2VecNative.DefaultSessionConfig();
                config.MaxSites = 4;
                config.MaxTokenSlots = 8;
                config.MaxDeltasPerFrame = 2;
                config.MaxDepth = 4;

                using (Cb2VecSession session = model.CreateSession(config))
                {
                    session.Reset(SingleSite(0));
                    float original = session.Predict();

                    // Replace token 0 with token 1 at site 0, lane 0.
                    Cb2VecTokenDeltaV1[] deltas =
                    {
                        new Cb2VecTokenDeltaV1(0, 0, 0, 1),
                    };
                    session.Push(deltas, deltas.Length);
                    float moved = session.Predict();
                    Assert.AreNotEqual(
                        original, moved, "Pushing a token change did not alter the score.");
                    Assert.AreEqual(
                        model.Predict(SingleSite(1)),
                        moved,
                        "An incrementally updated score must match a full evaluation.");

                    Assert.AreEqual(deltas.Length, session.Pop(), "Pop reported the wrong count.");
                    Assert.AreEqual(
                        original, session.Predict(), "Pop did not restore the original score.");
                }
            }
        }

        [Test]
        public void CheckpointResumesTrainerState()
        {
            byte[] checkpoint;
            float loss;
            using (Cb2VecTrainer trainer = Cb2VecTrainer.Create(Shape(), TrainerConfig()))
            {
                Cb2VecTrainingBatch dataset = Dataset();
                for (int epoch = 0; epoch < TrainingEpochs; ++epoch)
                    trainer.TrainEpoch(dataset);
                loss = trainer.Evaluate(dataset).MeanLoss;
                checkpoint = trainer.WriteCheckpoint();
                Assert.AreEqual(trainer.CheckpointLength(), checkpoint.Length);
            }

            using (Cb2VecTrainer resumed = Cb2VecTrainer.LoadCheckpoint(checkpoint))
            {
                Assert.AreEqual(
                    loss,
                    resumed.Evaluate(Dataset()).MeanLoss,
                    "A resumed trainer must evaluate identically to the one it came from.");
            }
        }

        private static byte[] TrainAndWriteArtifact()
        {
            using (Cb2VecTrainer trainer = Cb2VecTrainer.Create(Shape(), TrainerConfig()))
            {
                Cb2VecTrainingBatch dataset = Dataset();
                for (int epoch = 0; epoch < TrainingEpochs; ++epoch)
                    trainer.TrainEpoch(dataset);

                // No upstream source file here, so the provenance digest is zero.
                byte[] artifact = trainer.WriteArtifactV2(
                    Cb2VecNative.DefaultQuantization(),
                    new byte[32],
                    Cb2VecNative.EmptyMetadata());
                Assert.IsNotEmpty(artifact);
                return artifact;
            }
        }
    }
}
