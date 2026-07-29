using System;
using CB2Vec;

/// <summary>
/// End-to-end P/Invoke smoke test for the CB2Vec C# binding.
///
/// Covers the ABI 1.0 trainer/PTQ/reload path plus every ABI 1.1 addition:
/// incremental sessions, allocation-free inference, trainer checkpoints, and
/// version-2 artifacts. Each check returns a distinct exit code so a CI
/// failure names the step that broke.
/// </summary>
internal static class Program
{
    /// Five sites over a nine-token ragged layout, matching examples/c_api_smoke.c:
    ///   site 0 -> tokens[0..3) group 0    site 3 -> tokens[7..8) group 2
    ///   site 1 -> tokens[3..5) group 1    site 4 -> tokens[8..9) group 0
    ///   site 2 -> tokens[5..7) group 1
    private static readonly ushort[] Tokens = { 0, 3, 5, 1, 8, 2, 7, 4, 6 };
    private static readonly uint[] SiteOffsets = { 0, 3, 5, 7, 8, 9 };
    private static readonly uint[] SiteGroups = { 0, 1, 1, 2, 0 };

    private static byte[] _artifactCache;

    private static bool BitsEqual(float left, float right)
    {
        return BitConverter.SingleToInt32Bits(left) ==
               BitConverter.SingleToInt32Bits(right);
    }

    private static Cb2VecInput Input(ushort[] tokens)
    {
        return new Cb2VecInput(tokens, SiteOffsets, SiteGroups);
    }

    private static Cb2VecSessionConfigV1 SessionConfig()
    {
        Cb2VecSessionConfigV1 config = Cb2VecNative.DefaultSessionConfig();
        config.MaxSites = 8;
        config.MaxTokenSlots = 16;
        config.MaxDeltasPerFrame = 4;
        config.MaxDepth = 6;
        return config;
    }

    private static int Main()
    {
        // ABI 1.1 must still report major 1 so a 1.0 consumer keeps working.
        uint nativeAbi = Cb2VecNative.NativeAbiVersion;
        if ((nativeAbi >> 16) != 1 || nativeAbi < Cb2VecNative.AbiVersion10)
            return 1;

        Cb2VecModelShapeV1 shape = Cb2VecNative.DefaultShape();
        shape.TokenCount = 9;
        shape.GroupCount = 3;
        shape.Dim = 4;
        shape.FmRank = 2;

        Cb2VecTrainerConfigV1 config = Cb2VecNative.DefaultTrainerConfig();
        config.Activation = (uint)Cb2VecActivation.Relu;
        config.Pooling = (uint)Cb2VecPooling.Mean;
        config.Shuffle = 1;
        config.BatchSize = 2;

        // Mean pooling requires every model group to own at least one site, so
        // both samples span all three groups. Site 2 deliberately owns no
        // tokens, which is legal and still counts toward its group.
        var batch = new Cb2VecTrainingBatch(
            new ushort[] { 0, 0, 1, 2, 3, 1, 4 },
            new uint[] { 0, 2, 3, 3, 4, 6, 7 },
            new uint[] { 0, 1, 2, 0, 1, 2 },
            new uint[] { 0, 3, 6 },
            new float[] { 0.2f, 0.8f },
            new float[] { 1.0f, 2.0f });

        byte[] artifactV1;
        byte[] artifactV2;
        byte[] checkpoint;
        Cb2VecArtifactMetadataV1 schema = Cb2VecNative.Metadata(7, Repeat(0xC3, 16));

        using (Cb2VecTrainer trainer = Cb2VecTrainer.Create(shape, config))
        {
            Cb2VecTrainingMetricsV1 before = trainer.Evaluate(batch);
            Cb2VecTrainingMetricsV1 trained = trainer.TrainBatch(batch);
            if (!(before.MeanLoss >= 0.0f) || trained.OptimizerStep != 1)
                return 2;

            for (int epoch = 0; epoch < 3; ++epoch)
                trainer.TrainEpoch(batch);

            Cb2VecQuantizationConfigV1 quantization = Cb2VecNative.DefaultQuantization();
            artifactV1 = trainer.WriteArtifact(quantization, new byte[32]);
            artifactV2 = trainer.WriteArtifactV2(quantization, Repeat(0x5A, 32), schema);

            int declared = trainer.CheckpointLength();
            checkpoint = trainer.WriteCheckpoint();
            if (checkpoint.Length != declared)
                return 3;

            // A resumed trainer must track an uninterrupted one exactly, with
            // shuffling on so the permutation stream is exercised too.
            using (Cb2VecTrainer resumed = Cb2VecTrainer.LoadCheckpoint(checkpoint))
            {
                for (int epoch = 0; epoch < 3; ++epoch)
                {
                    Cb2VecTrainingMetricsV1 original = trainer.TrainEpoch(batch);
                    Cb2VecTrainingMetricsV1 restored = resumed.TrainEpoch(batch);
                    if (!BitsEqual(original.MeanLoss, restored.MeanLoss) ||
                        original.OptimizerStep != restored.OptimizerStep ||
                        original.CompletedEpochs != restored.CompletedEpochs)
                        return 4;
                }
            }
        }

        // A corrupted checkpoint must be refused, not silently accepted.
        byte[] corrupted = (byte[])checkpoint.Clone();
        corrupted[corrupted.Length - 1] ^= 0x01;
        try
        {
            using (Cb2VecTrainer.LoadCheckpoint(corrupted))
                return 5;
        }
        catch (Cb2VecException error)
        {
            if (error.Status != Cb2VecNative.ErrorCheckpoint)
                return 6;
        }

        // A version-2 artifact answers the inference recipe itself.
        Cb2VecArtifactInfoV1 probe = Cb2VecNative.ProbeArtifact(artifactV2);
        if (probe.ArtifactVersion != Cb2VecNative.ArtifactVersionV2 ||
            probe.HasInferenceConfig != 1 ||
            probe.Activation != (uint)Cb2VecActivation.Relu ||
            probe.Pooling != (uint)Cb2VecPooling.Mean ||
            probe.SchemaVersion != 7 ||
            probe.TokenCount != shape.TokenCount)
            return 7;
        if (Cb2VecNative.ProbeArtifact(artifactV1).HasInferenceConfig != 0)
            return 8;

        // A conflicting recipe must be refused rather than silently applied.
        Cb2VecInferenceConfigV1 wrong = Cb2VecNative.DefaultInference();
        wrong.Activation = (uint)Cb2VecActivation.Identity;
        wrong.Pooling = (uint)Cb2VecPooling.Sum;
        try
        {
            using (Cb2VecModel.Load(artifactV2, wrong))
                return 9;
        }
        catch (Cb2VecException error)
        {
            if (error.Status != Cb2VecNative.ErrorArtifact)
                return 10;
        }

        _artifactCache = artifactV2;

        int result;
        using (Cb2VecModel model = Cb2VecModel.Load(artifactV2, schema))
        {
            Cb2VecModelInfoV1 info = model.GetInfo();
            if (info.ArtifactVersion != Cb2VecNative.ArtifactVersionV2 ||
                info.Activation != (uint)Cb2VecActivation.Relu ||
                model.GetMetadata().SchemaVersion != 7)
                return 11;

            result = RunSessionChecks(model);
            if (result != 0)
                return result;

            result = RunAllocationChecks(model);
            if (result != 0)
                return result;

            result = RunDisposeOrderChecks(model);
            if (result != 0)
                return result;
        }

        Console.WriteLine(
            "CB2Vec " + Cb2VecNative.LibraryVersion + " C ABI 0x" +
            Cb2VecNative.AbiVersion.ToString("X8") +
            " C# session/checkpoint/artifact-v2 smoke passed.");
        return 0;
    }

    /// <summary>Incremental scores must match full rebuilds bit for bit.</summary>
    private static int RunSessionChecks(Cb2VecModel model)
    {
        Cb2VecSessionConfigV1 sessionConfig = SessionConfig();

        using (Cb2VecSession session = model.CreateSession(sessionConfig))
        using (var frame = new Cb2VecPinnedBuffer<Cb2VecTokenDeltaV1>(4))
        {
            // Scoring before a reset is a state error, not a crash.
            try
            {
                session.Predict();
                return 20;
            }
            catch (Cb2VecException error)
            {
                if (error.Status != Cb2VecNative.ErrorState)
                    return 21;
            }

            session.Reset(Input(Tokens));
            if (!BitsEqual(session.Predict(), model.Predict(Input(Tokens))))
                return 22;

            Cb2VecSessionInfoV1 info = session.GetInfo();
            if (info.SiteCount != 5 || info.TokenSlots != 9 || info.Depth != 0)
                return 23;

            // One move touching two sites at once.
            frame.Items[0] = new Cb2VecTokenDeltaV1(0, 0, 0, 6);
            frame.Items[1] = new Cb2VecTokenDeltaV1(2, 1, 7, 0);
            session.Push(frame, 2);

            ushort[] mutated = (ushort[])Tokens.Clone();
            mutated[0] = 6;
            mutated[6] = 0;
            if (!BitsEqual(session.Predict(), model.Predict(Input(mutated))))
                return 24;

            // A delta with the wrong expected old token must change nothing.
            frame.Items[0] = new Cb2VecTokenDeltaV1(0, 0, 99, 1);
            try
            {
                session.Push(frame, 1);
                return 25;
            }
            catch (Cb2VecException error)
            {
                if (error.Status != Cb2VecNative.ErrorInvalidArgument)
                    return 26;
            }
            if (session.GetInfo().Depth != 1 ||
                !BitsEqual(session.Predict(), model.Predict(Input(mutated))))
                return 27;

            // Exceeding the frame limit reports distinctly and changes nothing.
            try
            {
                session.Push(new[]
                {
                    new Cb2VecTokenDeltaV1(0, 0, 6, 1),
                    new Cb2VecTokenDeltaV1(0, 1, 3, 1),
                    new Cb2VecTokenDeltaV1(0, 2, 5, 1),
                    new Cb2VecTokenDeltaV1(1, 0, 1, 2),
                    new Cb2VecTokenDeltaV1(1, 1, 8, 2),
                }, 5);
                return 28;
            }
            catch (Cb2VecException error)
            {
                if (error.Status != Cb2VecNative.ErrorLimitExceeded)
                    return 29;
            }

            if (session.Pop() != 2)
                return 30;
            if (!BitsEqual(session.Predict(), model.Predict(Input(Tokens))))
                return 31;

            // Popping an empty stack reports state rather than underflowing.
            try
            {
                session.Pop();
                return 32;
            }
            catch (Cb2VecException error)
            {
                if (error.Status != Cb2VecNative.ErrorState)
                    return 33;
            }

            // A long random push/pop sequence must restore the start exactly.
            // Site 3 owns one lane, so its token is tracked with a small stack.
            int expectedBits = BitConverter.SingleToInt32Bits(session.Predict());
            var random = new Random(20260730);
            var history = new ushort[sessionConfig.MaxDepth + 1];
            int depth = 0;
            history[0] = Tokens[7];
            for (int step = 0; step < 5000; ++step)
            {
                bool pop = depth > 0 &&
                    (depth == (int)sessionConfig.MaxDepth || random.Next(100) < 45);
                if (pop)
                {
                    if (session.Pop() != 1)
                        return 34;
                    depth -= 1;
                }
                else
                {
                    var replacement = (ushort)random.Next(9);
                    frame.Items[0] = new Cb2VecTokenDeltaV1(3, 0, history[depth], replacement);
                    session.Push(frame, 1);
                    depth += 1;
                    history[depth] = replacement;
                }
                if (step % 7 == 0)
                    session.Predict();
            }
            while (depth > 0)
            {
                session.Pop();
                depth -= 1;
            }
            if (BitConverter.SingleToInt32Bits(session.Predict()) != expectedBits)
                return 35;
            if (!BitsEqual(session.Predict(), model.Predict(Input(Tokens))))
                return 36;
        }
        return 0;
    }

    /// <summary>
    /// The search loop and the pinned whole-input path must not allocate.
    /// </summary>
    private static int RunAllocationChecks(Cb2VecModel model)
    {
        var batch = new Cb2VecTrainingBatch(
            Tokens, SiteOffsets, SiteGroups, new uint[] { 0, 5 }, new float[1]);

        using (Cb2VecSession session = model.CreateSession(SessionConfig()))
        using (var frame = new Cb2VecPinnedBuffer<Cb2VecTokenDeltaV1>(2))
        using (var pinnedInput = new Cb2VecPinnedInput(Input(Tokens)))
        using (var pinnedBatch = new Cb2VecPinnedBatch(batch))
        using (var scores = new Cb2VecPinnedBuffer<float>(1))
        {
            session.Reset(pinnedInput);
            frame.Items[0] = new Cb2VecTokenDeltaV1(0, 0, 0, 4);
            frame.Items[1] = new Cb2VecTokenDeltaV1(2, 0, 2, 7);

            // Warm up: JIT the P/Invoke stubs and touch every lazy path.
            for (int i = 0; i < 64; ++i)
            {
                session.Push(frame, 2);
                session.Predict();
                session.Pop();
                model.PredictInto(pinnedInput);
                model.PredictBatchInto(pinnedBatch, scores);
            }

            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 5000; ++i)
            {
                session.Push(frame, 2);
                session.Predict();
                session.Pop();
            }
            long searchLoop = GC.GetAllocatedBytesForCurrentThread() - before;
            if (searchLoop != 0)
            {
                Console.Error.WriteLine(
                    "session search loop allocated " + searchLoop + " bytes");
                return 40;
            }

            before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 5000; ++i)
            {
                model.PredictInto(pinnedInput);
                model.PredictBatchInto(pinnedBatch, scores);
            }
            long wholeInput = GC.GetAllocatedBytesForCurrentThread() - before;
            if (wholeInput != 0)
            {
                Console.Error.WriteLine(
                    "pinned whole-input inference allocated " + wholeInput + " bytes");
                return 41;
            }

            // The reusable path must agree with the convenience path.
            float reference = model.Predict(Input(Tokens));
            if (!BitsEqual(model.PredictInto(pinnedInput), reference) ||
                !BitsEqual(scores.Items[0], reference))
                return 42;
        }
        return 0;
    }

    /// <summary>
    /// SafeHandle lifetime: many sessions share one model, a session outlives
    /// its model, disposal order does not matter, double dispose is a no-op,
    /// and use-after-dispose throws instead of touching freed native memory.
    /// </summary>
    private static int RunDisposeOrderChecks(Cb2VecModel sharedModel)
    {
        Cb2VecSessionConfigV1 sessionConfig = SessionConfig();

        var sessions = new Cb2VecSession[4];
        try
        {
            for (int index = 0; index < sessions.Length; ++index)
            {
                sessions[index] = sharedModel.CreateSession(sessionConfig);
                sessions[index].Reset(Input(Tokens));
                sessions[index].Push(
                    new[] { new Cb2VecTokenDeltaV1(0, 0, 0, (ushort)(index + 1)) }, 1);
            }
            for (int index = 0; index < sessions.Length; ++index)
            {
                ushort[] expected = (ushort[])Tokens.Clone();
                expected[0] = (ushort)(index + 1);
                if (!BitsEqual(sessions[index].Predict(), sharedModel.Predict(Input(expected))))
                    return 50;
            }
        }
        finally
        {
            foreach (Cb2VecSession session in sessions)
                if (session != null)
                    session.Dispose();
        }

        // Dispose the model first: the session keeps the weights alive.
        Cb2VecSession orphan;
        Cb2VecModel model = Cb2VecModel.Load(_artifactCache);
        try
        {
            orphan = model.CreateSession(sessionConfig);
            orphan.Reset(Input(Tokens));
        }
        finally
        {
            model.Dispose();
            model.Dispose(); // Double dispose must be a no-op.
        }

        if (!BitsEqual(orphan.Predict(), sharedModel.Predict(Input(Tokens))))
            return 51;

        orphan.Dispose();
        orphan.Dispose(); // Double dispose must be a no-op.

        // Using a disposed session must throw, not fault.
        try
        {
            orphan.Predict();
            return 52;
        }
        catch (ObjectDisposedException)
        {
        }
        try
        {
            orphan.Reset(Input(Tokens));
            return 53;
        }
        catch (ObjectDisposedException)
        {
        }
        try
        {
            orphan.GetInfo();
            return 54;
        }
        catch (ObjectDisposedException)
        {
        }
        try
        {
            orphan.Pop();
            return 55;
        }
        catch (ObjectDisposedException)
        {
        }
        return 0;
    }

    private static byte[] Repeat(byte value, int length)
    {
        var bytes = new byte[length];
        for (int index = 0; index < length; ++index)
            bytes[index] = value;
        return bytes;
    }
}
