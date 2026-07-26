using System;
using CB2Vec;

internal static class Program
{
    private static int Main()
    {
        Cb2VecModelShapeV1 shape = Cb2VecNative.DefaultShape();
        shape.TokenCount = 4;
        shape.GroupCount = 2;
        shape.Dim = 3;
        shape.FmRank = 2;

        Cb2VecTrainerConfigV1 config = Cb2VecNative.DefaultTrainerConfig();
        config.Pooling = (uint)Cb2VecPooling.Mean;
        config.Shuffle = 0;
        config.BatchSize = 2;

        var batch = new Cb2VecTrainingBatch(
            new ushort[] { 0, 0, 1, 2, 3, 1 },
            new uint[] { 0, 2, 3, 3, 4, 6 },
            new uint[] { 0, 0, 1, 0, 1 },
            new uint[] { 0, 3, 5 },
            new float[] { 0.2f, 0.8f },
            new float[] { 1.0f, 2.0f });
        var input = new Cb2VecInput(
            new ushort[] { 0, 0, 1 },
            new uint[] { 0, 2, 3, 3 },
            new uint[] { 0, 0, 1 });

        byte[] artifact;
        float directScore;
        using (Cb2VecTrainer trainer = Cb2VecTrainer.Create(shape, config))
        {
            Cb2VecTrainingMetricsV1 before = trainer.Evaluate(batch);
            Cb2VecTrainingMetricsV1 trained = trainer.TrainBatch(batch);
            if (!(before.MeanLoss >= 0.0f) || trained.OptimizerStep != 1)
                return 1;

            Cb2VecQuantizationConfigV1 quantization =
                Cb2VecNative.DefaultQuantization();
            using (Cb2VecModel model = trainer.Quantize(quantization))
                directScore = model.Predict(input);
            artifact = trainer.WriteArtifact(quantization, new byte[32]);
        }

        Cb2VecInferenceConfigV1 inference = Cb2VecNative.DefaultInference();
        inference.Pooling = (uint)Cb2VecPooling.Mean;
        using (Cb2VecModel loaded = Cb2VecModel.Load(artifact, inference))
        {
            float loadedScore = loaded.Predict(input);
            if (BitConverter.SingleToInt32Bits(directScore) !=
                BitConverter.SingleToInt32Bits(loadedScore))
                return 2;
        }

        Console.WriteLine(
            "CB2Vec " + Cb2VecNative.LibraryVersion +
            " C# trainer/PTQ/load smoke passed.");
        return 0;
    }
}
