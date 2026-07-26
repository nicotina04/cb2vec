// CB2Vec 0.2.1 / C ABI 1.0 Unity binding.
// Copy this file into the Unity project and place the matching native library
// in Assets/Plugins for the target platform.

using System;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

namespace CB2Vec
{
    public enum Cb2VecActivation : uint
    {
        Identity = 0,
        Relu = 1,
    }

    public enum Cb2VecPooling : uint
    {
        Sum = 0,
        Mean = 1,
    }

    public enum Cb2VecLoss : uint
    {
        BinaryCrossEntropyWithLogits = 0,
        MeanSquaredError = 1,
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Cb2VecModelShapeV1
    {
        public uint StructSize;
        public uint AbiVersion;
        public uint TokenCount;
        public uint GroupCount;
        public uint Dim;
        public uint FmRank;
        public uint Reserved0;
        public uint Reserved1;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Cb2VecTrainerConfigV1
    {
        public uint StructSize;
        public uint AbiVersion;
        public uint Activation;
        public uint Pooling;
        public uint Loss;
        public uint BatchSize;
        public uint Shuffle;
        public uint Flags;
        public ulong Seed;
        public float LearningRate;
        public float Beta1;
        public float Beta2;
        public float Epsilon;
        public uint Reserved0;
        public uint Reserved1;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Cb2VecQuantizationConfigV1
    {
        public uint StructSize;
        public uint AbiVersion;
        public int EmbeddingScale;
        public int HeadScale;
        public int FactorScale;
        public uint Flags;
        public uint Reserved0;
        public uint Reserved1;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Cb2VecInferenceConfigV1
    {
        public uint StructSize;
        public uint Activation;
        public uint Pooling;
        public uint Flags;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct Cb2VecTrainingBatchViewV1
    {
        internal uint StructSize;
        internal uint Flags;
        internal IntPtr Tokens;
        internal IntPtr SiteTokenOffsets;
        internal IntPtr SiteGroups;
        internal IntPtr SampleSiteOffsets;
        internal IntPtr Targets;
        internal IntPtr Weights;
        internal uint TokensLength;
        internal uint SiteCount;
        internal uint SampleCount;
        internal uint Reserved;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Cb2VecTrainingMetricsV1
    {
        public uint StructSize;
        public uint AbiVersion;
        public float MeanLoss;
        public uint Reserved;
        public double TotalWeight;
        public ulong SampleCount;
        public ulong BatchCount;
        public ulong OptimizerStep;
        public ulong CompletedEpochs;
        public ulong ReservedTail;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Cb2VecModelInfoV1
    {
        public uint StructSize;
        public uint AbiVersion;
        public uint ArtifactVersion;
        public uint Flags;
        public uint TokenCount;
        public uint GroupCount;
        public uint Dim;
        public uint FmRank;
        public uint Kind;
        public uint Activation;
        public uint Pooling;
        public int EmbeddingScale;
        public int HeadScale;
        public int FactorScale;
        public uint Reserved0;
        public uint Reserved1;
    }

    public sealed class Cb2VecException : Exception
    {
        public int Status { get; private set; }

        internal Cb2VecException(int status, string message)
            : base("CB2Vec status " + status + ": " + message)
        {
            Status = status;
        }
    }

    public sealed class Cb2VecInput
    {
        public ushort[] Tokens { get; private set; }
        public uint[] SiteTokenOffsets { get; private set; }
        public uint[] SiteGroups { get; private set; }

        public Cb2VecInput(
            ushort[] tokens,
            uint[] siteTokenOffsets,
            uint[] siteGroups)
        {
            if (tokens == null)
                throw new ArgumentNullException("tokens");
            if (siteTokenOffsets == null)
                throw new ArgumentNullException("siteTokenOffsets");
            if (siteGroups == null)
                throw new ArgumentNullException("siteGroups");
            if (siteTokenOffsets.Length != siteGroups.Length + 1)
                throw new ArgumentException(
                    "siteTokenOffsets must contain siteGroups.Length + 1 entries.");
            if (siteTokenOffsets.Length == 0 || siteTokenOffsets[0] != 0)
                throw new ArgumentException("siteTokenOffsets must start at zero.");
            if (siteTokenOffsets[siteTokenOffsets.Length - 1] != (uint)tokens.Length)
                throw new ArgumentException("siteTokenOffsets must end at tokens.Length.");
            for (int i = 1; i < siteTokenOffsets.Length; ++i)
            {
                if (siteTokenOffsets[i - 1] > siteTokenOffsets[i])
                    throw new ArgumentException("siteTokenOffsets must be monotonic.");
            }
            Tokens = tokens;
            SiteTokenOffsets = siteTokenOffsets;
            SiteGroups = siteGroups;
        }
    }

    public sealed class Cb2VecTrainingBatch
    {
        public ushort[] Tokens { get; private set; }
        public uint[] SiteTokenOffsets { get; private set; }
        public uint[] SiteGroups { get; private set; }
        public uint[] SampleSiteOffsets { get; private set; }
        public float[] Targets { get; private set; }
        public float[] Weights { get; private set; }

        public uint SampleCount
        {
            get { return checked((uint)Targets.Length); }
        }

        public Cb2VecTrainingBatch(
            ushort[] tokens,
            uint[] siteTokenOffsets,
            uint[] siteGroups,
            uint[] sampleSiteOffsets,
            float[] targets,
            float[] weights = null)
        {
            if (tokens == null)
                throw new ArgumentNullException("tokens");
            if (siteTokenOffsets == null)
                throw new ArgumentNullException("siteTokenOffsets");
            if (siteGroups == null)
                throw new ArgumentNullException("siteGroups");
            if (sampleSiteOffsets == null)
                throw new ArgumentNullException("sampleSiteOffsets");
            if (targets == null)
                throw new ArgumentNullException("targets");
            if (targets.Length == 0)
                throw new ArgumentException("A training batch must contain a sample.");
            if (siteTokenOffsets.Length != siteGroups.Length + 1)
                throw new ArgumentException(
                    "siteTokenOffsets must contain siteGroups.Length + 1 entries.");
            if (sampleSiteOffsets.Length != targets.Length + 1)
                throw new ArgumentException(
                    "sampleSiteOffsets must contain targets.Length + 1 entries.");
            if (siteTokenOffsets[0] != 0 ||
                siteTokenOffsets[siteTokenOffsets.Length - 1] != (uint)tokens.Length)
                throw new ArgumentException(
                    "siteTokenOffsets must start at zero and end at tokens.Length.");
            if (sampleSiteOffsets[0] != 0 ||
                sampleSiteOffsets[sampleSiteOffsets.Length - 1] != (uint)siteGroups.Length)
                throw new ArgumentException(
                    "sampleSiteOffsets must start at zero and end at siteGroups.Length.");
            for (int i = 1; i < siteTokenOffsets.Length; ++i)
            {
                if (siteTokenOffsets[i - 1] > siteTokenOffsets[i])
                    throw new ArgumentException("siteTokenOffsets must be monotonic.");
            }
            for (int i = 1; i < sampleSiteOffsets.Length; ++i)
            {
                if (sampleSiteOffsets[i - 1] > sampleSiteOffsets[i])
                    throw new ArgumentException("sampleSiteOffsets must be monotonic.");
            }
            if (weights != null && weights.Length != 0 && weights.Length != targets.Length)
                throw new ArgumentException(
                    "weights must be empty or contain one value per target.");

            Tokens = tokens;
            SiteTokenOffsets = siteTokenOffsets;
            SiteGroups = siteGroups;
            SampleSiteOffsets = sampleSiteOffsets;
            Targets = targets;
            Weights = weights ?? new float[0];
        }
    }

    internal sealed class PinnedArray : IDisposable
    {
        private GCHandle _handle;
        private bool _allocated;
        internal IntPtr Pointer { get; private set; }

        internal PinnedArray(Array array)
        {
            if (array == null || array.Length == 0)
            {
                Pointer = IntPtr.Zero;
                return;
            }
            _handle = GCHandle.Alloc(array, GCHandleType.Pinned);
            _allocated = true;
            Pointer = _handle.AddrOfPinnedObject();
        }

        public void Dispose()
        {
            if (_allocated)
            {
                _handle.Free();
                _allocated = false;
                Pointer = IntPtr.Zero;
            }
        }
    }

    internal sealed class PinnedInput : IDisposable
    {
        private readonly PinnedArray _tokens;
        private readonly PinnedArray _offsets;
        private readonly PinnedArray _groups;

        internal IntPtr Tokens { get { return _tokens.Pointer; } }
        internal IntPtr Offsets { get { return _offsets.Pointer; } }
        internal IntPtr Groups { get { return _groups.Pointer; } }

        internal PinnedInput(Cb2VecInput input)
        {
            _tokens = new PinnedArray(input.Tokens);
            _offsets = new PinnedArray(input.SiteTokenOffsets);
            _groups = new PinnedArray(input.SiteGroups);
        }

        public void Dispose()
        {
            _groups.Dispose();
            _offsets.Dispose();
            _tokens.Dispose();
        }
    }

    internal sealed class PinnedBatch : IDisposable
    {
        private readonly PinnedArray _tokens;
        private readonly PinnedArray _siteOffsets;
        private readonly PinnedArray _siteGroups;
        private readonly PinnedArray _sampleOffsets;
        private readonly PinnedArray _targets;
        private readonly PinnedArray _weights;

        internal Cb2VecTrainingBatchViewV1 View;

        internal PinnedBatch(Cb2VecTrainingBatch batch)
        {
            _tokens = new PinnedArray(batch.Tokens);
            _siteOffsets = new PinnedArray(batch.SiteTokenOffsets);
            _siteGroups = new PinnedArray(batch.SiteGroups);
            _sampleOffsets = new PinnedArray(batch.SampleSiteOffsets);
            _targets = new PinnedArray(batch.Targets);
            _weights = new PinnedArray(batch.Weights);
            View = new Cb2VecTrainingBatchViewV1
            {
                StructSize = checked((uint)Marshal.SizeOf(typeof(Cb2VecTrainingBatchViewV1))),
                Flags = 0,
                Tokens = _tokens.Pointer,
                SiteTokenOffsets = _siteOffsets.Pointer,
                SiteGroups = _siteGroups.Pointer,
                SampleSiteOffsets = _sampleOffsets.Pointer,
                Targets = _targets.Pointer,
                Weights = _weights.Pointer,
                TokensLength = checked((uint)batch.Tokens.Length),
                SiteCount = checked((uint)batch.SiteGroups.Length),
                SampleCount = batch.SampleCount,
                Reserved = 0,
            };
        }

        public void Dispose()
        {
            _weights.Dispose();
            _targets.Dispose();
            _sampleOffsets.Dispose();
            _siteGroups.Dispose();
            _siteOffsets.Dispose();
            _tokens.Dispose();
        }
    }

    internal sealed class Cb2VecTrainerHandle : SafeHandleZeroOrMinusOneIsInvalid
    {
        public Cb2VecTrainerHandle() : base(true) { }

        protected override bool ReleaseHandle()
        {
            return NativeMethods.cb2vec_trainer_free_v1(handle) == Cb2VecNative.Ok;
        }
    }

    internal sealed class Cb2VecModelHandle : SafeHandleZeroOrMinusOneIsInvalid
    {
        public Cb2VecModelHandle() : base(true) { }

        protected override bool ReleaseHandle()
        {
            return NativeMethods.cb2vec_model_free_v1(handle) == Cb2VecNative.Ok;
        }
    }

    public static class Cb2VecNative
    {
        public const uint AbiVersion = 0x00010000;
        public const int Ok = 0;
        public const int ErrorBufferTooSmall = -7;

        static Cb2VecNative()
        {
            ValidateLayouts();
            uint native = NativeMethods.cb2vec_abi_version();
            uint nativeMajor = native >> 16;
            if (nativeMajor != 1)
                throw new DllNotFoundException(
                    "Incompatible CB2Vec ABI 0x" + native.ToString("X8") + ".");
        }

        public static void EnsureCompatible()
        {
            // Calling this method runs the static constructor.
        }

        public static string LibraryVersion
        {
            get
            {
                EnsureCompatible();
                return Marshal.PtrToStringAnsi(NativeMethods.cb2vec_library_version()) ?? "";
            }
        }

        public static Cb2VecModelShapeV1 DefaultShape()
        {
            EnsureCompatible();
            Cb2VecModelShapeV1 value;
            Check(NativeMethods.cb2vec_model_shape_default_v1(out value));
            return value;
        }

        public static Cb2VecTrainerConfigV1 DefaultTrainerConfig()
        {
            EnsureCompatible();
            Cb2VecTrainerConfigV1 value;
            Check(NativeMethods.cb2vec_trainer_config_default_v1(out value));
            return value;
        }

        public static Cb2VecQuantizationConfigV1 DefaultQuantization()
        {
            EnsureCompatible();
            Cb2VecQuantizationConfigV1 value;
            Check(NativeMethods.cb2vec_quantization_config_default_v1(out value));
            return value;
        }

        public static Cb2VecInferenceConfigV1 DefaultInference()
        {
            EnsureCompatible();
            Cb2VecInferenceConfigV1 value;
            Check(NativeMethods.cb2vec_inference_config_default_v1(out value));
            return value;
        }

        internal static void Check(int status)
        {
            if (status == Ok)
                return;
            IntPtr pointer = NativeMethods.cb2vec_last_error();
            string message = pointer == IntPtr.Zero
                ? "unknown native error"
                : (Marshal.PtrToStringAnsi(pointer) ?? "unknown native error");
            throw new Cb2VecException(status, message);
        }

        private static void ValidateLayouts()
        {
            RequireSize(typeof(Cb2VecModelShapeV1), 32);
            RequireSize(typeof(Cb2VecTrainerConfigV1), 64);
            RequireSize(typeof(Cb2VecQuantizationConfigV1), 32);
            RequireSize(typeof(Cb2VecInferenceConfigV1), 16);
            RequireSize(typeof(Cb2VecTrainingMetricsV1), 64);
            RequireSize(typeof(Cb2VecModelInfoV1), 64);
            RequireSize(typeof(Cb2VecTrainingBatchViewV1), IntPtr.Size == 8 ? 72 : 48);
            RequireOffset(typeof(Cb2VecTrainerConfigV1), "Seed", 32);
            RequireOffset(typeof(Cb2VecTrainerConfigV1), "LearningRate", 40);
            RequireOffset(typeof(Cb2VecTrainingMetricsV1), "TotalWeight", 16);
            RequireOffset(typeof(Cb2VecTrainingMetricsV1), "SampleCount", 24);
            RequireOffset(typeof(Cb2VecModelInfoV1), "FactorScale", 52);
        }

        private static void RequireSize(Type type, int expected)
        {
            int actual = Marshal.SizeOf(type);
            if (actual != expected)
                throw new TypeLoadException(
                    type.Name + " is " + actual + " bytes; expected " + expected + ".");
        }

        private static void RequireOffset(Type type, string field, int expected)
        {
            int actual = checked((int)Marshal.OffsetOf(type, field));
            if (actual != expected)
                throw new TypeLoadException(
                    type.Name + "." + field + " offset is " + actual +
                    "; expected " + expected + ".");
        }
    }

    public sealed class Cb2VecTrainer : IDisposable
    {
        private Cb2VecTrainerHandle _handle;

        private Cb2VecTrainer(Cb2VecTrainerHandle handle)
        {
            _handle = handle;
        }

        public static Cb2VecTrainer Create(
            Cb2VecModelShapeV1 shape,
            Cb2VecTrainerConfigV1 config)
        {
            Cb2VecNative.EnsureCompatible();
            Cb2VecTrainerHandle handle;
            Cb2VecNative.Check(
                NativeMethods.cb2vec_trainer_create_v1(ref shape, ref config, out handle));
            return new Cb2VecTrainer(handle);
        }

        public static Cb2VecTrainer LoadArtifact(
            byte[] artifact,
            Cb2VecTrainerConfigV1 config)
        {
            if (artifact == null)
                throw new ArgumentNullException("artifact");
            Cb2VecNative.EnsureCompatible();
            using (PinnedArray pinned = new PinnedArray(artifact))
            {
                Cb2VecTrainerHandle handle;
                Cb2VecNative.Check(
                    NativeMethods.cb2vec_trainer_load_artifact_v1(
                        pinned.Pointer,
                        checked((uint)artifact.Length),
                        ref config,
                        out handle));
                return new Cb2VecTrainer(handle);
            }
        }

        public Cb2VecModelInfoV1 GetInfo()
        {
            Cb2VecModelInfoV1 info;
            Cb2VecNative.Check(
                NativeMethods.cb2vec_trainer_get_info_v1(_handle, out info));
            return info;
        }

        public float PredictLogit(Cb2VecInput input)
        {
            if (input == null)
                throw new ArgumentNullException("input");
            using (PinnedInput pinned = new PinnedInput(input))
            {
                float score;
                Cb2VecNative.Check(
                    NativeMethods.cb2vec_trainer_predict_logit_v1(
                        _handle,
                        pinned.Tokens,
                        checked((uint)input.Tokens.Length),
                        pinned.Offsets,
                        pinned.Groups,
                        checked((uint)input.SiteGroups.Length),
                        out score));
                return score;
            }
        }

        public float PredictProbability(Cb2VecInput input)
        {
            if (input == null)
                throw new ArgumentNullException("input");
            using (PinnedInput pinned = new PinnedInput(input))
            {
                float probability;
                Cb2VecNative.Check(
                    NativeMethods.cb2vec_trainer_predict_probability_v1(
                        _handle,
                        pinned.Tokens,
                        checked((uint)input.Tokens.Length),
                        pinned.Offsets,
                        pinned.Groups,
                        checked((uint)input.SiteGroups.Length),
                        out probability));
                return probability;
            }
        }

        public Cb2VecTrainingMetricsV1 Evaluate(Cb2VecTrainingBatch batch)
        {
            return RunDataset(batch, NativeMethods.cb2vec_trainer_evaluate_v1);
        }

        public Cb2VecTrainingMetricsV1 TrainBatch(Cb2VecTrainingBatch batch)
        {
            return RunDataset(batch, NativeMethods.cb2vec_trainer_train_batch_v1);
        }

        public Cb2VecTrainingMetricsV1 TrainEpoch(Cb2VecTrainingBatch dataset)
        {
            return RunDataset(dataset, NativeMethods.cb2vec_trainer_train_epoch_v1);
        }

        public Cb2VecModel Quantize(Cb2VecQuantizationConfigV1 quantization)
        {
            Cb2VecModelHandle model;
            Cb2VecNative.Check(
                NativeMethods.cb2vec_trainer_quantize_v1(
                    _handle, ref quantization, out model));
            return new Cb2VecModel(model);
        }

        public byte[] WriteArtifact(
            Cb2VecQuantizationConfigV1 quantization,
            byte[] sourceSha256)
        {
            if (sourceSha256 == null)
                throw new ArgumentNullException("sourceSha256");
            if (sourceSha256.Length != 32)
                throw new ArgumentException("sourceSha256 must contain exactly 32 bytes.");

            using (PinnedArray digest = new PinnedArray(sourceSha256))
            {
                uint required;
                int probe = NativeMethods.cb2vec_trainer_write_artifact_v1(
                    _handle,
                    ref quantization,
                    digest.Pointer,
                    IntPtr.Zero,
                    0,
                    out required);
                if (probe != Cb2VecNative.ErrorBufferTooSmall)
                    Cb2VecNative.Check(probe);

                byte[] artifact = new byte[checked((int)required)];
                using (PinnedArray output = new PinnedArray(artifact))
                {
                    uint written;
                    Cb2VecNative.Check(
                        NativeMethods.cb2vec_trainer_write_artifact_v1(
                            _handle,
                            ref quantization,
                            digest.Pointer,
                            output.Pointer,
                            required,
                            out written));
                    if (written != required)
                        throw new InvalidOperationException(
                            "CB2Vec artifact byte count changed between calls.");
                }
                return artifact;
            }
        }

        private delegate int DatasetCall(
            Cb2VecTrainerHandle trainer,
            ref Cb2VecTrainingBatchViewV1 batch,
            out Cb2VecTrainingMetricsV1 metrics);

        private Cb2VecTrainingMetricsV1 RunDataset(
            Cb2VecTrainingBatch batch,
            DatasetCall call)
        {
            if (batch == null)
                throw new ArgumentNullException("batch");
            using (PinnedBatch pinned = new PinnedBatch(batch))
            {
                Cb2VecTrainingMetricsV1 metrics;
                Cb2VecNative.Check(call(_handle, ref pinned.View, out metrics));
                return metrics;
            }
        }

        public void Dispose()
        {
            if (_handle != null)
            {
                _handle.Dispose();
                _handle = null;
            }
            GC.SuppressFinalize(this);
        }
    }

    public sealed class Cb2VecModel : IDisposable
    {
        private Cb2VecModelHandle _handle;

        internal Cb2VecModel(Cb2VecModelHandle handle)
        {
            _handle = handle;
        }

        public static Cb2VecModel Load(
            byte[] artifact,
            Cb2VecInferenceConfigV1 inference)
        {
            if (artifact == null)
                throw new ArgumentNullException("artifact");
            Cb2VecNative.EnsureCompatible();
            using (PinnedArray pinned = new PinnedArray(artifact))
            {
                Cb2VecModelHandle handle;
                Cb2VecNative.Check(
                    NativeMethods.cb2vec_model_load_v1(
                        pinned.Pointer,
                        checked((uint)artifact.Length),
                        ref inference,
                        out handle));
                return new Cb2VecModel(handle);
            }
        }

        public Cb2VecModelInfoV1 GetInfo()
        {
            Cb2VecModelInfoV1 info;
            Cb2VecNative.Check(
                NativeMethods.cb2vec_model_get_info_v1(_handle, out info));
            return info;
        }

        public float Predict(Cb2VecInput input)
        {
            if (input == null)
                throw new ArgumentNullException("input");
            using (PinnedInput pinned = new PinnedInput(input))
            {
                float score;
                Cb2VecNative.Check(
                    NativeMethods.cb2vec_model_predict_v1(
                        _handle,
                        pinned.Tokens,
                        checked((uint)input.Tokens.Length),
                        pinned.Offsets,
                        pinned.Groups,
                        checked((uint)input.SiteGroups.Length),
                        out score));
                return score;
            }
        }

        public float[] PredictBatch(Cb2VecTrainingBatch batch)
        {
            if (batch == null)
                throw new ArgumentNullException("batch");
            float[] scores = new float[checked((int)batch.SampleCount)];
            using (PinnedBatch pinned = new PinnedBatch(batch))
            using (PinnedArray output = new PinnedArray(scores))
            {
                Cb2VecNative.Check(
                    NativeMethods.cb2vec_model_predict_batch_v1(
                        _handle,
                        ref pinned.View,
                        output.Pointer,
                        checked((uint)scores.Length)));
            }
            return scores;
        }

        public float[] PredictBatch(
            ushort[] tokens,
            uint[] siteTokenOffsets,
            uint[] siteGroups,
            uint[] sampleSiteOffsets)
        {
            if (sampleSiteOffsets == null)
                throw new ArgumentNullException("sampleSiteOffsets");
            if (sampleSiteOffsets.Length < 2)
                throw new ArgumentException(
                    "sampleSiteOffsets must delimit at least one sample.");
            var ignoredTargets = new float[sampleSiteOffsets.Length - 1];
            return PredictBatch(
                new Cb2VecTrainingBatch(
                    tokens,
                    siteTokenOffsets,
                    siteGroups,
                    sampleSiteOffsets,
                    ignoredTargets));
        }

        public void Dispose()
        {
            if (_handle != null)
            {
                _handle.Dispose();
                _handle = null;
            }
            GC.SuppressFinalize(this);
        }
    }

    internal static class NativeMethods
    {
        private const string Library = "cb2vec";
        private const CallingConvention Cdecl = CallingConvention.Cdecl;

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern uint cb2vec_abi_version();

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern IntPtr cb2vec_library_version();

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern IntPtr cb2vec_last_error();

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_model_shape_default_v1(
            out Cb2VecModelShapeV1 shape);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_trainer_config_default_v1(
            out Cb2VecTrainerConfigV1 config);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_quantization_config_default_v1(
            out Cb2VecQuantizationConfigV1 config);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_inference_config_default_v1(
            out Cb2VecInferenceConfigV1 config);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_trainer_create_v1(
            ref Cb2VecModelShapeV1 shape,
            ref Cb2VecTrainerConfigV1 config,
            out Cb2VecTrainerHandle trainer);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_trainer_load_artifact_v1(
            IntPtr artifact,
            uint artifactLength,
            ref Cb2VecTrainerConfigV1 config,
            out Cb2VecTrainerHandle trainer);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_trainer_get_info_v1(
            Cb2VecTrainerHandle trainer,
            out Cb2VecModelInfoV1 info);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_trainer_predict_logit_v1(
            Cb2VecTrainerHandle trainer,
            IntPtr tokens,
            uint tokensLength,
            IntPtr siteOffsets,
            IntPtr siteGroups,
            uint siteCount,
            out float score);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_trainer_predict_probability_v1(
            Cb2VecTrainerHandle trainer,
            IntPtr tokens,
            uint tokensLength,
            IntPtr siteOffsets,
            IntPtr siteGroups,
            uint siteCount,
            out float probability);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_trainer_evaluate_v1(
            Cb2VecTrainerHandle trainer,
            ref Cb2VecTrainingBatchViewV1 batch,
            out Cb2VecTrainingMetricsV1 metrics);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_trainer_train_batch_v1(
            Cb2VecTrainerHandle trainer,
            ref Cb2VecTrainingBatchViewV1 batch,
            out Cb2VecTrainingMetricsV1 metrics);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_trainer_train_epoch_v1(
            Cb2VecTrainerHandle trainer,
            ref Cb2VecTrainingBatchViewV1 batch,
            out Cb2VecTrainingMetricsV1 metrics);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_trainer_quantize_v1(
            Cb2VecTrainerHandle trainer,
            ref Cb2VecQuantizationConfigV1 quantization,
            out Cb2VecModelHandle model);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_trainer_write_artifact_v1(
            Cb2VecTrainerHandle trainer,
            ref Cb2VecQuantizationConfigV1 quantization,
            IntPtr sourceSha256,
            IntPtr output,
            uint outputCapacity,
            out uint requiredOrWritten);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_trainer_free_v1(IntPtr trainer);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_model_load_v1(
            IntPtr artifact,
            uint artifactLength,
            ref Cb2VecInferenceConfigV1 inference,
            out Cb2VecModelHandle model);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_model_get_info_v1(
            Cb2VecModelHandle model,
            out Cb2VecModelInfoV1 info);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_model_predict_v1(
            Cb2VecModelHandle model,
            IntPtr tokens,
            uint tokensLength,
            IntPtr siteOffsets,
            IntPtr siteGroups,
            uint siteCount,
            out float score);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_model_predict_batch_v1(
            Cb2VecModelHandle model,
            ref Cb2VecTrainingBatchViewV1 batch,
            IntPtr scores,
            uint scoresLength);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_model_free_v1(IntPtr model);
    }
}
