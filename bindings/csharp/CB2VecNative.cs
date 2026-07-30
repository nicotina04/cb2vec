// CB2Vec 0.3.0 / C ABI 1.1 Unity binding.
//
// Copy this file to Assets/CB2Vec/Runtime/ in the Unity project and place the
// matching native library under Assets/CB2Vec/Plugins/ for the target
// platform. See unity/README.md for a ready made Assets/CB2Vec tree with
// correct PluginImporter settings and assembly definitions.
//
// Two API tiers:
//
//   * Convenience  - Cb2VecModel.Predict(Cb2VecInput), PredictBatch(...).
//                    Pins on every call. Fine outside a hot loop.
//   * Zero-alloc   - Cb2VecSession for incremental make/predict/undo search,
//                    and Cb2VecPinnedInput / Cb2VecPinnedBatch /
//                    Cb2VecPinnedBuffer<T> for repeated whole-input inference.
//                    After construction these perform no managed allocation
//                    and no per-call GCHandle pin.
//
// No unsafe code and no /unsafe compiler switch is required.

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

    [StructLayout(LayoutKind.Sequential)]
    public struct Cb2VecSessionConfigV1
    {
        public uint StructSize;
        public uint AbiVersion;
        public uint MaxSites;
        public uint MaxTokenSlots;
        public uint MaxDeltasPerFrame;
        public uint MaxDepth;
        public uint Flags;
        public uint Reserved0;
    }

    /// <summary>One token replacement inside a pushed search frame.</summary>
    /// <remarks>
    /// <c>Site</c> indexes the site table installed by
    /// <see cref="Cb2VecSession.Reset"/>, <c>Lane</c> indexes the token within
    /// that site, and <c>OldToken</c> must equal the token the session
    /// currently holds there. One frame may not touch the same slot twice.
    /// </remarks>
    [StructLayout(LayoutKind.Sequential)]
    public struct Cb2VecTokenDeltaV1
    {
        public uint Site;
        public uint Lane;
        public ushort OldToken;
        public ushort NewToken;

        public Cb2VecTokenDeltaV1(uint site, uint lane, ushort oldToken, ushort newToken)
        {
            Site = site;
            Lane = lane;
            OldToken = oldToken;
            NewToken = newToken;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Cb2VecSessionInfoV1
    {
        public uint StructSize;
        public uint AbiVersion;
        public uint SiteCount;
        public uint TokenSlots;
        public uint GroupCount;
        public uint Depth;
        public uint MaterializedDepth;
        public uint PendingDeltas;
        public uint MaxSites;
        public uint MaxTokenSlots;
        public uint MaxDeltasPerFrame;
        public uint MaxDepth;
        public uint Activation;
        public uint Pooling;
        public uint Flags;
        public uint Reserved0;
    }

    /// <summary>
    /// Consumer-defined identity of the token vocabulary a model was trained
    /// against. CB2Vec never interprets these values; they let an application
    /// refuse a model whose schema no longer matches its own code.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct Cb2VecArtifactMetadataV1
    {
        public uint StructSize;
        public uint AbiVersion;
        public uint SchemaVersion;
        public uint Flags;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
        public byte[] SchemaDigest;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Cb2VecArtifactInfoV1
    {
        public uint StructSize;
        public uint AbiVersion;
        public uint ArtifactVersion;
        public uint Kind;
        public uint TokenCount;
        public uint GroupCount;
        public uint Dim;
        public uint FmRank;
        public uint HasInferenceConfig;
        public uint Activation;
        public uint Pooling;
        public uint SchemaVersion;
        public int EmbeddingScale;
        public int HeadScale;
        public int FactorScale;
        public uint Flags;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 32)]
        public byte[] SourceSha256;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
        public byte[] SchemaDigest;
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

    /// <summary>
    /// A caller-owned array pinned once and reused across many native calls.
    /// </summary>
    /// <remarks>
    /// Use this when the same buffer feeds repeated inference. Mutate
    /// <see cref="Items"/> in place between calls; the pin, and therefore the
    /// address the native side sees, stays valid until <see cref="Dispose"/>.
    /// </remarks>
    public sealed class Cb2VecPinnedBuffer<T> : IDisposable where T : struct
    {
        private PinnedArray _pinned;

        /// <summary>The pinned storage. Write into it; do not replace it.</summary>
        public T[] Items { get; private set; }

        public int Length { get { return Items.Length; } }

        internal IntPtr Pointer
        {
            get
            {
                if (_pinned == null)
                    throw new ObjectDisposedException("Cb2VecPinnedBuffer");
                return _pinned.Pointer;
            }
        }

        public Cb2VecPinnedBuffer(int length)
            : this(new T[length])
        {
        }

        public Cb2VecPinnedBuffer(T[] items)
        {
            if (items == null)
                throw new ArgumentNullException("items");
            Items = items;
            _pinned = new PinnedArray(items);
        }

        public void Dispose()
        {
            if (_pinned != null)
            {
                _pinned.Dispose();
                _pinned = null;
            }
        }
    }

    /// <summary>
    /// A <see cref="Cb2VecInput"/> pinned once for repeated whole-input
    /// inference through <see cref="Cb2VecModel.PredictInto"/>.
    /// </summary>
    public sealed class Cb2VecPinnedInput : IDisposable
    {
        private PinnedArray _tokens;
        private PinnedArray _offsets;
        private PinnedArray _groups;

        public Cb2VecInput Input { get; private set; }

        internal IntPtr Tokens { get { return _tokens.Pointer; } }
        internal IntPtr Offsets { get { return _offsets.Pointer; } }
        internal IntPtr Groups { get { return _groups.Pointer; } }
        internal uint TokensLength { get; private set; }
        internal uint SiteCount { get; private set; }

        public Cb2VecPinnedInput(Cb2VecInput input)
        {
            if (input == null)
                throw new ArgumentNullException("input");
            Input = input;
            _tokens = new PinnedArray(input.Tokens);
            _offsets = new PinnedArray(input.SiteTokenOffsets);
            _groups = new PinnedArray(input.SiteGroups);
            TokensLength = checked((uint)input.Tokens.Length);
            SiteCount = checked((uint)input.SiteGroups.Length);
        }

        public void Dispose()
        {
            if (_groups != null)
            {
                _groups.Dispose();
                _offsets.Dispose();
                _tokens.Dispose();
                _groups = null;
                _offsets = null;
                _tokens = null;
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

    /// <summary>
    /// A <see cref="Cb2VecTrainingBatch"/> pinned once for repeated batch
    /// inference through <see cref="Cb2VecModel.PredictBatchInto"/>.
    /// </summary>
    public sealed class Cb2VecPinnedBatch : IDisposable
    {
        private PinnedBatch _pinned;

        public Cb2VecTrainingBatch Batch { get; private set; }

        public uint SampleCount { get { return Batch.SampleCount; } }

        internal ref Cb2VecTrainingBatchViewV1 View
        {
            get
            {
                if (_pinned == null)
                    throw new ObjectDisposedException("Cb2VecPinnedBatch");
                return ref _pinned.View;
            }
        }

        public Cb2VecPinnedBatch(Cb2VecTrainingBatch batch)
        {
            if (batch == null)
                throw new ArgumentNullException("batch");
            Batch = batch;
            _pinned = new PinnedBatch(batch);
        }

        public void Dispose()
        {
            if (_pinned != null)
            {
                _pinned.Dispose();
                _pinned = null;
            }
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

    internal sealed class Cb2VecSessionHandle : SafeHandleZeroOrMinusOneIsInvalid
    {
        public Cb2VecSessionHandle() : base(true) { }

        protected override bool ReleaseHandle()
        {
            return NativeMethods.cb2vec_session_free_v1(handle) == Cb2VecNative.Ok;
        }
    }

    public static class Cb2VecNative
    {
        public const uint AbiVersion = 0x00010001;
        public const uint AbiVersion10 = 0x00010000;
        public const int Ok = 0;
        public const int ErrorNullPointer = -1;
        public const int ErrorInvalidArgument = -2;
        public const int ErrorAbiMismatch = -3;
        public const int ErrorArtifact = -4;
        public const int ErrorModel = -5;
        public const int ErrorNumeric = -6;
        public const int ErrorBufferTooSmall = -7;
        public const int ErrorLimitExceeded = -8;
        public const int ErrorState = -9;
        public const int ErrorCheckpoint = -10;
        public const int ErrorOutOfMemory = -11;
        public const int ErrorPanic = -127;

        public const uint ArtifactVersion = 1;
        public const uint ArtifactVersionV2 = 2;

        /// <summary>
        /// Largest number of token slots one session may hold, so that every
        /// integer accumulator stays inside <c>int</c> without a checked add.
        /// </summary>
        public const uint SessionMaxTokenSlots = 65535;

        static Cb2VecNative()
        {
            ValidateLayouts();
            uint native = NativeMethods.cb2vec_abi_version();
            uint nativeMajor = native >> 16;
            // Minor revisions are additive, so only the major must agree.
            if (nativeMajor != 1)
                throw new DllNotFoundException(
                    "Incompatible CB2Vec ABI 0x" + native.ToString("X8") + ".");
        }

        public static void EnsureCompatible()
        {
            // Calling this method runs the static constructor.
        }

        /// <summary>
        /// The ABI revision the loaded native library reports. Compare only
        /// the major half: minor revisions are additive.
        /// </summary>
        public static uint NativeAbiVersion
        {
            get
            {
                EnsureCompatible();
                return NativeMethods.cb2vec_abi_version();
            }
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

        public static Cb2VecSessionConfigV1 DefaultSessionConfig()
        {
            EnsureCompatible();
            Cb2VecSessionConfigV1 value;
            Check(NativeMethods.cb2vec_session_config_default_v1(out value));
            return value;
        }

        /// <summary>An all-zero, "unspecified" schema identity.</summary>
        public static Cb2VecArtifactMetadataV1 EmptyMetadata()
        {
            return new Cb2VecArtifactMetadataV1
            {
                StructSize = checked((uint)Marshal.SizeOf(typeof(Cb2VecArtifactMetadataV1))),
                AbiVersion = AbiVersion,
                SchemaVersion = 0,
                Flags = 0,
                SchemaDigest = new byte[16],
            };
        }

        public static Cb2VecArtifactMetadataV1 Metadata(uint schemaVersion, byte[] schemaDigest)
        {
            if (schemaDigest == null)
                throw new ArgumentNullException("schemaDigest");
            if (schemaDigest.Length != 16)
                throw new ArgumentException("schemaDigest must contain exactly 16 bytes.");
            Cb2VecArtifactMetadataV1 value = EmptyMetadata();
            value.SchemaVersion = schemaVersion;
            Array.Copy(schemaDigest, value.SchemaDigest, 16);
            return value;
        }

        /// <summary>
        /// Reads artifact metadata without building a model: format version,
        /// shape, quantization scales, whether the file carries its own
        /// inference recipe, and the consumer-defined schema identity.
        /// </summary>
        public static Cb2VecArtifactInfoV1 ProbeArtifact(byte[] artifact)
        {
            if (artifact == null)
                throw new ArgumentNullException("artifact");
            EnsureCompatible();
            using (PinnedArray pinned = new PinnedArray(artifact))
            {
                Cb2VecArtifactInfoV1 info;
                Check(NativeMethods.cb2vec_artifact_probe_v1(
                    pinned.Pointer, checked((uint)artifact.Length), out info));
                return info;
            }
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
            RequireSize(typeof(Cb2VecSessionConfigV1), 32);
            RequireSize(typeof(Cb2VecTokenDeltaV1), 12);
            RequireSize(typeof(Cb2VecSessionInfoV1), 64);
            RequireSize(typeof(Cb2VecArtifactMetadataV1), 32);
            RequireSize(typeof(Cb2VecArtifactInfoV1), 112);
            RequireOffset(typeof(Cb2VecTrainerConfigV1), "Seed", 32);
            RequireOffset(typeof(Cb2VecTrainerConfigV1), "LearningRate", 40);
            RequireOffset(typeof(Cb2VecTrainingMetricsV1), "TotalWeight", 16);
            RequireOffset(typeof(Cb2VecTrainingMetricsV1), "SampleCount", 24);
            RequireOffset(typeof(Cb2VecModelInfoV1), "FactorScale", 52);
            RequireOffset(typeof(Cb2VecTokenDeltaV1), "OldToken", 8);
            RequireOffset(typeof(Cb2VecSessionInfoV1), "MaxSites", 32);
            RequireOffset(typeof(Cb2VecArtifactMetadataV1), "SchemaDigest", 16);
            RequireOffset(typeof(Cb2VecArtifactInfoV1), "SourceSha256", 64);
            RequireOffset(typeof(Cb2VecArtifactInfoV1), "SchemaDigest", 96);
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

        /// <summary>Exact byte length this trainer's checkpoint will occupy.</summary>
        public int CheckpointLength()
        {
            uint length;
            Cb2VecNative.Check(
                NativeMethods.cb2vec_trainer_checkpoint_len_v1(_handle, out length));
            return checked((int)length);
        }

        /// <summary>
        /// Serializes complete trainer state: weights, Adam moments, optimizer
        /// step, shuffle RNG, completed epochs, and the trainer config.
        /// </summary>
        /// <remarks>
        /// Unlike <see cref="WriteArtifact"/>, which produces a lightweight
        /// deployment file, resuming from a checkpoint reproduces an
        /// uninterrupted run bit for bit.
        /// </remarks>
        public byte[] WriteCheckpoint()
        {
            uint required;
            int probe = NativeMethods.cb2vec_trainer_write_checkpoint_v1(
                _handle, IntPtr.Zero, 0, out required);
            if (probe != Cb2VecNative.ErrorBufferTooSmall)
                Cb2VecNative.Check(probe);

            byte[] checkpoint = new byte[checked((int)required)];
            using (PinnedArray output = new PinnedArray(checkpoint))
            {
                uint written;
                Cb2VecNative.Check(NativeMethods.cb2vec_trainer_write_checkpoint_v1(
                    _handle, output.Pointer, required, out written));
                if (written != required)
                    throw new InvalidOperationException(
                        "CB2Vec checkpoint byte count changed between calls.");
            }
            return checkpoint;
        }

        /// <summary>
        /// Restores a trainer that continues exactly where the checkpoint
        /// stopped. The trainer config travels with the file.
        /// </summary>
        public static Cb2VecTrainer LoadCheckpoint(byte[] checkpoint)
        {
            if (checkpoint == null)
                throw new ArgumentNullException("checkpoint");
            Cb2VecNative.EnsureCompatible();
            using (PinnedArray pinned = new PinnedArray(checkpoint))
            {
                Cb2VecTrainerHandle handle;
                Cb2VecNative.Check(NativeMethods.cb2vec_trainer_load_checkpoint_v1(
                    pinned.Pointer, checked((uint)checkpoint.Length), out handle));
                return new Cb2VecTrainer(handle);
            }
        }

        /// <summary>
        /// Writes a version-2 artifact that stores its own activation and
        /// pooling, taken from this trainer, so a consumer cannot load it with
        /// the wrong recipe.
        /// </summary>
        public byte[] WriteArtifactV2(
            Cb2VecQuantizationConfigV1 quantization,
            byte[] sourceSha256,
            Cb2VecArtifactMetadataV1 metadata)
        {
            if (sourceSha256 == null)
                throw new ArgumentNullException("sourceSha256");
            if (sourceSha256.Length != 32)
                throw new ArgumentException("sourceSha256 must contain exactly 32 bytes.");
            if (metadata.SchemaDigest == null || metadata.SchemaDigest.Length != 16)
                throw new ArgumentException("metadata.SchemaDigest must contain exactly 16 bytes.");

            using (PinnedArray digest = new PinnedArray(sourceSha256))
            {
                uint required;
                int probe = NativeMethods.cb2vec_trainer_write_artifact_v2(
                    _handle,
                    ref quantization,
                    digest.Pointer,
                    ref metadata,
                    IntPtr.Zero,
                    0,
                    out required);
                if (probe != Cb2VecNative.ErrorBufferTooSmall)
                    Cb2VecNative.Check(probe);

                byte[] artifact = new byte[checked((int)required)];
                using (PinnedArray output = new PinnedArray(artifact))
                {
                    uint written;
                    Cb2VecNative.Check(NativeMethods.cb2vec_trainer_write_artifact_v2(
                        _handle,
                        ref quantization,
                        digest.Pointer,
                        ref metadata,
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

        /// <summary>
        /// Loads a model with an explicit inference recipe.
        /// </summary>
        /// <remarks>
        /// A version-2 artifact whose stored recipe disagrees with
        /// <paramref name="inference"/> is rejected rather than silently
        /// scored with the wrong activation or pooling.
        /// </remarks>
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

        /// <summary>
        /// Loads a version-2 artifact using the inference recipe it stores.
        /// </summary>
        /// <remarks>
        /// This is the mismatch-proof path: the activation and pooling come
        /// from the file that was written by the trainer that produced the
        /// weights. Version-1 artifacts have no stored recipe and must use the
        /// <see cref="Load(byte[], Cb2VecInferenceConfigV1)"/> overload.
        /// </remarks>
        public static Cb2VecModel Load(byte[] artifact)
        {
            return Load(artifact, (Cb2VecArtifactMetadataV1?)null);
        }

        /// <summary>
        /// Loads a version-2 artifact and refuses it unless its recorded
        /// schema identity matches <paramref name="expectedSchema"/>.
        /// </summary>
        /// <remarks>
        /// Artifacts that record no schema identity are accepted, so an
        /// unlabeled model can still be loaded deliberately. Pass
        /// <c>null</c> to skip the check entirely.
        /// </remarks>
        public static Cb2VecModel Load(
            byte[] artifact,
            Cb2VecArtifactMetadataV1? expectedSchema)
        {
            if (artifact == null)
                throw new ArgumentNullException("artifact");
            Cb2VecNative.EnsureCompatible();
            using (PinnedArray pinned = new PinnedArray(artifact))
            {
                Cb2VecModelHandle handle;
                int status;
                if (expectedSchema.HasValue)
                {
                    Cb2VecArtifactMetadataV1 schema = expectedSchema.Value;
                    if (schema.SchemaDigest == null || schema.SchemaDigest.Length != 16)
                        throw new ArgumentException(
                            "expectedSchema.SchemaDigest must contain exactly 16 bytes.");
                    status = NativeMethods.cb2vec_model_load_v2_schema(
                        pinned.Pointer,
                        checked((uint)artifact.Length),
                        IntPtr.Zero,
                        ref schema,
                        out handle);
                }
                else
                {
                    status = NativeMethods.cb2vec_model_load_v2(
                        pinned.Pointer,
                        checked((uint)artifact.Length),
                        IntPtr.Zero,
                        IntPtr.Zero,
                        out handle);
                }
                Cb2VecNative.Check(status);
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

        /// <summary>
        /// Creates an incremental search session over this model.
        /// </summary>
        /// <remarks>
        /// The session shares ownership of the weights, so it stays valid even
        /// if this model is disposed first. Create one session per search
        /// thread; sessions never share mutable state.
        /// </remarks>
        public Cb2VecSession CreateSession(Cb2VecSessionConfigV1 config)
        {
            Cb2VecSessionHandle session;
            Cb2VecNative.Check(NativeMethods.cb2vec_session_create_v1(
                _handle, ref config, out session));
            return new Cb2VecSession(session);
        }

        /// <summary>
        /// Scores an already-pinned input without allocating or pinning.
        /// </summary>
        /// <remarks>
        /// Use this for repeated whole-input inference outside a make/undo
        /// search. Inside one, prefer <see cref="Cb2VecSession"/>, which also
        /// avoids recomputing features that did not change.
        /// </remarks>
        public float PredictInto(Cb2VecPinnedInput input)
        {
            if (input == null)
                throw new ArgumentNullException("input");
            float score;
            Cb2VecNative.Check(NativeMethods.cb2vec_model_predict_v1(
                _handle,
                input.Tokens,
                input.TokensLength,
                input.Offsets,
                input.Groups,
                input.SiteCount,
                out score));
            return score;
        }

        /// <summary>
        /// Scores a pinned batch into a pinned caller-owned score buffer,
        /// without allocating or pinning.
        /// </summary>
        /// <remarks>
        /// Scores are written only after every sample predicts successfully,
        /// so a failed call leaves <paramref name="scores"/> untouched.
        /// </remarks>
        public void PredictBatchInto(Cb2VecPinnedBatch batch, Cb2VecPinnedBuffer<float> scores)
        {
            if (batch == null)
                throw new ArgumentNullException("batch");
            if (scores == null)
                throw new ArgumentNullException("scores");
            if (scores.Length != batch.SampleCount)
                throw new ArgumentException(
                    "scores must contain exactly one value per batch sample.");
            Cb2VecNative.Check(NativeMethods.cb2vec_model_predict_batch_v1(
                _handle,
                ref batch.View,
                scores.Pointer,
                checked((uint)scores.Length)));
        }

        /// <summary>Schema identity recorded in the artifact this model came from.</summary>
        public Cb2VecArtifactMetadataV1 GetMetadata()
        {
            Cb2VecArtifactMetadataV1 metadata;
            Cb2VecNative.Check(
                NativeMethods.cb2vec_model_get_metadata_v1(_handle, out metadata));
            return metadata;
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

    /// <summary>
    /// Incremental make/predict/undo evaluator over one immutable model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A session owns the integer evaluator state for one search thread. Every
    /// buffer it needs is allocated when it is created, so after
    /// <see cref="Reset"/> the push/predict/pop loop performs no managed
    /// allocation, no native allocation, and no per-call pin.
    /// </para>
    /// <para><b>Threading.</b> A session is single-owner: use one session per
    /// search thread. Any number of sessions may share one
    /// <see cref="Cb2VecModel"/>, which is immutable.</para>
    /// <para><b>Lifetime.</b> A session keeps the model's weights alive on its
    /// own, so disposing the model first is safe. Handles may be disposed in
    /// any order, and disposing twice is a no-op.</para>
    /// <example>
    /// <code>
    /// using (var session = model.CreateSession(config))
    /// using (var frame = new Cb2VecPinnedBuffer&lt;Cb2VecTokenDeltaV1&gt;(4))
    /// {
    ///     session.Reset(initialInput);
    ///     frame.Items[0] = new Cb2VecTokenDeltaV1(site, lane, oldToken, newToken);
    ///     session.Push(frame, 1);
    ///     float score = session.Predict();
    ///     session.Pop();
    /// }
    /// </code>
    /// </example>
    /// </remarks>
    public sealed class Cb2VecSession : IDisposable
    {
        private Cb2VecSessionHandle _handle;

        internal Cb2VecSession(Cb2VecSessionHandle handle)
        {
            _handle = handle;
        }

        /// <summary>Installs a complete position and clears the frame stack.</summary>
        public void Reset(Cb2VecInput input)
        {
            if (input == null)
                throw new ArgumentNullException("input");
            using (PinnedInput pinned = new PinnedInput(input))
            {
                Cb2VecNative.Check(NativeMethods.cb2vec_session_reset_v1(
                    Handle(),
                    pinned.Tokens,
                    checked((uint)input.Tokens.Length),
                    pinned.Offsets,
                    pinned.Groups,
                    checked((uint)input.SiteGroups.Length)));
            }
        }

        /// <summary>
        /// Installs a position from an already-pinned input, without pinning.
        /// </summary>
        public void Reset(Cb2VecPinnedInput input)
        {
            if (input == null)
                throw new ArgumentNullException("input");
            Cb2VecSessionHandle handle = Live();
            bool added = false;
            try
            {
                handle.DangerousAddRef(ref added);
                Cb2VecNative.Check(NativeMethods.cb2vec_session_reset_v1(
                    handle.DangerousGetHandle(),
                    input.Tokens,
                    input.TokensLength,
                    input.Offsets,
                    input.Groups,
                    input.SiteCount));
            }
            finally
            {
                if (added)
                    handle.DangerousRelease();
            }
        }

        /// <summary>
        /// Pushes one search move's replacements as a single reversible frame.
        /// </summary>
        /// <remarks>
        /// Every delta is validated before anything changes, so a rejected
        /// frame leaves the session exactly as it was and does not consume
        /// depth. <paramref name="count"/> may be zero, which still pushes a
        /// frame and keeps push/pop balanced for a null move.
        /// </remarks>
        public void Push(Cb2VecPinnedBuffer<Cb2VecTokenDeltaV1> deltas, int count)
        {
            if (deltas == null)
                throw new ArgumentNullException("deltas");
            if (count < 0 || count > deltas.Length)
                throw new ArgumentOutOfRangeException("count");
            Cb2VecSessionHandle handle = Live();
            bool added = false;
            try
            {
                handle.DangerousAddRef(ref added);
                Cb2VecNative.Check(NativeMethods.cb2vec_session_push_v1(
                    handle.DangerousGetHandle(), deltas.Pointer, (uint)count));
            }
            finally
            {
                if (added)
                    handle.DangerousRelease();
            }
        }

        /// <summary>Pushes every delta in a reusable buffer.</summary>
        public void Push(Cb2VecPinnedBuffer<Cb2VecTokenDeltaV1> deltas)
        {
            if (deltas == null)
                throw new ArgumentNullException("deltas");
            Push(deltas, deltas.Length);
        }

        /// <summary>
        /// Convenience overload that pins <paramref name="deltas"/> for the
        /// call. Prefer the <see cref="Cb2VecPinnedBuffer{T}"/> overload inside
        /// a search loop.
        /// </summary>
        public void Push(Cb2VecTokenDeltaV1[] deltas, int count)
        {
            if (deltas == null)
                throw new ArgumentNullException("deltas");
            if (count < 0 || count > deltas.Length)
                throw new ArgumentOutOfRangeException("count");
            using (PinnedArray pinned = new PinnedArray(deltas))
            {
                Cb2VecNative.Check(NativeMethods.cb2vec_session_push_v1(
                    Handle(), pinned.Pointer, (uint)count));
            }
        }

        /// <summary>Applies pending frames; <see cref="Predict"/> does this too.</summary>
        public void Materialize()
        {
            Cb2VecSessionHandle handle = Live();
            bool added = false;
            try
            {
                handle.DangerousAddRef(ref added);
                Cb2VecNative.Check(
                    NativeMethods.cb2vec_session_materialize_v1(handle.DangerousGetHandle()));
            }
            finally
            {
                if (added)
                    handle.DangerousRelease();
            }
        }

        /// <summary>
        /// Materializes pending frames and scores the current position.
        /// Bit-identical to <see cref="Cb2VecModel.Predict"/> over the same
        /// tokens with the same inference recipe.
        /// </summary>
        public float Predict()
        {
            Cb2VecSessionHandle handle = Live();
            bool added = false;
            try
            {
                handle.DangerousAddRef(ref added);
                float score;
                Cb2VecNative.Check(NativeMethods.cb2vec_session_predict_v1(
                    handle.DangerousGetHandle(), out score));
                return score;
            }
            finally
            {
                if (added)
                    handle.DangerousRelease();
            }
        }

        /// <summary>Undoes the most recent frame and returns its delta count.</summary>
        public int Pop()
        {
            Cb2VecSessionHandle handle = Live();
            bool added = false;
            try
            {
                handle.DangerousAddRef(ref added);
                uint popped;
                Cb2VecNative.Check(NativeMethods.cb2vec_session_pop_v1(
                    handle.DangerousGetHandle(), out popped));
                return checked((int)popped);
            }
            finally
            {
                if (added)
                    handle.DangerousRelease();
            }
        }

        public Cb2VecSessionInfoV1 GetInfo()
        {
            Cb2VecSessionInfoV1 info;
            Cb2VecNative.Check(
                NativeMethods.cb2vec_session_get_info_v1(Handle(), out info));
            return info;
        }

        /// <summary>
        /// Returns the live handle, or throws if this session was already
        /// disposed. Every entry point goes through here, so disposing a
        /// session and then using it raises <see cref="ObjectDisposedException"/>
        /// rather than touching freed native memory.
        /// </summary>
        private Cb2VecSessionHandle Live()
        {
            if (_handle == null || _handle.IsClosed)
                throw new ObjectDisposedException("Cb2VecSession");
            return _handle;
        }

        private IntPtr Handle()
        {
            return Live().DangerousGetHandle();
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

        // ---- ABI 1.1 ----

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_session_config_default_v1(
            out Cb2VecSessionConfigV1 config);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_session_create_v1(
            Cb2VecModelHandle model,
            ref Cb2VecSessionConfigV1 config,
            out Cb2VecSessionHandle session);

        // The session hot path takes a raw handle rather than a SafeHandle so
        // the call is provably free of managed allocation. Callers hold a
        // DangerousAddRef for the duration of each call.
        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_session_reset_v1(
            IntPtr session,
            IntPtr tokens,
            uint tokensLength,
            IntPtr siteOffsets,
            IntPtr siteGroups,
            uint siteCount);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_session_push_v1(
            IntPtr session,
            IntPtr deltas,
            uint deltaCount);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_session_materialize_v1(IntPtr session);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_session_predict_v1(
            IntPtr session,
            out float score);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_session_pop_v1(
            IntPtr session,
            out uint poppedDeltas);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_session_get_info_v1(
            IntPtr session,
            out Cb2VecSessionInfoV1 info);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_session_free_v1(IntPtr session);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_trainer_checkpoint_len_v1(
            Cb2VecTrainerHandle trainer,
            out uint length);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_trainer_write_checkpoint_v1(
            Cb2VecTrainerHandle trainer,
            IntPtr output,
            uint outputCapacity,
            out uint requiredOrWritten);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_trainer_load_checkpoint_v1(
            IntPtr checkpoint,
            uint checkpointLength,
            out Cb2VecTrainerHandle trainer);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_artifact_probe_v1(
            IntPtr artifact,
            uint artifactLength,
            out Cb2VecArtifactInfoV1 info);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_trainer_write_artifact_v2(
            Cb2VecTrainerHandle trainer,
            ref Cb2VecQuantizationConfigV1 quantization,
            IntPtr sourceSha256,
            ref Cb2VecArtifactMetadataV1 metadata,
            IntPtr output,
            uint outputCapacity,
            out uint requiredOrWritten);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_model_load_v2(
            IntPtr artifact,
            uint artifactLength,
            IntPtr inference,
            IntPtr expectedSchema,
            out Cb2VecModelHandle model);

        // Same entry point, typed overload for a non-null schema check.
        [DllImport(Library, EntryPoint = "cb2vec_model_load_v2",
                   ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_model_load_v2_schema(
            IntPtr artifact,
            uint artifactLength,
            IntPtr inference,
            ref Cb2VecArtifactMetadataV1 expectedSchema,
            out Cb2VecModelHandle model);

        [DllImport(Library, ExactSpelling = true, CallingConvention = Cdecl)]
        internal static extern int cb2vec_model_get_metadata_v1(
            Cb2VecModelHandle model,
            out Cb2VecArtifactMetadataV1 metadata);
    }
}
