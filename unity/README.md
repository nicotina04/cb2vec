# CB2Vec Unity native plug-in package

A ready-to-copy `Assets/Plugins` tree for Unity 6 LTS, with the
`PluginImporter` settings already correct. Copy the built binaries in, copy the
tree into your project, and Unity imports each library for exactly one
platform.

## Layout

```text
unity/Assets/Plugins/
  CB2VecNative.cs                        <- copy from bindings/csharp/
  CB2VecNative.cs.meta
  x86_64/
    cb2vec.dll                           <- copy from target/release/
    cb2vec.dll.meta                      Editor + Standalone Windows x86_64
  Android/
    arm64-v8a/
      libcb2vec.so                       <- primary Android target
      libcb2vec.so.meta                  Android only, CPU ARM64
    armeabi-v7a/
      libcb2vec.so
      libcb2vec.so.meta                  Android only, CPU ARMv7
    x86_64/
      libcb2vec.so                       <- emulator
      libcb2vec.so.meta                  Android only, CPU X86_64
  Editor/
    CB2VecPluginImportFixer.cs           corrects settings on import
    CB2VecPluginImportFixer.cs.meta
```

Only the `.meta` files, the C# binding, and the Editor script are checked in.
The binaries come from your own build, so the tree cannot go stale against the
crate version you are actually using.

## Install

1. **Build the native libraries.**

   Windows Editor and player:

   ```powershell
   cargo build --release --no-default-features
   # -> target/release/cb2vec.dll
   ```

   Android, with `cargo-ndk` and the NDK that ships with your Unity Editor:

   ```powershell
   rustup target add aarch64-linux-android `
     armv7-linux-androideabi `
     x86_64-linux-android
   cargo install cargo-ndk --locked --version 4.1.2

   $env:ANDROID_NDK_HOME = `
     'C:\Program Files\Unity\Hub\Editor\<version>\Editor\Data\PlaybackEngines\AndroidPlayer\NDK'
   # At or below the project's Minimum API Level. 23 covers a project set to 25.
   $env:CARGO_NDK_PLATFORM = '23'

   cargo ndk -t arm64-v8a -t armeabi-v7a -t x86_64 `
     -o .\build\android\jniLibs `
     build --release --no-default-features
   ```

2. **Copy the binaries into this tree**, next to their `.meta` files:

   ```text
   target/release/cb2vec.dll                      -> Assets/Plugins/x86_64/
   build/android/jniLibs/arm64-v8a/libcb2vec.so   -> Assets/Plugins/Android/arm64-v8a/
   build/android/jniLibs/armeabi-v7a/libcb2vec.so -> Assets/Plugins/Android/armeabi-v7a/
   build/android/jniLibs/x86_64/libcb2vec.so      -> Assets/Plugins/Android/x86_64/
   bindings/csharp/CB2VecNative.cs                -> Assets/Plugins/
   ```

3. **Verify before importing.** This catches a mislabelled ABI without opening
   Unity:

   ```sh
   python3 tools/verify_unity_plugins.py --require-binaries
   ```

4. **Copy `unity/Assets/Plugins` into your project's `Assets/` folder**, keeping
   the `.meta` files. Unity picks up the settings as-is; no Inspector work.

## Verify inside Unity

The Editor script adds two menu items:

- **Tools > CB2Vec > Validate Native Plugin Import Settings** — logs any
  plugin whose platform or CPU is wrong, or that is missing.
- **Tools > CB2Vec > Fix Native Plugin Import Settings** — rewrites them.

The script also runs automatically on import, so a library copied in *without*
its `.meta` still ends up configured correctly. That matters because Unity's
default guess for an unknown `.so` is "every platform, CPU ARMv7" — the exact
misconfiguration that ships an ARM64 library labelled ARMv7.

Then confirm the library actually loads:

```csharp
using UnityEngine;
using CB2Vec;

public sealed class CB2VecProbe : MonoBehaviour
{
    private void Start()
    {
        Debug.Log($"CB2Vec {Cb2VecNative.LibraryVersion} " +
                  $"ABI 0x{Cb2VecNative.NativeAbiVersion:X8}");
    }
}
```

Expected checks:

| Where | Expectation |
|---|---|
| Windows Editor | logs the version and ABI `0x00010001` |
| `Build Settings > Android`, ARM64 only | `arm64-v8a` is packaged, others are not |
| Device (`adb logcat`) | same version line at startup |
| APK inspection | `lib/arm64-v8a/libcb2vec.so` present, `lib/armeabi-v7a/` absent |

## Player settings

| Setting | Value |
|---|---|
| Scripting Backend | IL2CPP |
| Target Architectures | ARM64 (add ARMv7 only if you still ship 32-bit) |
| Minimum API Level | 25 or higher (the libraries are built for 23+) |
| Api Compatibility Level | .NET Standard 2.1 |
| Allow unsafe code | **not required** |

`[DllImport("cb2vec")]` resolves to `cb2vec.dll` on Windows and
`libcb2vec.so` on Android; the import name is the same on every platform, with
no `lib` prefix and no extension.

## Threading in a search

One immutable `Cb2VecModel`, one `Cb2VecSession` per search thread:

```csharp
// Once, on load.
_model = Cb2VecModel.Load(artifactBytes);   // v2 artifact carries its recipe

// Once per search thread.
var config = Cb2VecNative.DefaultSessionConfig();
config.MaxSites = 225;          // a 15x15 board
config.MaxTokenSlots = 900;
config.MaxDeltasPerFrame = 8;
config.MaxDepth = 64;           // deepest ply the search will reach
_session = _model.CreateSession(config);
_frame = new Cb2VecPinnedBuffer<Cb2VecTokenDeltaV1>(8);
```

Sessions never share mutable state, and a session keeps the model's weights
alive on its own, so disposal order does not matter.

## Notes

- The `.meta` GUIDs are fixed and must not be regenerated. Unity keys a
  plugin's import settings to its GUID; a new GUID silently resets them.
- Do not add the Android `.so` files as Editor, Windows, Linux, or macOS
  plugins. The verifier and the Editor script both reject that.
- `panic = "unwind"` must stay set in the release profile so a Rust panic
  becomes an ABI status instead of aborting the player. A parent Cargo
  workspace that forces `panic = "abort"` removes that safety net.
