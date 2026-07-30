#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace CB2Vec.EditorTools
{
    /// <summary>
    /// Keeps the CB2Vec native plugins' import settings correct.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The shipped <c>.meta</c> files already carry the right settings, but a
    /// <c>.meta</c> is easy to lose: copying only the binaries, a partial
    /// merge, or a source-control filter that skips <c>*.meta</c> all leave
    /// Unity to guess. Unity's guess for an unknown <c>.so</c> is "enabled for
    /// every platform, CPU ARMv7", which produces exactly the failure this
    /// script exists to prevent: an <c>arm64-v8a</c> library shipped as ARMv7,
    /// or an Android <c>.so</c> loaded into the Editor.
    /// </para>
    /// <para>
    /// The rule is positional: a plugin's directory determines its platform
    /// and CPU. Nothing else is inspected, so the mapping cannot drift from
    /// what the build actually produced.
    /// </para>
    /// <para>
    /// Each rule is matched as a path *suffix* anchored at the package root
    /// (<c>CB2Vec/Plugins/...</c>), so the package works wherever it is
    /// dropped inside <c>Assets</c> while still refusing to claim some other
    /// library that merely shares a file name.
    /// </para>
    /// </remarks>
    public sealed class CB2VecPluginImportFixer : AssetPostprocessor
    {
        private const string MenuRoot = "Tools/CB2Vec/";

        /// <summary>Directory suffix -> required import settings.</summary>
        private struct PluginRule
        {
            public string PathSuffix;
            public bool Android;
            public string AndroidCpu;
            public bool EditorAndWindows;
        }

        private static readonly PluginRule[] Rules =
        {
            new PluginRule
            {
                PathSuffix = "CB2Vec/Plugins/Android/arm64-v8a/libcb2vec.so",
                Android = true,
                AndroidCpu = "ARM64",
                EditorAndWindows = false,
            },
            new PluginRule
            {
                PathSuffix = "CB2Vec/Plugins/Android/armeabi-v7a/libcb2vec.so",
                Android = true,
                AndroidCpu = "ARMv7",
                EditorAndWindows = false,
            },
            new PluginRule
            {
                PathSuffix = "CB2Vec/Plugins/Android/x86_64/libcb2vec.so",
                Android = true,
                AndroidCpu = "X86_64",
                EditorAndWindows = false,
            },
            new PluginRule
            {
                PathSuffix = "CB2Vec/Plugins/x86_64/cb2vec.dll",
                Android = false,
                AndroidCpu = "AnyCPU",
                EditorAndWindows = true,
            },
        };

        private static void OnPostprocessAllAssets(
            string[] imported,
            string[] deleted,
            string[] moved,
            string[] movedFrom)
        {
            foreach (string path in imported)
                Apply(path, false);
            foreach (string path in moved)
                Apply(path, false);
        }

        [MenuItem(MenuRoot + "Fix Native Plugin Import Settings")]
        public static void FixAll()
        {
            int fixedCount = 0;
            foreach (string guid in AssetDatabase.FindAssets(string.Empty, new[] { "Assets" }))
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                if (Apply(path, true))
                    fixedCount += 1;
            }
            AssetDatabase.Refresh();
            Debug.Log("CB2Vec: corrected " + fixedCount + " native plugin import setting(s).");
        }

        /// <summary>
        /// Reports every CB2Vec plugin whose settings are wrong, and every one
        /// that is missing. Returns an empty list when the install is sound.
        /// </summary>
        public static List<string> Validate()
        {
            var problems = new List<string>();
            foreach (PluginRule rule in Rules)
            {
                string path = FindAsset(rule.PathSuffix);
                if (path == null)
                {
                    problems.Add("missing: no asset path ends with " + rule.PathSuffix);
                    continue;
                }

                var importer = AssetImporter.GetAtPath(path) as PluginImporter;
                if (importer == null)
                {
                    problems.Add(path + ": not imported as a native plugin");
                    continue;
                }
                if (importer.GetCompatibleWithAnyPlatform())
                    problems.Add(path + ": 'Any Platform' must be off");
                if (importer.GetCompatibleWithPlatform(BuildTarget.Android) != rule.Android)
                {
                    problems.Add(path + ": Android should be " +
                                 (rule.Android ? "enabled" : "disabled"));
                }
                if (rule.Android)
                {
                    string cpu = importer.GetPlatformData(BuildTarget.Android, "CPU");
                    if (!string.Equals(cpu, rule.AndroidCpu, StringComparison.Ordinal))
                    {
                        problems.Add(path + ": Android CPU is '" + cpu + "', expected '" +
                                     rule.AndroidCpu + "'");
                    }
                }
                if (importer.GetCompatibleWithEditor() != rule.EditorAndWindows)
                {
                    problems.Add(path + ": Editor should be " +
                                 (rule.EditorAndWindows ? "enabled" : "disabled"));
                }
                if (importer.GetCompatibleWithPlatform(BuildTarget.StandaloneWindows64) !=
                    rule.EditorAndWindows)
                {
                    problems.Add(path + ": Windows x86_64 should be " +
                                 (rule.EditorAndWindows ? "enabled" : "disabled"));
                }
                foreach (BuildTarget target in ForbiddenDesktopTargets(rule))
                {
                    if (importer.GetCompatibleWithPlatform(target))
                        problems.Add(path + ": " + target + " must be disabled");
                }
            }
            return problems;
        }

        [MenuItem(MenuRoot + "Validate Native Plugin Import Settings")]
        public static void ValidateMenu()
        {
            List<string> problems = Validate();
            if (problems.Count == 0)
            {
                Debug.Log("CB2Vec: native plugin import settings are correct.");
                return;
            }
            Debug.LogError("CB2Vec: " + problems.Count + " plugin problem(s):\n  " +
                           string.Join("\n  ", problems.ToArray()));
        }

        private static IEnumerable<BuildTarget> ForbiddenDesktopTargets(PluginRule rule)
        {
            // An Android library must never be offered to a desktop player; a
            // Windows DLL must never be offered to Linux or macOS.
            yield return BuildTarget.StandaloneLinux64;
            yield return BuildTarget.StandaloneOSX;
            if (!rule.EditorAndWindows)
                yield return BuildTarget.StandaloneWindows64;
        }

        private static bool Apply(string path, bool force)
        {
            if (string.IsNullOrEmpty(path))
                return false;
            string normalized = path.Replace('\\', '/');
            foreach (PluginRule rule in Rules)
            {
                if (!normalized.EndsWith(rule.PathSuffix, StringComparison.Ordinal))
                    continue;
                return Configure(normalized, rule, force);
            }
            return false;
        }

        private static bool Configure(string path, PluginRule rule, bool force)
        {
            var importer = AssetImporter.GetAtPath(path) as PluginImporter;
            if (importer == null)
                return false;

            bool changed = false;
            if (importer.GetCompatibleWithAnyPlatform())
            {
                importer.SetCompatibleWithAnyPlatform(false);
                changed = true;
            }
            changed |= SetPlatform(importer, BuildTarget.Android, rule.Android);
            changed |= SetPlatform(importer, BuildTarget.StandaloneWindows64,
                                   rule.EditorAndWindows);
            changed |= SetPlatform(importer, BuildTarget.StandaloneLinux64, false);
            changed |= SetPlatform(importer, BuildTarget.StandaloneOSX, false);

            if (importer.GetCompatibleWithEditor() != rule.EditorAndWindows)
            {
                importer.SetCompatibleWithEditor(rule.EditorAndWindows);
                changed = true;
            }
            if (rule.EditorAndWindows)
            {
                changed |= SetEditorData(importer, "OS", "Windows");
                changed |= SetEditorData(importer, "CPU", "x86_64");
            }
            if (rule.Android)
            {
                string cpu = importer.GetPlatformData(BuildTarget.Android, "CPU");
                if (!string.Equals(cpu, rule.AndroidCpu, StringComparison.Ordinal))
                {
                    importer.SetPlatformData(BuildTarget.Android, "CPU", rule.AndroidCpu);
                    changed = true;
                }
            }

            if (changed || force)
            {
                importer.SaveAndReimport();
                if (changed)
                {
                    Debug.Log("CB2Vec: corrected import settings for " + path +
                              (rule.Android ? " (Android " + rule.AndroidCpu + ")"
                                            : " (Editor/Windows x86_64)"));
                }
            }
            return changed;
        }

        private static bool SetPlatform(PluginImporter importer, BuildTarget target, bool enabled)
        {
            if (importer.GetCompatibleWithPlatform(target) == enabled)
                return false;
            importer.SetCompatibleWithPlatform(target, enabled);
            return true;
        }

        private static bool SetEditorData(PluginImporter importer, string key, string value)
        {
            if (string.Equals(importer.GetEditorData(key), value, StringComparison.Ordinal))
                return false;
            importer.SetEditorData(key, value);
            return true;
        }

        private static string FindAsset(string suffix)
        {
            string fileName = Path.GetFileNameWithoutExtension(suffix);
            foreach (string guid in AssetDatabase.FindAssets(fileName))
            {
                string path = AssetDatabase.GUIDToAssetPath(guid).Replace('\\', '/');
                if (path.EndsWith(suffix, StringComparison.Ordinal))
                    return path;
            }
            return null;
        }
    }
}
#endif
