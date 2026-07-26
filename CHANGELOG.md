# Changelog

All notable changes to CB2Vec are documented in this file.

## 0.1.1 - 2026-07-26

- Move the canonical repository to
  [github.com/nicotina04/cb2vec](https://github.com/nicotina04/cb2vec).
- Add Windows, Linux, macOS, MSRV, lint, documentation, and package CI.
- Document the default JSON feature and cover canonical, legacy, and rejected
  JSON model inputs.
- Keep the runtime API and artifact format unchanged from 0.1.0.

## 0.1.0 - 2026-07-26

- Extract the game-independent codebook model and integer access kernel from
  FIGRID.
- Add floating-point and quantized linear/FM scoring.
- Add exact flat and class-base-plus-residual embedding representations.
- Add the canonical `CB2VEC01` artifact and legacy `NORUCBF1` reader.
- Add the preallocated reversible token journal used by make/undo search.
