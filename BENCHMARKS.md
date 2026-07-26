# Benchmark methodology and evidence

CB2Vec is intended for mutable categorical state, especially workloads that
repeatedly make a small change, evaluate, and undo. A useful benchmark must
therefore measure more than an isolated forward pass.

This document separates three questions:

1. Does a codebook representation fit the problem better than a sparse flat
   NNUE?
2. Does incremental token maintenance reduce end-to-end search cost?
3. Does either change preserve decisions and improve task quality?

The answers below are case-study evidence, not a promise that every CB2Vec
consumer will reproduce the same ratios.

## Gomoku engine case study

An earlier evaluator comparison in a Gomoku engine
([FIGRID](https://github.com/nicotina04/figrid-board)) held the engine and
search framework fixed and swapped the learned leaf evaluator. The codebook
lineage was later separated into CB2Vec.

| Metric | Flat NORU-style NNUE | Codebook evaluator | Observed difference |
|---|---:|---:|---:|
| Parameters | 7,480,065 | 66,849 | about 112 times fewer |
| Search p50 NPS | 9.2k | 44.9k | about 4.88 times higher |
| Search depth p50 / p90 | 6 / 7 | 7 / 9 | +1 / +2 ply |
| 300-game same-binary result | 139 / 300 | 161 / 300 | parity-plus, below the preregistered 165-win strength gate |

The comparison used the same 6,163,315 training labels and a frozen
61,782-row validation set. Offline BCE was `0.394960` for the flat model and
`0.373724` for the codebook; live-band BCE was `0.744398` and `0.729012`
respectively.

### What this result does and does not say

This was a representation-and-evaluator comparison inside one engine before
the standalone CB2Vec 0.2 trainer and C ABI existed. It is evidence that a
categorical codebook can be a better systems fit for a suitable workload. It
is not a direct crate microbenchmark of current `noru` versus current
`cb2vec`, and it does not establish a universal 4.88-times speedup.

The 161/300 arena result also missed the preregistered 165-win deployment
strength threshold. The honest conclusion was efficiency with parity-plus
playing evidence, not a strict strength win.

## Exact incremental-update case study

After the codebook evaluator was deployed, exact directional token deltas
replaced full accumulator refreshes for changed `(site, lane)` pairs.

| Workload | Wall-time ratio | Saving |
|---|---:|---:|
| Fixed-root search with tactical VCT disabled | `0.803242` | 19.68% |
| Sealed product search with VCT enabled | `0.907485` | 9.25% |

Correctness gates covered 100,000 mixed make/undo operations, 100,000
directional transitions, and a sealed 1,022-root search corpus. Decisions and
node counts were identical. The generic reversible journal was then extracted
into CB2Vec; extraction itself was an architecture change, not an additional
speed claim.

The public experiment report is
[Exact codebook directional deltas](https://github.com/nicotina04/figrid-board/blob/main/experiments/2026-07-25/cb_d1_directional_delta_results.md).
Related packed-window, frontier, and journal evidence remains in the
[FIGRID measured state-update path](https://github.com/nicotina04/figrid-board#measured-state-update-path).

## How to benchmark a new consumer

Record the exact CB2Vec version, model digest, compiler, target CPU, hardware,
thread count, and input corpus. Then measure at least:

| Layer | Required measurements |
|---|---|
| Model | parameter count, artifact bytes, resident bytes after loading |
| Evaluator | full rebuild, single-token replacement, score-only latency |
| Search transaction | make + update + score + undo latency and allocations |
| Workload | wall time, actual visited nodes, throughput, reached depth |
| Correctness | rebuild parity, long mixed make/undo audit, decision parity |
| Quality | held-out loss by relevant slice and a task-level paired gate |

Run the old and new evaluator in the same executable when practical. Keep
search, roots, time controls, and tactical subsystems fixed. Report
distributions rather than a single mean, especially p50 and p90/p95.

## Evidence still needed

The strongest next result would be a second, non-Gomoku consumer such as a
scheduling, graph-coloring, or tactical-state adapter. That would test the
general API and the learned representation independently of line-pattern
geometry.

A current-version NORU/CB2Vec comparison should also publish its full harness
and raw result bundle. Until then, the historical case study above should be
used as a reason to benchmark the representation, not as a headline guarantee.
