//! Alpha-beta style make/predict/undo over an `IncrementalSession`.
//!
//! The point of the session is that a branch which is entered and abandoned
//! costs almost nothing: pushing a move updates only the logical tokens, and
//! the integer evaluator state is touched lazily, and only for the sites the
//! move changed. Scores stay bit-identical to a full rebuild.
//!
//! Run with `cargo run --example search_session`.

use cb2vec::{
    Activation, CodebookWeights, GroupedTokens, IncrementalSession, InferenceConfig, Pooling,
    SessionDelta, SessionLimits, predict_quantized,
};

/// A 3x3 board: nine sites, one token each, three row groups.
const SITES: usize = 9;
const EMPTY: u16 = 0;

fn site_offsets() -> Vec<u32> {
    (0..=SITES as u32).collect()
}

fn site_groups() -> Vec<u32> {
    (0..SITES as u32).map(|site| site / 3).collect()
}

fn main() {
    let weights = CodebookWeights::deterministic(3, 3, 8, 2).quantize_i16_s32_s64();
    let inference = InferenceConfig::new(Activation::Relu, Pooling::Mean);
    let offsets = site_offsets();
    let groups = site_groups();

    // Capacities are chosen once. Nothing after this allocates.
    let limits = SessionLimits::new(SITES, SITES, 2, 16);
    let mut session = IncrementalSession::new(&weights, inference, limits).unwrap();

    let mut board = vec![EMPTY; SITES];
    session.reset(&board, &offsets, &groups).unwrap();
    println!("empty board: {:+.6}", session.predict().unwrap());

    // One ply of search: try every legal move, score it, and undo.
    let mut best = (f32::NEG_INFINITY, usize::MAX);
    for (site, &occupant) in board.iter().enumerate() {
        if occupant != EMPTY {
            continue;
        }
        session
            .push(&[SessionDelta::new(site as u32, 0, EMPTY, 1)])
            .unwrap();
        let score = session.predict().unwrap();
        if score > best.0 {
            best = (score, site);
        }
        session.pop().unwrap();
    }
    println!("best first move: site {} at {:+.6}", best.1, best.0);

    // Play it, then a reply, and confirm the incremental score still matches a
    // full rebuild exactly.
    session
        .push(&[SessionDelta::new(best.1 as u32, 0, EMPTY, 1)])
        .unwrap();
    board[best.1] = 1;
    let reply = (0..SITES).find(|&site| board[site] == EMPTY).unwrap();
    session
        .push(&[SessionDelta::new(reply as u32, 0, EMPTY, 2)])
        .unwrap();
    board[reply] = 2;

    let incremental = session.predict().unwrap();
    let rebuilt = predict_quantized(
        &GroupedTokens::new(
            board.clone(),
            offsets.iter().map(|&value| value as usize).collect(),
            groups.iter().map(|&value| value as usize).collect(),
        )
        .unwrap(),
        &weights,
        inference,
    )
    .unwrap();
    assert_eq!(
        incremental.to_bits(),
        rebuilt.to_bits(),
        "incremental and full-rebuild scores must be bit-identical"
    );
    println!("after two plies: {incremental:+.6} (matches full rebuild bitwise)");

    // A delta whose expected old token is stale is rejected, and rejecting it
    // leaves the position untouched.
    let stale = session.push(&[SessionDelta::new(best.1 as u32, 0, EMPTY, 2)]);
    assert!(stale.is_err(), "a stale old token must be refused");
    assert_eq!(session.depth(), 2);
    assert_eq!(session.predict().unwrap().to_bits(), incremental.to_bits());
    println!("rejected stale move: {}", stale.unwrap_err());

    // Unwinding restores the starting position exactly.
    session.rewind();
    let restored = session.predict().unwrap();
    session.reset(&[EMPTY; SITES], &offsets, &groups).unwrap();
    assert_eq!(restored.to_bits(), session.predict().unwrap().to_bits());
    println!("unwound to empty board: {restored:+.6}");
}
