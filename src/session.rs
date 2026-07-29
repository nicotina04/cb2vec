//! Preallocated incremental evaluation session for make/predict/undo search.
//!
//! A session owns the integer state that quantized inference needs — per-site
//! embedding sums and per-group pooled accumulators — plus a reversible frame
//! stack of token replacements. Search engines push one frame per move, score
//! the current position, and pop to restore the previous one.
//!
//! # Relationship to [`ReversibleTokenJournal`]
//!
//! [`ReversibleTokenJournal`] is the uniform-lane journal: every site owns
//! exactly `LANES` tokens, fixed at compile time. CB2Vec's deployment input
//! ([`GroupedTokens`]) is ragged — site `s` owns
//! `tokens[site_offsets[s]..site_offsets[s + 1]]` — and a session's shape is
//! only known at run time, behind a C ABI. This module therefore reimplements
//! the same three invariants over a runtime-sized flat slot space:
//!
//! * frames below `materialized_depth` have been applied to the numeric state,
//!   and materialized frames always form a prefix of the pushed frames;
//! * validation of a whole frame completes before any logical token, numeric
//!   accumulator, or depth counter is mutated;
//! * every buffer is sized once at construction, so the push/materialize/
//!   predict/pop loop performs no heap allocation.
//!
//! Exact embedding replacement reuses
//! [`QuantizedCodebookAccess::add_embedding_delta_to`], the same kernel the
//! non-incremental path uses.
//!
//! # Exactness
//!
//! [`IncrementalSession::predict`] is bit-identical to
//! [`predict_quantized`](crate::predict_quantized) on the same tokens and the
//! same [`InferenceConfig`]. Group accumulators hold
//! `sum over sites in group of activation(site_sum)`, exactly as
//! [`materialize_features_quantized`](crate::materialize_features_quantized)
//! computes it, and both then call
//! [`score_quantized_grouped`](crate::score_quantized_grouped).
//!
//! [`ReversibleTokenJournal`]: crate::ReversibleTokenJournal
//! [`GroupedTokens`]: crate::GroupedTokens

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use crate::{
    Activation, InferenceConfig, ModelError, Pooling, QuantizedCodebookAccess,
    score_quantized_grouped,
};

/// Largest number of token slots a session may hold.
///
/// Quantized embedding components are `i16`, so a site sum is bounded by
/// `slots * 32768` and a group accumulator by `slots_in_group * 32768`.
/// Capping total slots at this value keeps every accumulator inside `i32`
/// without a checked add in the hot loop.
pub const SESSION_MAX_TOKEN_SLOTS: usize = (i32::MAX as usize) / 32_768;

const _: () = assert!(SESSION_MAX_TOKEN_SLOTS == 65_535);

/// One token replacement inside a search frame.
///
/// `lane` indexes the token within its site, so the replaced slot is
/// `site_offsets[site] + lane`.
///
/// The layout is `repr(C)` and matches `Cb2VecTokenDeltaV1` in
/// `include/cb2vec.h` exactly, so the C ABI can borrow a caller's delta array
/// without copying or converting it. `src/ffi.rs` pins that equivalence with
/// compile-time size, alignment, and field-offset assertions.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SessionDelta {
    pub(crate) site: u32,
    pub(crate) lane: u32,
    pub(crate) old: u16,
    pub(crate) new: u16,
}

impl SessionDelta {
    #[inline]
    pub const fn new(site: u32, lane: u32, old_token: u16, new_token: u16) -> Self {
        Self {
            site,
            lane,
            old: old_token,
            new: new_token,
        }
    }

    #[inline]
    pub const fn site(self) -> u32 {
        self.site
    }

    #[inline]
    pub const fn lane(self) -> u32 {
        self.lane
    }

    #[inline]
    pub const fn old_token(self) -> u16 {
        self.old
    }

    #[inline]
    pub const fn new_token(self) -> u16 {
        self.new
    }

    #[inline]
    pub const fn reversed(self) -> Self {
        Self {
            site: self.site,
            lane: self.lane,
            old: self.new,
            new: self.old,
        }
    }
}

/// Fixed capacities chosen once, when a session is created.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SessionLimits {
    /// Largest site count a reset may install.
    pub max_sites: usize,
    /// Largest total token count a reset may install.
    pub max_token_slots: usize,
    /// Largest number of replacements one pushed frame may carry.
    pub max_deltas_per_frame: usize,
    /// Largest number of frames that may be live at once.
    pub max_depth: usize,
}

impl SessionLimits {
    pub const fn new(
        max_sites: usize,
        max_token_slots: usize,
        max_deltas_per_frame: usize,
        max_depth: usize,
    ) -> Self {
        Self {
            max_sites,
            max_token_slots,
            max_deltas_per_frame,
            max_depth,
        }
    }

    fn validate(self) -> Result<usize, SessionError> {
        for (field, value) in [
            ("max_sites", self.max_sites),
            ("max_token_slots", self.max_token_slots),
            ("max_deltas_per_frame", self.max_deltas_per_frame),
            ("max_depth", self.max_depth),
        ] {
            if value == 0 {
                return Err(SessionError::ZeroLimit(field));
            }
        }
        if self.max_token_slots > SESSION_MAX_TOKEN_SLOTS {
            return Err(SessionError::LimitExceeded {
                field: "max_token_slots",
                requested: self.max_token_slots,
                limit: SESSION_MAX_TOKEN_SLOTS,
            });
        }
        self.max_depth
            .checked_mul(self.max_deltas_per_frame)
            .ok_or(SessionError::CapacityOverflow("delta arena"))
    }
}

/// Observable session state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SessionInfo {
    pub site_count: usize,
    pub token_slots: usize,
    pub group_count: usize,
    pub depth: usize,
    pub materialized_depth: usize,
    pub pending_deltas: usize,
    pub limits: SessionLimits,
}

/// Error returned by session construction, reset, or a search operation.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SessionError {
    Model(ModelError),
    ZeroLimit(&'static str),
    CapacityOverflow(&'static str),
    AllocationFailed {
        collection: &'static str,
        requested: usize,
    },
    LimitExceeded {
        field: &'static str,
        requested: usize,
        limit: usize,
    },
    NotReady,
    OffsetTable(&'static str),
    SiteOutOfRange {
        site: usize,
        site_count: usize,
    },
    LaneOutOfRange {
        site: usize,
        lane: usize,
        lane_count: usize,
    },
    TokenOutOfRange {
        token: u16,
        token_count: usize,
    },
    GroupOutOfRange {
        site: usize,
        group: usize,
        group_count: usize,
    },
    EmptyMeanGroup {
        group: usize,
    },
    OldTokenMismatch {
        site: usize,
        lane: usize,
        expected: u16,
        actual: u16,
    },
    DuplicateSlot {
        site: usize,
        lane: usize,
    },
    EmptyStack,
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(error) => write!(f, "invalid model: {error}"),
            Self::ZeroLimit(field) => write!(f, "session {field} must be non-zero"),
            Self::CapacityOverflow(field) => write!(f, "session {field} capacity overflow"),
            Self::AllocationFailed {
                collection,
                requested,
            } => write!(
                f,
                "failed to allocate {requested} elements for session {collection}"
            ),
            Self::LimitExceeded {
                field,
                requested,
                limit,
            } => write!(f, "{field} is {requested}, but the limit is {limit}"),
            Self::NotReady => write!(f, "session has no state; call reset first"),
            Self::OffsetTable(message) => write!(f, "invalid site offset table: {message}"),
            Self::SiteOutOfRange { site, site_count } => {
                write!(f, "site {site} is outside site count {site_count}")
            }
            Self::LaneOutOfRange {
                site,
                lane,
                lane_count,
            } => write!(
                f,
                "site {site} lane {lane} is outside lane count {lane_count}"
            ),
            Self::TokenOutOfRange { token, token_count } => {
                write!(f, "token {token} is outside codebook size {token_count}")
            }
            Self::GroupOutOfRange {
                site,
                group,
                group_count,
            } => write!(
                f,
                "site {site} group {group} is outside group count {group_count}"
            ),
            Self::EmptyMeanGroup { group } => {
                write!(f, "mean pooling group {group} has no sites")
            }
            Self::OldTokenMismatch {
                site,
                lane,
                expected,
                actual,
            } => write!(
                f,
                "site {site} lane {lane} holds {actual}, but the delta expected {expected}"
            ),
            Self::DuplicateSlot { site, lane } => {
                write!(
                    f,
                    "site {site} lane {lane} appears more than once in one frame"
                )
            }
            Self::EmptyStack => write!(f, "session frame stack is empty"),
        }
    }
}

impl Error for SessionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Model(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ModelError> for SessionError {
    fn from(error: ModelError) -> Self {
        Self::Model(error)
    }
}

/// Preallocated incremental evaluator over one immutable quantized model.
///
/// # Threading
///
/// A session is single-owner: every call needs exclusive access. The weights
/// it borrows are immutable, so any number of sessions may share one model.
///
/// # Allocation
///
/// [`IncrementalSession::new`] performs every allocation. [`Self::reset`],
/// [`Self::push`], [`Self::materialize_pending`], [`Self::predict`], and
/// [`Self::pop`] only write into buffers that already exist.
pub struct IncrementalSession<W> {
    weights: W,
    inference: InferenceConfig,
    limits: SessionLimits,
    dim: usize,
    group_count: usize,
    token_count: usize,

    ready: bool,
    site_count: usize,
    token_slots: usize,
    site_offsets: Vec<u32>,
    site_groups: Vec<u32>,
    slot_site: Vec<u32>,
    logical: Vec<u16>,
    site_sums: Vec<i32>,
    group_features: Vec<i32>,
    group_divisors: Vec<usize>,
    group_counts: Vec<usize>,

    deltas: Vec<SessionDelta>,
    frame_ends: Vec<u32>,
    depth: usize,
    materialized_depth: usize,
    delta_len: usize,

    slot_stamp: Vec<u32>,
    site_stamp: Vec<u32>,
    touched: Vec<u32>,
    stamp: u32,
}

impl<W: QuantizedCodebookAccess> IncrementalSession<W> {
    /// Allocates every buffer the search loop will need.
    pub fn new(
        weights: W,
        inference: InferenceConfig,
        limits: SessionLimits,
    ) -> Result<Self, SessionError> {
        let arena_capacity = limits.validate()?;
        let shape = crate::ModelShape::new(
            weights.token_count(),
            weights.group_count(),
            weights.dim(),
            weights.fm_rank(),
        )?;
        if weights.embedding_scale() <= 0 {
            return Err(ModelError::NonPositiveScale("embedding_scale").into());
        }
        if weights.head_scale() <= 0 {
            return Err(ModelError::NonPositiveScale("head_scale").into());
        }
        if weights.factor_scale() <= 0 {
            return Err(ModelError::NonPositiveScale("factor_scale").into());
        }
        if weights.head().len() != shape.feature_len()? {
            return Err(ModelError::LengthMismatch {
                field: "head",
                actual: weights.head().len(),
                expected: shape.feature_len()?,
            }
            .into());
        }
        if weights.factors().len() != shape.factor_len()? {
            return Err(ModelError::LengthMismatch {
                field: "factors",
                actual: weights.factors().len(),
                expected: shape.factor_len()?,
            }
            .into());
        }
        if !weights.bias().is_finite() {
            return Err(ModelError::NonFinite("bias").into());
        }

        let dim = shape.dim();
        let group_count = shape.group_count();
        let site_state_len = limits
            .max_sites
            .checked_mul(dim)
            .ok_or(SessionError::CapacityOverflow("site sums"))?;

        Ok(Self {
            weights,
            inference,
            limits,
            dim,
            group_count,
            token_count: shape.token_count(),

            ready: false,
            site_count: 0,
            token_slots: 0,
            site_offsets: filled(limits.max_sites + 1, 0u32, "site offsets")?,
            site_groups: filled(limits.max_sites, 0u32, "site groups")?,
            slot_site: filled(limits.max_token_slots, 0u32, "slot sites")?,
            logical: filled(limits.max_token_slots, 0u16, "logical tokens")?,
            site_sums: filled(site_state_len, 0i32, "site sums")?,
            group_features: filled(shape.feature_len()?, 0i32, "group features")?,
            group_divisors: filled(group_count, 1usize, "group divisors")?,
            group_counts: filled(group_count, 0usize, "group counts")?,

            deltas: filled(arena_capacity, SessionDelta::default(), "delta arena")?,
            frame_ends: filled(limits.max_depth, 0u32, "frame ends")?,
            depth: 0,
            materialized_depth: 0,
            delta_len: 0,

            slot_stamp: filled(limits.max_token_slots, 0u32, "slot stamps")?,
            site_stamp: filled(limits.max_sites, 0u32, "site stamps")?,
            touched: filled(limits.max_deltas_per_frame, 0u32, "touched sites")?,
            stamp: 0,
        })
    }

    /// Installs a complete position and discards every pushed frame.
    ///
    /// The layout arguments match [`GroupedTokens`](crate::GroupedTokens):
    /// `site_offsets` has `site_count + 1` monotonic entries starting at zero
    /// and ending at `tokens.len()`, and `site_groups` has one group per site.
    /// Nothing is mutated unless every check passes, so a rejected reset keeps
    /// whatever position was already installed.
    pub fn reset(
        &mut self,
        tokens: &[u16],
        site_offsets: &[u32],
        site_groups: &[u32],
    ) -> Result<(), SessionError> {
        let site_count = site_groups.len();
        if site_count > self.limits.max_sites {
            return Err(SessionError::LimitExceeded {
                field: "site_count",
                requested: site_count,
                limit: self.limits.max_sites,
            });
        }
        if tokens.len() > self.limits.max_token_slots {
            return Err(SessionError::LimitExceeded {
                field: "token_slots",
                requested: tokens.len(),
                limit: self.limits.max_token_slots,
            });
        }
        if site_offsets.len() != site_count + 1 {
            return Err(SessionError::OffsetTable(
                "site_offsets must have site_count + 1 entries",
            ));
        }
        if site_offsets[0] != 0 {
            return Err(SessionError::OffsetTable("site_offsets must start at zero"));
        }
        if site_offsets[site_count] as usize != tokens.len() {
            return Err(SessionError::OffsetTable(
                "site_offsets must end at the token count",
            ));
        }
        if site_offsets.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(SessionError::OffsetTable("site_offsets must be monotonic"));
        }
        for (site, &group) in site_groups.iter().enumerate() {
            if group as usize >= self.group_count {
                return Err(SessionError::GroupOutOfRange {
                    site,
                    group: group as usize,
                    group_count: self.group_count,
                });
            }
        }
        for &token in tokens {
            if usize::from(token) >= self.token_count {
                return Err(SessionError::TokenOutOfRange {
                    token,
                    token_count: self.token_count,
                });
            }
        }
        // `group_counts` is scratch, so pooling can be validated before commit.
        self.group_counts.fill(0);
        for &group in site_groups {
            self.group_counts[group as usize] += 1;
        }
        if self.inference.pooling == Pooling::Mean
            && let Some(group) = self.group_counts.iter().position(|&count| count == 0)
        {
            return Err(SessionError::EmptyMeanGroup { group });
        }

        // Validation is complete; commit.
        self.site_count = site_count;
        self.token_slots = tokens.len();
        self.site_offsets[..site_count + 1].copy_from_slice(site_offsets);
        self.site_groups[..site_count].copy_from_slice(site_groups);
        self.logical[..tokens.len()].copy_from_slice(tokens);
        self.depth = 0;
        self.materialized_depth = 0;
        self.delta_len = 0;
        self.slot_stamp[..tokens.len()].fill(0);
        self.site_stamp[..site_count].fill(0);
        self.stamp = 0;

        match self.inference.pooling {
            Pooling::Sum => self.group_divisors.fill(1),
            Pooling::Mean => self.group_divisors.copy_from_slice(&self.group_counts),
        }
        for site in 0..site_count {
            let start = site_offsets[site] as usize;
            let end = site_offsets[site + 1] as usize;
            self.slot_site[start..end].fill(site as u32);
        }

        self.rebuild_numeric_state();
        self.ready = true;
        Ok(())
    }

    fn rebuild_numeric_state(&mut self) {
        let dim = self.dim;
        let activation = self.inference.activation;
        let site_count = self.site_count;
        let Self {
            weights,
            site_offsets,
            site_groups,
            logical,
            site_sums,
            group_features,
            ..
        } = self;

        site_sums[..site_count * dim].fill(0);
        group_features.fill(0);
        for site in 0..site_count {
            let start = site_offsets[site] as usize;
            let end = site_offsets[site + 1] as usize;
            let site_start = site * dim;
            let sums = &mut site_sums[site_start..site_start + dim];
            for &token in &logical[start..end] {
                weights.add_embedding_to(token, sums);
            }
            let group_start = site_groups[site] as usize * dim;
            for component in 0..dim {
                group_features[group_start + component] +=
                    activate(activation, site_sums[site_start + component]);
            }
        }
    }

    /// Records one search move's replacements as a single reversible frame.
    ///
    /// Every delta is checked — site, lane, expected old token, replacement
    /// token, and duplicate slots — before any logical token or depth counter
    /// changes, so a rejected frame leaves the session exactly as it was.
    /// Numeric state is not touched until the frame is materialized.
    pub fn push(&mut self, deltas: &[SessionDelta]) -> Result<usize, SessionError> {
        if !self.ready {
            return Err(SessionError::NotReady);
        }
        if self.depth >= self.limits.max_depth {
            return Err(SessionError::LimitExceeded {
                field: "depth",
                requested: self.depth + 1,
                limit: self.limits.max_depth,
            });
        }
        if deltas.len() > self.limits.max_deltas_per_frame {
            return Err(SessionError::LimitExceeded {
                field: "frame_deltas",
                requested: deltas.len(),
                limit: self.limits.max_deltas_per_frame,
            });
        }

        let stamp = self.next_stamp();
        for delta in deltas {
            let site = delta.site as usize;
            if site >= self.site_count {
                return Err(SessionError::SiteOutOfRange {
                    site,
                    site_count: self.site_count,
                });
            }
            let start = self.site_offsets[site] as usize;
            let lane_count = self.site_offsets[site + 1] as usize - start;
            let lane = delta.lane as usize;
            if lane >= lane_count {
                return Err(SessionError::LaneOutOfRange {
                    site,
                    lane,
                    lane_count,
                });
            }
            if usize::from(delta.new) >= self.token_count {
                return Err(SessionError::TokenOutOfRange {
                    token: delta.new,
                    token_count: self.token_count,
                });
            }
            let slot = start + lane;
            if self.logical[slot] != delta.old {
                return Err(SessionError::OldTokenMismatch {
                    site,
                    lane,
                    expected: delta.old,
                    actual: self.logical[slot],
                });
            }
            if self.slot_stamp[slot] == stamp {
                return Err(SessionError::DuplicateSlot { site, lane });
            }
            self.slot_stamp[slot] = stamp;
        }

        // Validation is complete; commit.
        let start = self.delta_len;
        self.deltas[start..start + deltas.len()].copy_from_slice(deltas);
        for delta in deltas {
            let slot = self.site_offsets[delta.site as usize] as usize + delta.lane as usize;
            self.logical[slot] = delta.new;
        }
        self.delta_len = start + deltas.len();
        self.frame_ends[self.depth] = self.delta_len as u32;
        self.depth += 1;
        Ok(deltas.len())
    }

    /// Applies every pushed-but-unapplied frame to the numeric accumulators.
    pub fn materialize_pending(&mut self) {
        while self.materialized_depth < self.depth {
            let (start, end) = self.frame_range(self.materialized_depth);
            self.replay(start, end, false);
            self.materialized_depth += 1;
        }
    }

    /// Materializes pending frames and scores the current position.
    ///
    /// Bit-identical to [`predict_quantized`](crate::predict_quantized) over
    /// the same tokens, layout, and [`InferenceConfig`].
    pub fn predict(&mut self) -> Result<f32, SessionError> {
        if !self.ready {
            return Err(SessionError::NotReady);
        }
        self.materialize_pending();
        score_quantized_grouped(&self.group_features, &self.weights, &self.group_divisors)
            .map_err(SessionError::Model)
    }

    /// Undoes the most recent frame and returns how many deltas it held.
    pub fn pop(&mut self) -> Result<usize, SessionError> {
        if self.depth == 0 {
            return Err(SessionError::EmptyStack);
        }
        self.depth -= 1;
        let (start, end) = self.frame_range(self.depth);
        if self.materialized_depth > self.depth {
            debug_assert_eq!(
                self.materialized_depth,
                self.depth + 1,
                "materialized frames must form a prefix"
            );
            self.replay(start, end, true);
            self.materialized_depth = self.depth;
        }
        for index in (start..end).rev() {
            let delta = self.deltas[index];
            let slot = self.site_offsets[delta.site as usize] as usize + delta.lane as usize;
            self.logical[slot] = delta.old;
        }
        self.delta_len = start;
        Ok(end - start)
    }

    /// Undoes every pushed frame, restoring the position installed by reset.
    pub fn rewind(&mut self) {
        while self.depth > 0 {
            let _ = self.pop();
        }
    }

    #[inline]
    fn frame_range(&self, frame: usize) -> (usize, usize) {
        let start = if frame == 0 {
            0
        } else {
            self.frame_ends[frame - 1] as usize
        };
        (start, self.frame_ends[frame] as usize)
    }

    /// Applies `deltas[start..end]` to the numeric state.
    ///
    /// Each touched site's activated contribution is removed once, its
    /// embedding sums are updated in place, and the new contribution is added
    /// back. That keeps the result exact under a non-linear activation.
    fn replay(&mut self, start: usize, end: usize, reverse: bool) {
        let dim = self.dim;
        let activation = self.inference.activation;
        let stamp = self.next_stamp();
        let mut touched_len = 0usize;

        let Self {
            weights,
            deltas,
            site_sums,
            group_features,
            site_groups,
            site_stamp,
            touched,
            ..
        } = self;

        for index in start..end {
            let delta = if reverse {
                deltas[start + (end - 1 - index)].reversed()
            } else {
                deltas[index]
            };
            let site = delta.site() as usize;
            let site_start = site * dim;
            if site_stamp[site] != stamp {
                site_stamp[site] = stamp;
                touched[touched_len] = site as u32;
                touched_len += 1;
                let group_start = site_groups[site] as usize * dim;
                for component in 0..dim {
                    group_features[group_start + component] -=
                        activate(activation, site_sums[site_start + component]);
                }
            }
            weights.add_embedding_delta_to(
                delta.old_token(),
                delta.new_token(),
                &mut site_sums[site_start..site_start + dim],
            );
        }

        for &site in &touched[..touched_len] {
            let site = site as usize;
            let site_start = site * dim;
            let group_start = site_groups[site] as usize * dim;
            for component in 0..dim {
                group_features[group_start + component] +=
                    activate(activation, site_sums[site_start + component]);
            }
        }
    }

    /// Issues a fresh generation for the per-slot and per-site scratch stamps.
    ///
    /// Values are monotonic between wraparounds, so a stamp issued for one
    /// array can never collide with a stale value left in the other.
    #[inline]
    fn next_stamp(&mut self) -> u32 {
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.slot_stamp.fill(0);
            self.site_stamp.fill(0);
            self.stamp = 1;
        }
        self.stamp
    }

    /// Immutable weights backing this session.
    #[inline]
    pub fn weights(&self) -> &W {
        &self.weights
    }

    /// Activation and pooling recipe fixed at construction.
    #[inline]
    pub fn inference_config(&self) -> InferenceConfig {
        self.inference
    }

    /// Capacities fixed at construction.
    #[inline]
    pub fn limits(&self) -> SessionLimits {
        self.limits
    }

    /// Whether a position has been installed by [`Self::reset`].
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.ready
    }

    #[inline]
    pub fn depth(&self) -> usize {
        self.depth
    }

    #[inline]
    pub fn materialized_depth(&self) -> usize {
        self.materialized_depth
    }

    #[inline]
    pub fn site_count(&self) -> usize {
        self.site_count
    }

    #[inline]
    pub fn token_slots(&self) -> usize {
        self.token_slots
    }

    /// Current tokens, in the flat slot order installed by [`Self::reset`].
    #[inline]
    pub fn logical_tokens(&self) -> &[u16] {
        &self.logical[..self.token_slots]
    }

    /// Pooled integer group accumulators, valid after materialization.
    #[inline]
    pub fn group_features(&self) -> &[i32] {
        &self.group_features
    }

    /// Pooling divisor for each group.
    #[inline]
    pub fn group_divisors(&self) -> &[usize] {
        &self.group_divisors
    }

    pub fn info(&self) -> SessionInfo {
        SessionInfo {
            site_count: self.site_count,
            token_slots: self.token_slots,
            group_count: self.group_count,
            depth: self.depth,
            materialized_depth: self.materialized_depth,
            pending_deltas: self.delta_len,
            limits: self.limits,
        }
    }
}

impl<W> fmt::Debug for IncrementalSession<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IncrementalSession")
            .field("ready", &self.ready)
            .field("site_count", &self.site_count)
            .field("token_slots", &self.token_slots)
            .field("depth", &self.depth)
            .field("materialized_depth", &self.materialized_depth)
            .field("limits", &self.limits)
            .finish_non_exhaustive()
    }
}

#[inline(always)]
fn activate(activation: Activation, value: i32) -> i32 {
    match activation {
        Activation::Identity => value,
        Activation::Relu => value.max(0),
    }
}

fn filled<T: Clone>(
    len: usize,
    value: T,
    collection: &'static str,
) -> Result<Vec<T>, SessionError> {
    let bytes = size_of::<T>()
        .checked_mul(len)
        .ok_or(SessionError::CapacityOverflow(collection))?;
    if bytes > isize::MAX as usize {
        return Err(SessionError::CapacityOverflow(collection));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| SessionError::AllocationFailed {
            collection,
            requested: len,
        })?;
    values.resize(len, value);
    Ok(values)
}

/// Borrowed access, so a session can hold `&QuantizedCodebookWeights`.
impl<T: QuantizedCodebookAccess + ?Sized> QuantizedCodebookAccess for &T {
    #[inline(always)]
    fn dim(&self) -> usize {
        (**self).dim()
    }

    #[inline(always)]
    fn fm_rank(&self) -> usize {
        (**self).fm_rank()
    }

    #[inline(always)]
    fn embedding_scale(&self) -> i32 {
        (**self).embedding_scale()
    }

    #[inline(always)]
    fn head_scale(&self) -> i32 {
        (**self).head_scale()
    }

    #[inline(always)]
    fn factor_scale(&self) -> i32 {
        (**self).factor_scale()
    }

    #[inline(always)]
    fn bias(&self) -> f32 {
        (**self).bias()
    }

    #[inline(always)]
    fn token_count(&self) -> usize {
        (**self).token_count()
    }

    #[inline(always)]
    fn head(&self) -> &[i16] {
        (**self).head()
    }

    #[inline(always)]
    fn factors(&self) -> &[i16] {
        (**self).factors()
    }

    #[inline(always)]
    fn embedding(&self, token: u16, component: usize) -> i16 {
        (**self).embedding(token, component)
    }

    #[inline(always)]
    fn add_embedding_to(&self, token: u16, out: &mut [i32]) {
        (**self).add_embedding_to(token, out);
    }

    #[inline(always)]
    fn add_embedding_delta_to(&self, old_token: u16, new_token: u16, out: &mut [i32]) {
        (**self).add_embedding_delta_to(old_token, new_token, out);
    }
}

/// Shared ownership, so many sessions can outlive the handle they came from.
impl<T: QuantizedCodebookAccess + ?Sized> QuantizedCodebookAccess for Arc<T> {
    #[inline(always)]
    fn dim(&self) -> usize {
        (**self).dim()
    }

    #[inline(always)]
    fn fm_rank(&self) -> usize {
        (**self).fm_rank()
    }

    #[inline(always)]
    fn embedding_scale(&self) -> i32 {
        (**self).embedding_scale()
    }

    #[inline(always)]
    fn head_scale(&self) -> i32 {
        (**self).head_scale()
    }

    #[inline(always)]
    fn factor_scale(&self) -> i32 {
        (**self).factor_scale()
    }

    #[inline(always)]
    fn bias(&self) -> f32 {
        (**self).bias()
    }

    #[inline(always)]
    fn token_count(&self) -> usize {
        (**self).token_count()
    }

    #[inline(always)]
    fn head(&self) -> &[i16] {
        (**self).head()
    }

    #[inline(always)]
    fn factors(&self) -> &[i16] {
        (**self).factors()
    }

    #[inline(always)]
    fn embedding(&self, token: u16, component: usize) -> i16 {
        (**self).embedding(token, component)
    }

    #[inline(always)]
    fn add_embedding_to(&self, token: u16, out: &mut [i32]) {
        (**self).add_embedding_to(token, out);
    }

    #[inline(always)]
    fn add_embedding_delta_to(&self, old_token: u16, new_token: u16, out: &mut [i32]) {
        (**self).add_embedding_delta_to(old_token, new_token, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CodebookWeights, GroupedTokens, QuantizedCodebookWeights, predict_quantized};

    struct Fixture {
        weights: QuantizedCodebookWeights,
        tokens: Vec<u16>,
        offsets: Vec<u32>,
        groups: Vec<u32>,
    }

    impl Fixture {
        fn new() -> Self {
            Self {
                weights: CodebookWeights::deterministic(9, 3, 4, 2).quantize_i16_s32_s64(),
                //          site 0 | site 1 | site 2 |site 3| site 4
                tokens: vec![0, 3, 5, 1, 8, 2, 7, 4, 6],
                offsets: vec![0, 3, 5, 7, 8, 9],
                groups: vec![0, 1, 1, 2, 0],
            }
        }

        fn limits(&self) -> SessionLimits {
            SessionLimits::new(8, 16, 6, 12)
        }

        fn session(
            &self,
            inference: InferenceConfig,
        ) -> IncrementalSession<&QuantizedCodebookWeights> {
            let mut session =
                IncrementalSession::new(&self.weights, inference, self.limits()).unwrap();
            session
                .reset(&self.tokens, &self.offsets, &self.groups)
                .unwrap();
            session
        }

        fn rebuild(&self, tokens: &[u16], inference: InferenceConfig) -> f32 {
            let input = GroupedTokens::new(
                tokens.to_vec(),
                self.offsets.iter().map(|&value| value as usize).collect(),
                self.groups.iter().map(|&value| value as usize).collect(),
            )
            .unwrap();
            predict_quantized(&input, &self.weights, inference).unwrap()
        }
    }

    fn configs() -> [InferenceConfig; 4] {
        [
            InferenceConfig::new(Activation::Relu, Pooling::Mean),
            InferenceConfig::new(Activation::Relu, Pooling::Sum),
            InferenceConfig::new(Activation::Identity, Pooling::Mean),
            InferenceConfig::new(Activation::Identity, Pooling::Sum),
        ]
    }

    #[test]
    fn incremental_score_matches_full_rebuild_bitwise() {
        let fixture = Fixture::new();
        for inference in configs() {
            let mut session = fixture.session(inference);
            assert_eq!(
                session.predict().unwrap().to_bits(),
                fixture.rebuild(&fixture.tokens, inference).to_bits(),
                "initial position for {inference:?}"
            );

            // Two lanes at one site plus a lane at another site, in one frame.
            session
                .push(&[
                    SessionDelta::new(0, 0, 0, 6),
                    SessionDelta::new(0, 2, 5, 1),
                    SessionDelta::new(2, 1, 7, 0),
                ])
                .unwrap();
            let expected = [6, 3, 1, 1, 8, 2, 0, 4, 6];
            assert_eq!(session.logical_tokens(), expected);
            assert_eq!(
                session.predict().unwrap().to_bits(),
                fixture.rebuild(&expected, inference).to_bits(),
                "after push for {inference:?}"
            );

            session.push(&[SessionDelta::new(4, 0, 6, 8)]).unwrap();
            let expected = [6, 3, 1, 1, 8, 2, 0, 4, 8];
            assert_eq!(
                session.predict().unwrap().to_bits(),
                fixture.rebuild(&expected, inference).to_bits(),
                "after second push for {inference:?}"
            );

            session.pop().unwrap();
            session.pop().unwrap();
            assert_eq!(session.logical_tokens(), fixture.tokens);
            assert_eq!(
                session.predict().unwrap().to_bits(),
                fixture.rebuild(&fixture.tokens, inference).to_bits(),
                "after pop for {inference:?}"
            );
        }
    }

    #[test]
    fn long_random_push_pop_sequence_restores_the_initial_state() {
        let fixture = Fixture::new();
        let inference = InferenceConfig::new(Activation::Relu, Pooling::Mean);
        let mut session = fixture.session(inference);
        let baseline = session.predict().unwrap().to_bits();
        let baseline_features = session.group_features().to_vec();

        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let mut live: Vec<Vec<SessionDelta>> = Vec::new();
        let mut current = fixture.tokens.clone();
        let mut frame = Vec::new();
        for step in 0..4_000 {
            let pop =
                live.len() == session.limits().max_depth || (!live.is_empty() && next() % 100 < 45);
            if pop {
                let popped = live.pop().unwrap();
                assert_eq!(session.pop().unwrap(), popped.len());
                for delta in popped.iter().rev() {
                    let slot =
                        fixture.offsets[delta.site() as usize] as usize + delta.lane() as usize;
                    current[slot] = delta.old_token();
                }
            } else {
                frame.clear();
                let mut used = [false; 16];
                let count = 1 + (next() % session.limits().max_deltas_per_frame as u64) as usize;
                for _ in 0..count {
                    let site = (next() % fixture.groups.len() as u64) as usize;
                    let lanes = (fixture.offsets[site + 1] - fixture.offsets[site]) as usize;
                    let lane = (next() % lanes as u64) as usize;
                    let slot = fixture.offsets[site] as usize + lane;
                    if used[slot] {
                        continue;
                    }
                    used[slot] = true;
                    let new = (next() % 9) as u16;
                    frame.push(SessionDelta::new(
                        site as u32,
                        lane as u32,
                        current[slot],
                        new,
                    ));
                }
                session.push(&frame).unwrap();
                for delta in &frame {
                    let slot =
                        fixture.offsets[delta.site() as usize] as usize + delta.lane() as usize;
                    current[slot] = delta.new_token();
                }
                live.push(frame.clone());
            }

            assert_eq!(session.logical_tokens(), current.as_slice());
            // Score often enough to exercise interleaved materialization.
            if step % 3 == 0 {
                assert_eq!(
                    session.predict().unwrap().to_bits(),
                    fixture.rebuild(&current, inference).to_bits(),
                    "step {step}"
                );
            }
        }

        session.rewind();
        assert_eq!(session.depth(), 0);
        assert_eq!(session.materialized_depth(), 0);
        assert_eq!(session.logical_tokens(), fixture.tokens);
        assert_eq!(session.predict().unwrap().to_bits(), baseline);
        assert_eq!(session.group_features(), baseline_features);
    }

    #[test]
    fn rejected_frames_leave_the_session_unchanged() {
        let fixture = Fixture::new();
        let inference = InferenceConfig::new(Activation::Relu, Pooling::Mean);
        let mut session = fixture.session(inference);
        session.predict().unwrap();
        let before_score = session.predict().unwrap().to_bits();
        let before_features = session.group_features().to_vec();

        let invalid = [
            (
                vec![SessionDelta::new(0, 0, 4, 1)],
                SessionError::OldTokenMismatch {
                    site: 0,
                    lane: 0,
                    expected: 4,
                    actual: 0,
                },
            ),
            (
                vec![SessionDelta::new(9, 0, 0, 1)],
                SessionError::SiteOutOfRange {
                    site: 9,
                    site_count: 5,
                },
            ),
            (
                vec![SessionDelta::new(3, 2, 4, 1)],
                SessionError::LaneOutOfRange {
                    site: 3,
                    lane: 2,
                    lane_count: 1,
                },
            ),
            (
                vec![SessionDelta::new(0, 0, 0, 99)],
                SessionError::TokenOutOfRange {
                    token: 99,
                    token_count: 9,
                },
            ),
            (
                vec![SessionDelta::new(0, 0, 0, 1), SessionDelta::new(0, 0, 0, 2)],
                SessionError::DuplicateSlot { site: 0, lane: 0 },
            ),
            // A valid leading delta followed by an invalid one must not apply.
            (
                vec![SessionDelta::new(1, 0, 1, 2), SessionDelta::new(1, 9, 0, 2)],
                SessionError::LaneOutOfRange {
                    site: 1,
                    lane: 9,
                    lane_count: 2,
                },
            ),
        ];

        for (frame, expected) in invalid {
            assert_eq!(session.push(&frame), Err(expected));
            assert_eq!(session.depth(), 0);
            assert_eq!(session.logical_tokens(), fixture.tokens);
            assert_eq!(session.group_features(), before_features);
            assert_eq!(session.predict().unwrap().to_bits(), before_score);
        }

        // A rejected frame must not consume depth or corrupt later frames.
        session.push(&[SessionDelta::new(0, 0, 0, 1)]).unwrap();
        assert_eq!(session.depth(), 1);
        assert_eq!(
            session.predict().unwrap().to_bits(),
            fixture
                .rebuild(&[1, 3, 5, 1, 8, 2, 7, 4, 6], inference)
                .to_bits()
        );
        session.pop().unwrap();
        assert_eq!(session.predict().unwrap().to_bits(), before_score);
    }

    #[test]
    fn depth_and_frame_limits_are_reported_and_recoverable() {
        let fixture = Fixture::new();
        let inference = InferenceConfig::new(Activation::Identity, Pooling::Sum);
        let mut session = fixture.session(inference);
        let limits = session.limits();

        for step in 0..limits.max_depth {
            let token = (step % 9) as u16;
            let old = session.logical_tokens()[0];
            session
                .push(&[SessionDelta::new(0, 0, old, token)])
                .unwrap();
        }
        let old = session.logical_tokens()[0];
        assert_eq!(
            session.push(&[SessionDelta::new(0, 0, old, 1)]),
            Err(SessionError::LimitExceeded {
                field: "depth",
                requested: limits.max_depth + 1,
                limit: limits.max_depth,
            })
        );
        assert_eq!(session.depth(), limits.max_depth);
        // Still usable at the ceiling, and popping frees exactly one slot.
        session.predict().unwrap();
        session.pop().unwrap();
        let old = session.logical_tokens()[0];
        session.push(&[SessionDelta::new(0, 0, old, 1)]).unwrap();
        assert_eq!(session.depth(), limits.max_depth);
        session.rewind();
        assert_eq!(session.logical_tokens(), fixture.tokens);

        let oversized: Vec<SessionDelta> = (0..limits.max_deltas_per_frame + 1)
            .map(|lane| SessionDelta::new(0, (lane % 3) as u32, 0, 1))
            .collect();
        assert_eq!(
            session.push(&oversized),
            Err(SessionError::LimitExceeded {
                field: "frame_deltas",
                requested: limits.max_deltas_per_frame + 1,
                limit: limits.max_deltas_per_frame,
            })
        );
        assert_eq!(session.pop(), Err(SessionError::EmptyStack));
    }

    #[test]
    fn empty_frames_and_zero_lane_sites_are_supported() {
        let weights = CodebookWeights::deterministic(4, 2, 2, 0).quantize_i16_s32_s64();
        let inference = InferenceConfig::new(Activation::Relu, Pooling::Mean);
        let mut session =
            IncrementalSession::new(&weights, inference, SessionLimits::new(4, 4, 4, 4)).unwrap();
        // Site 1 owns no tokens.
        session.reset(&[1, 2], &[0, 1, 1, 2], &[0, 1, 1]).unwrap();
        let before = session.predict().unwrap().to_bits();
        assert_eq!(session.push(&[]).unwrap(), 0);
        assert_eq!(session.depth(), 1);
        assert_eq!(session.predict().unwrap().to_bits(), before);
        assert_eq!(session.pop().unwrap(), 0);
        assert_eq!(
            session.push(&[SessionDelta::new(1, 0, 0, 1)]),
            Err(SessionError::LaneOutOfRange {
                site: 1,
                lane: 0,
                lane_count: 0,
            })
        );
    }

    #[test]
    fn many_sessions_share_one_model() {
        let fixture = Fixture::new();
        let shared = Arc::new(fixture.weights.clone());
        let inference = InferenceConfig::new(Activation::Relu, Pooling::Mean);
        let mut sessions: Vec<IncrementalSession<Arc<QuantizedCodebookWeights>>> = (0..4)
            .map(|_| {
                let mut session =
                    IncrementalSession::new(Arc::clone(&shared), inference, fixture.limits())
                        .unwrap();
                session
                    .reset(&fixture.tokens, &fixture.offsets, &fixture.groups)
                    .unwrap();
                session
            })
            .collect();

        // Diverge each session, then confirm none observed another's writes.
        for (index, session) in sessions.iter_mut().enumerate() {
            session
                .push(&[SessionDelta::new(0, 0, 0, index as u16 + 1)])
                .unwrap();
        }
        for (index, session) in sessions.iter_mut().enumerate() {
            let mut expected = fixture.tokens.clone();
            expected[0] = index as u16 + 1;
            assert_eq!(
                session.predict().unwrap().to_bits(),
                fixture.rebuild(&expected, inference).to_bits()
            );
        }
        assert_eq!(Arc::strong_count(&shared), 5);
        drop(sessions);
        assert_eq!(Arc::strong_count(&shared), 1);
    }

    #[test]
    fn reset_validates_before_replacing_state() {
        let fixture = Fixture::new();
        let inference = InferenceConfig::new(Activation::Relu, Pooling::Mean);
        let mut session = fixture.session(inference);
        let before = session.predict().unwrap().to_bits();

        assert!(matches!(
            session.reset(&[0, 1], &[0, 2], &[9]),
            Err(SessionError::GroupOutOfRange { group: 9, .. })
        ));
        assert!(matches!(
            session.reset(&[0, 99], &[0, 2], &[0]),
            Err(SessionError::TokenOutOfRange { token: 99, .. })
        ));
        assert!(matches!(
            session.reset(&[0, 1], &[0, 1], &[0]),
            Err(SessionError::OffsetTable(_))
        ));
        assert!(matches!(
            session.reset(&[0; 20], &[0, 20], &[0]),
            Err(SessionError::LimitExceeded {
                field: "token_slots",
                ..
            })
        ));
        assert_eq!(session.predict().unwrap().to_bits(), before);

        // Mean pooling needs every model group populated, and rejecting that
        // must still leave the previously installed position intact.
        assert_eq!(
            session.reset(&[0, 1], &[0, 1, 2], &[0, 1]),
            Err(SessionError::EmptyMeanGroup { group: 2 })
        );
        assert!(session.is_ready());
        assert_eq!(session.logical_tokens(), fixture.tokens);
        assert_eq!(session.predict().unwrap().to_bits(), before);

        // A session that has never been reset refuses to score.
        let mut fresh =
            IncrementalSession::new(&fixture.weights, inference, fixture.limits()).unwrap();
        assert!(!fresh.is_ready());
        assert_eq!(fresh.predict(), Err(SessionError::NotReady));
        assert_eq!(
            fresh.push(&[SessionDelta::new(0, 0, 0, 1)]),
            Err(SessionError::NotReady)
        );

        // Sum pooling tolerates empty groups, and reset clears the stack.
        let mut session = IncrementalSession::new(
            &fixture.weights,
            InferenceConfig::new(Activation::Relu, Pooling::Sum),
            fixture.limits(),
        )
        .unwrap();
        session.reset(&[0, 1], &[0, 1, 2], &[0, 1]).unwrap();
        session.push(&[SessionDelta::new(0, 0, 0, 2)]).unwrap();
        session
            .reset(&fixture.tokens, &fixture.offsets, &fixture.groups)
            .unwrap();
        assert_eq!(session.depth(), 0);
        assert_eq!(session.info().pending_deltas, 0);
        assert_eq!(session.logical_tokens(), fixture.tokens);
    }

    #[test]
    fn invalid_limits_and_models_fail_closed() {
        let fixture = Fixture::new();
        let inference = InferenceConfig::new(Activation::Relu, Pooling::Mean);
        assert_eq!(
            IncrementalSession::new(&fixture.weights, inference, SessionLimits::new(0, 1, 1, 1))
                .err(),
            Some(SessionError::ZeroLimit("max_sites"))
        );
        assert_eq!(
            IncrementalSession::new(
                &fixture.weights,
                inference,
                SessionLimits::new(1, SESSION_MAX_TOKEN_SLOTS + 1, 1, 1),
            )
            .err(),
            Some(SessionError::LimitExceeded {
                field: "max_token_slots",
                requested: SESSION_MAX_TOKEN_SLOTS + 1,
                limit: SESSION_MAX_TOKEN_SLOTS,
            })
        );

        let mut broken = fixture.weights.clone();
        broken.head_scale = 0;
        assert_eq!(
            IncrementalSession::new(&broken, inference, SessionLimits::new(1, 1, 1, 1)).err(),
            Some(SessionError::Model(ModelError::NonPositiveScale(
                "head_scale"
            )))
        );
    }

    #[test]
    fn the_search_loop_does_not_allocate() {
        let fixture = Fixture::new();
        let inference = InferenceConfig::new(Activation::Relu, Pooling::Mean);
        let mut session = fixture.session(inference);
        let frame = [SessionDelta::new(0, 0, 0, 4), SessionDelta::new(2, 0, 2, 7)];

        // Warm up anything the first call would lazily initialize.
        for _ in 0..4 {
            session.push(&frame).unwrap();
            session.predict().unwrap();
            session.pop().unwrap();
        }

        let guard = crate::testing::AllocationGuard::new();
        for _ in 0..1_000 {
            session.push(&frame).unwrap();
            let score = session.predict().unwrap();
            assert!(score.is_finite());
            session.pop().unwrap();
        }
        // reset must also stay inside its preallocated buffers.
        session
            .reset(&fixture.tokens, &fixture.offsets, &fixture.groups)
            .unwrap();
        guard.assert_no_allocations("session search loop");
    }
}
