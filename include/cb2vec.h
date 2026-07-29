#ifndef CB2VEC_H
#define CB2VEC_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define CB2VEC_API __declspec(dllimport)
#define CB2VEC_CALL __cdecl
#else
#define CB2VEC_API
#define CB2VEC_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * CB2Vec C ABI 1.1
 *
 * Version axes are independent:
 *   - crate/library release: 0.3.0
 *   - binary artifact format: 1 and 2
 *   - trainer checkpoint format: 1
 *   - C ABI: 1.1 (0x00010001)
 *
 * ABI 1.1 is a strictly additive revision of 1.0. Every 1.0 symbol keeps its
 * signature and behavior, and every versioned struct still accepts
 * CB2VEC_ABI_VERSION_1_0 in its abi_version field, so a consumer built
 * against 1.0 links and runs unchanged. Check the major version only:
 *
 *     if ((cb2vec_abi_version() >> 16) != 1) { bail(); }
 *
 * Caller-owned arrays are borrowed only for the duration of a call. A pointer
 * may be NULL only when its corresponding element count is zero, or where the
 * documentation says NULL is an explicit opt-out. Handles are library-owned
 * and must be released exactly once by their matching free function. Do not
 * free a handle while another thread is using it.
 *
 * Threading and lifetimes
 * -----------------------
 * Cb2VecModel is immutable; its prediction calls may run concurrently.
 *
 * Cb2VecSession is single-owner: every call on one session, including
 * cb2vec_session_get_info_v1, must come from one thread at a time. Sessions
 * do NOT share mutable state, so the intended search topology is one
 * immutable model shared by many sessions, one session per search thread.
 *
 * A session holds a reference to its model's weights. Freeing the model
 * handle while sessions are alive is therefore safe and defined: the weights
 * stay alive until the last session is freed. Handles may be freed in any
 * order.
 *
 * Every call on a given Cb2VecTrainer, including read-only calls, must be
 * externally serialized. Invalid/dangling handles, double-free, undersized
 * allocations, and racing free calls are caller undefined behavior.
 */

typedef int32_t Cb2VecStatus;
typedef struct Cb2VecTrainer Cb2VecTrainer;
typedef struct Cb2VecModel Cb2VecModel;
typedef struct Cb2VecSession Cb2VecSession;

#define CB2VEC_ABI_VERSION UINT32_C(0x00010001)
#define CB2VEC_ABI_VERSION_1_0 UINT32_C(0x00010000)

#define CB2VEC_OK INT32_C(0)
#define CB2VEC_ERROR_NULL_POINTER INT32_C(-1)
#define CB2VEC_ERROR_INVALID_ARGUMENT INT32_C(-2)
#define CB2VEC_ERROR_ABI_MISMATCH INT32_C(-3)
#define CB2VEC_ERROR_ARTIFACT INT32_C(-4)
#define CB2VEC_ERROR_MODEL INT32_C(-5)
#define CB2VEC_ERROR_NUMERIC INT32_C(-6)
#define CB2VEC_ERROR_BUFFER_TOO_SMALL INT32_C(-7)
/* Added in ABI 1.1. */
#define CB2VEC_ERROR_LIMIT_EXCEEDED INT32_C(-8)
#define CB2VEC_ERROR_STATE INT32_C(-9)
#define CB2VEC_ERROR_CHECKPOINT INT32_C(-10)
#define CB2VEC_ERROR_OUT_OF_MEMORY INT32_C(-11)
#define CB2VEC_ERROR_PANIC INT32_C(-127)

#define CB2VEC_ACTIVATION_IDENTITY UINT32_C(0)
#define CB2VEC_ACTIVATION_RELU UINT32_C(1)
#define CB2VEC_POOLING_SUM UINT32_C(0)
#define CB2VEC_POOLING_MEAN UINT32_C(1)
#define CB2VEC_LOSS_BCE_WITH_LOGITS UINT32_C(0)
#define CB2VEC_LOSS_MSE UINT32_C(1)
#define CB2VEC_MODEL_KIND_FLAT UINT32_C(0)
#define CB2VEC_MODEL_KIND_FACTORED UINT32_C(1)
#define CB2VEC_MODEL_KIND_FP32 UINT32_C(2)
#define CB2VEC_MODEL_FLAG_LEGACY_MAGIC UINT32_C(1)
#define CB2VEC_MODEL_FLAG_FLATTENED_AT_LOAD UINT32_C(2)

#define CB2VEC_ARTIFACT_VERSION UINT32_C(1)
#define CB2VEC_ARTIFACT_VERSION_V2 UINT32_C(2)
#define CB2VEC_CHECKPOINT_VERSION UINT32_C(1)
/*
 * A session's token slots must fit an i32 accumulator without a checked add
 * in the hot loop: |embedding| <= 32768, so slots * 32768 <= INT32_MAX.
 */
#define CB2VEC_SESSION_MAX_TOKEN_SLOTS UINT32_C(65535)

typedef struct Cb2VecModelShapeV1 {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t token_count;
    uint32_t group_count;
    uint32_t dim;
    uint32_t fm_rank;
    uint32_t reserved[2];
} Cb2VecModelShapeV1;

typedef struct Cb2VecTrainerConfigV1 {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t activation;
    uint32_t pooling;
    uint32_t loss;
    uint32_t batch_size;
    uint32_t shuffle;
    uint32_t flags;
    uint64_t seed;
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    uint32_t reserved[2];
} Cb2VecTrainerConfigV1;

typedef struct Cb2VecQuantizationConfigV1 {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t embedding_scale;
    int32_t head_scale;
    int32_t factor_scale;
    uint32_t flags;
    uint32_t reserved[2];
} Cb2VecQuantizationConfigV1;

typedef struct Cb2VecInferenceConfigV1 {
    uint32_t struct_size;
    uint32_t activation;
    uint32_t pooling;
    uint32_t flags;
} Cb2VecInferenceConfigV1;

/*
 * A flattened batch concatenates all samples' sites and tokens.
 *
 * site_token_offsets has site_count + 1 entries, starts at zero, is
 * monotonic, and ends at tokens_len. sample_site_offsets has sample_count + 1
 * entries, starts at zero, is monotonic, and ends at site_count. targets has
 * sample_count values. weights is either NULL (unit weights) or has
 * sample_count positive finite values.
 */
typedef struct Cb2VecTrainingBatchV1 {
    uint32_t struct_size;
    uint32_t flags;
    const uint16_t *tokens;
    const uint32_t *site_token_offsets;
    const uint32_t *site_groups;
    const uint32_t *sample_site_offsets;
    const float *targets;
    const float *weights;
    uint32_t tokens_len;
    uint32_t site_count;
    uint32_t sample_count;
    uint32_t reserved;
} Cb2VecTrainingBatchV1;

typedef struct Cb2VecTrainingMetricsV1 {
    uint32_t struct_size;
    uint32_t abi_version;
    float mean_loss;
    uint32_t reserved;
    double total_weight;
    uint64_t sample_count;
    uint64_t batch_count;
    uint64_t optimizer_step;
    uint64_t completed_epochs;
    uint64_t reserved_tail;
} Cb2VecTrainingMetricsV1;

typedef struct Cb2VecModelInfoV1 {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t artifact_version;
    uint32_t flags;
    uint32_t token_count;
    uint32_t group_count;
    uint32_t dim;
    uint32_t fm_rank;
    uint32_t kind;
    uint32_t activation;
    uint32_t pooling;
    int32_t embedding_scale;
    int32_t head_scale;
    int32_t factor_scale;
    uint32_t reserved[2];
} Cb2VecModelInfoV1;

/* ---- ABI 1.1 additions ---- */

/*
 * Fixed capacities for one incremental search session. Every buffer the
 * search loop needs is allocated once, from these numbers. Exceeding one at
 * run time returns CB2VEC_ERROR_LIMIT_EXCEEDED rather than reallocating.
 *
 * max_token_slots must not exceed CB2VEC_SESSION_MAX_TOKEN_SLOTS.
 */
typedef struct Cb2VecSessionConfigV1 {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t max_sites;
    uint32_t max_token_slots;
    uint32_t max_deltas_per_frame;
    uint32_t max_depth;
    uint32_t flags;
    uint32_t reserved[1];
} Cb2VecSessionConfigV1;

/*
 * One token replacement in a pushed search frame.
 *
 * site indexes the site table installed by cb2vec_session_reset_v1, lane
 * indexes the token within that site, and old_token must equal the token the
 * session currently holds there. A frame may not touch the same (site, lane)
 * twice.
 */
typedef struct Cb2VecTokenDeltaV1 {
    uint32_t site;
    uint32_t lane;
    uint16_t old_token;
    uint16_t new_token;
} Cb2VecTokenDeltaV1;

typedef struct Cb2VecSessionInfoV1 {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t site_count;
    uint32_t token_slots;
    uint32_t group_count;
    uint32_t depth;
    uint32_t materialized_depth;
    uint32_t pending_deltas;
    uint32_t max_sites;
    uint32_t max_token_slots;
    uint32_t max_deltas_per_frame;
    uint32_t max_depth;
    uint32_t activation;
    uint32_t pooling;
    uint32_t flags;
    uint32_t reserved[1];
} Cb2VecSessionInfoV1;

/*
 * Consumer-defined identity of the token vocabulary a model was trained
 * against. CB2Vec never interprets these values; they exist so an application
 * can refuse a model whose schema no longer matches its own code. Version 0
 * with an all-zero digest means "unspecified".
 */
typedef struct Cb2VecArtifactMetadataV1 {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t schema_version;
    uint32_t flags;
    uint8_t schema_digest[16];
} Cb2VecArtifactMetadataV1;

/* Everything readable from artifact bytes without building a model. */
typedef struct Cb2VecArtifactInfoV1 {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t artifact_version;
    uint32_t kind;
    uint32_t token_count;
    uint32_t group_count;
    uint32_t dim;
    uint32_t fm_rank;
    uint32_t has_inference_config;
    uint32_t activation;
    uint32_t pooling;
    uint32_t schema_version;
    int32_t embedding_scale;
    int32_t head_scale;
    int32_t factor_scale;
    uint32_t flags;
    uint8_t source_sha256[32];
    uint8_t schema_digest[16];
} Cb2VecArtifactInfoV1;

CB2VEC_API uint32_t CB2VEC_CALL cb2vec_abi_version(void);
CB2VEC_API const char *CB2VEC_CALL cb2vec_library_version(void);
CB2VEC_API const char *CB2VEC_CALL cb2vec_last_error(void);

CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_model_shape_default_v1(
    Cb2VecModelShapeV1 *out_shape);
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_trainer_config_default_v1(
    Cb2VecTrainerConfigV1 *out_config);
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_quantization_config_default_v1(
    Cb2VecQuantizationConfigV1 *out_config);
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_inference_config_default_v1(
    Cb2VecInferenceConfigV1 *out_config);

CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_trainer_create_v1(
    const Cb2VecModelShapeV1 *shape,
    const Cb2VecTrainerConfigV1 *config,
    Cb2VecTrainer **out_trainer);
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_trainer_load_artifact_v1(
    const uint8_t *artifact,
    uint32_t artifact_len,
    const Cb2VecTrainerConfigV1 *config,
    Cb2VecTrainer **out_trainer);
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_trainer_get_info_v1(
    const Cb2VecTrainer *trainer,
    Cb2VecModelInfoV1 *out_info);
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_trainer_predict_logit_v1(
    const Cb2VecTrainer *trainer,
    const uint16_t *tokens,
    uint32_t tokens_len,
    const uint32_t *site_offsets,
    const uint32_t *site_groups,
    uint32_t site_count,
    float *out_score);
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_trainer_predict_probability_v1(
    const Cb2VecTrainer *trainer,
    const uint16_t *tokens,
    uint32_t tokens_len,
    const uint32_t *site_offsets,
    const uint32_t *site_groups,
    uint32_t site_count,
    float *out_probability);
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_trainer_evaluate_v1(
    const Cb2VecTrainer *trainer,
    const Cb2VecTrainingBatchV1 *batch,
    Cb2VecTrainingMetricsV1 *out_metrics);
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_trainer_train_batch_v1(
    Cb2VecTrainer *trainer,
    const Cb2VecTrainingBatchV1 *batch,
    Cb2VecTrainingMetricsV1 *out_metrics);
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_trainer_train_epoch_v1(
    Cb2VecTrainer *trainer,
    const Cb2VecTrainingBatchV1 *dataset,
    Cb2VecTrainingMetricsV1 *out_metrics);
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_trainer_quantize_v1(
    const Cb2VecTrainer *trainer,
    const Cb2VecQuantizationConfigV1 *quantization,
    Cb2VecModel **out_model);
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_trainer_write_artifact_v1(
    const Cb2VecTrainer *trainer,
    const Cb2VecQuantizationConfigV1 *quantization,
    const uint8_t *source_sha256_32,
    uint8_t *out_bytes,
    uint32_t out_capacity,
    uint32_t *out_required_or_written);
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_trainer_free_v1(
    Cb2VecTrainer *trainer);

CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_model_load_v1(
    const uint8_t *artifact,
    uint32_t artifact_len,
    const Cb2VecInferenceConfigV1 *inference,
    Cb2VecModel **out_model);
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_model_get_info_v1(
    const Cb2VecModel *model,
    Cb2VecModelInfoV1 *out_info);
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_model_predict_v1(
    const Cb2VecModel *model,
    const uint16_t *tokens,
    uint32_t tokens_len,
    const uint32_t *site_offsets,
    const uint32_t *site_groups,
    uint32_t site_count,
    float *out_score);
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_model_predict_batch_v1(
    const Cb2VecModel *model,
    const Cb2VecTrainingBatchV1 *batch,
    float *out_scores,
    uint32_t out_scores_len);
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_model_free_v1(
    Cb2VecModel *model);

/* ---- ABI 1.1: incremental search sessions ---- */

CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_session_config_default_v1(
    Cb2VecSessionConfigV1 *out_config);
/*
 * Creates a session over an immutable model. The session shares ownership of
 * the model's weights, so the model handle may be freed first. The session is
 * not scorable until cb2vec_session_reset_v1 installs a position.
 */
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_session_create_v1(
    const Cb2VecModel *model,
    const Cb2VecSessionConfigV1 *config,
    Cb2VecSession **out_session);
/*
 * Installs a complete position and discards every pushed frame. site_offsets
 * has site_count + 1 monotonic entries starting at zero and ending at
 * tokens_len; site_groups has one group index per site. Nothing changes
 * unless every check passes.
 */
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_session_reset_v1(
    Cb2VecSession *session,
    const uint16_t *tokens,
    uint32_t tokens_len,
    const uint32_t *site_offsets,
    const uint32_t *site_groups,
    uint32_t site_count);
/*
 * Pushes one search move's replacements as a single reversible frame. Every
 * delta is validated first, so a rejected frame changes nothing and does not
 * consume depth. delta_count may be zero, which still pushes a frame and
 * keeps push/pop balanced for a null move.
 */
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_session_push_v1(
    Cb2VecSession *session,
    const Cb2VecTokenDeltaV1 *deltas,
    uint32_t delta_count);
/* Applies pending frames to the numeric state; predict does this implicitly. */
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_session_materialize_v1(
    Cb2VecSession *session);
/*
 * Scores the current position. Bit-identical to cb2vec_model_predict_v1 over
 * the same tokens with the same inference recipe.
 */
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_session_predict_v1(
    Cb2VecSession *session,
    float *out_score);
/*
 * Undoes the most recent frame. out_popped_deltas may be NULL. Popping with
 * no frames left returns CB2VEC_ERROR_STATE.
 */
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_session_pop_v1(
    Cb2VecSession *session,
    uint32_t *out_popped_deltas);
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_session_get_info_v1(
    const Cb2VecSession *session,
    Cb2VecSessionInfoV1 *out_info);
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_session_free_v1(
    Cb2VecSession *session);

/* ---- ABI 1.1: trainer checkpoints ---- */

/*
 * Unlike an inference artifact, a checkpoint restores Adam moments, the
 * optimizer step, the shuffle RNG, and completed epochs, so a resumed run is
 * bit-identical to an uninterrupted one. The trainer config travels with the
 * file. A NULL out_bytes with zero capacity is a size probe that returns
 * CB2VEC_ERROR_BUFFER_TOO_SMALL.
 */
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_trainer_checkpoint_len_v1(
    const Cb2VecTrainer *trainer,
    uint32_t *out_len);
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_trainer_write_checkpoint_v1(
    const Cb2VecTrainer *trainer,
    uint8_t *out_bytes,
    uint32_t out_capacity,
    uint32_t *out_required_or_written);
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_trainer_load_checkpoint_v1(
    const uint8_t *checkpoint,
    uint32_t checkpoint_len,
    Cb2VecTrainer **out_trainer);

/* ---- ABI 1.1: artifact version 2 ---- */

/* Reads artifact metadata without building a model. */
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_artifact_probe_v1(
    const uint8_t *artifact,
    uint32_t artifact_len,
    Cb2VecArtifactInfoV1 *out_info);
/*
 * Writes an artifact that stores its own activation and pooling, taken from
 * the trainer, so a consumer cannot load it with the wrong recipe. metadata
 * may be NULL to record no schema identity.
 */
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_trainer_write_artifact_v2(
    const Cb2VecTrainer *trainer,
    const Cb2VecQuantizationConfigV1 *quantization,
    const uint8_t *source_sha256_32,
    const Cb2VecArtifactMetadataV1 *metadata,
    uint8_t *out_bytes,
    uint32_t out_capacity,
    uint32_t *out_required_or_written);
/*
 * Loads a model, preferring the recipe stored in a version-2 artifact.
 *
 * inference may be NULL. When it is not, a version-2 artifact whose stored
 * recipe disagrees returns CB2VEC_ERROR_ARTIFACT instead of silently scoring
 * with the wrong activation or pooling. A version-1 artifact still requires a
 * non-NULL inference.
 *
 * expected_schema may be NULL. When it is not, an artifact that carries a
 * schema identity must match it exactly; artifacts that carry none are
 * accepted.
 */
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_model_load_v2(
    const uint8_t *artifact,
    uint32_t artifact_len,
    const Cb2VecInferenceConfigV1 *inference,
    const Cb2VecArtifactMetadataV1 *expected_schema,
    Cb2VecModel **out_model);
CB2VEC_API Cb2VecStatus CB2VEC_CALL cb2vec_model_get_metadata_v1(
    const Cb2VecModel *model,
    Cb2VecArtifactMetadataV1 *out_metadata);

#ifdef __cplusplus
} /* extern "C" */
#endif

#if defined(__cplusplus) && __cplusplus >= 201103L
#define CB2VEC_STATIC_ASSERT(condition, message) static_assert(condition, message)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define CB2VEC_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#else
#define CB2VEC_STATIC_ASSERT(condition, message)
#endif

CB2VEC_STATIC_ASSERT(sizeof(Cb2VecModelShapeV1) == 32,
                     "Cb2VecModelShapeV1 ABI mismatch");
CB2VEC_STATIC_ASSERT(sizeof(Cb2VecTrainerConfigV1) == 64,
                     "Cb2VecTrainerConfigV1 ABI mismatch");
CB2VEC_STATIC_ASSERT(sizeof(Cb2VecQuantizationConfigV1) == 32,
                     "Cb2VecQuantizationConfigV1 ABI mismatch");
CB2VEC_STATIC_ASSERT(sizeof(Cb2VecInferenceConfigV1) == 16,
                     "Cb2VecInferenceConfigV1 ABI mismatch");
CB2VEC_STATIC_ASSERT(sizeof(Cb2VecTrainingMetricsV1) == 64,
                     "Cb2VecTrainingMetricsV1 ABI mismatch");
CB2VEC_STATIC_ASSERT(sizeof(Cb2VecModelInfoV1) == 64,
                     "Cb2VecModelInfoV1 ABI mismatch");
#if UINTPTR_MAX == UINT64_MAX
CB2VEC_STATIC_ASSERT(sizeof(Cb2VecTrainingBatchV1) == 72,
                     "64-bit Cb2VecTrainingBatchV1 ABI mismatch");
#elif UINTPTR_MAX == UINT32_MAX
CB2VEC_STATIC_ASSERT(sizeof(Cb2VecTrainingBatchV1) == 48,
                     "32-bit Cb2VecTrainingBatchV1 ABI mismatch");
#endif
CB2VEC_STATIC_ASSERT(offsetof(Cb2VecTrainerConfigV1, seed) == 32,
                     "Cb2VecTrainerConfigV1.seed offset mismatch");
CB2VEC_STATIC_ASSERT(offsetof(Cb2VecTrainerConfigV1, learning_rate) == 40,
                     "Cb2VecTrainerConfigV1.learning_rate offset mismatch");
CB2VEC_STATIC_ASSERT(offsetof(Cb2VecTrainingMetricsV1, total_weight) == 16,
                     "Cb2VecTrainingMetricsV1.total_weight offset mismatch");
CB2VEC_STATIC_ASSERT(offsetof(Cb2VecTrainingMetricsV1, sample_count) == 24,
                     "Cb2VecTrainingMetricsV1.sample_count offset mismatch");
CB2VEC_STATIC_ASSERT(offsetof(Cb2VecModelInfoV1, factor_scale) == 52,
                     "Cb2VecModelInfoV1.factor_scale offset mismatch");
CB2VEC_STATIC_ASSERT(sizeof(Cb2VecSessionConfigV1) == 32,
                     "Cb2VecSessionConfigV1 ABI mismatch");
CB2VEC_STATIC_ASSERT(sizeof(Cb2VecTokenDeltaV1) == 12,
                     "Cb2VecTokenDeltaV1 ABI mismatch");
CB2VEC_STATIC_ASSERT(sizeof(Cb2VecSessionInfoV1) == 64,
                     "Cb2VecSessionInfoV1 ABI mismatch");
CB2VEC_STATIC_ASSERT(sizeof(Cb2VecArtifactMetadataV1) == 32,
                     "Cb2VecArtifactMetadataV1 ABI mismatch");
CB2VEC_STATIC_ASSERT(sizeof(Cb2VecArtifactInfoV1) == 112,
                     "Cb2VecArtifactInfoV1 ABI mismatch");
CB2VEC_STATIC_ASSERT(offsetof(Cb2VecTokenDeltaV1, old_token) == 8,
                     "Cb2VecTokenDeltaV1.old_token offset mismatch");
CB2VEC_STATIC_ASSERT(offsetof(Cb2VecSessionInfoV1, max_sites) == 32,
                     "Cb2VecSessionInfoV1.max_sites offset mismatch");
CB2VEC_STATIC_ASSERT(offsetof(Cb2VecArtifactMetadataV1, schema_digest) == 16,
                     "Cb2VecArtifactMetadataV1.schema_digest offset mismatch");
CB2VEC_STATIC_ASSERT(offsetof(Cb2VecArtifactInfoV1, source_sha256) == 64,
                     "Cb2VecArtifactInfoV1.source_sha256 offset mismatch");
CB2VEC_STATIC_ASSERT(offsetof(Cb2VecArtifactInfoV1, schema_digest) == 96,
                     "Cb2VecArtifactInfoV1.schema_digest offset mismatch");

#undef CB2VEC_STATIC_ASSERT

#endif /* CB2VEC_H */
