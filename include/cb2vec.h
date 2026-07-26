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
 * CB2Vec C ABI 1.0
 *
 * Version axes are independent:
 *   - crate/library release: 0.2.1
 *   - binary artifact format: 1
 *   - C ABI: 1.0 (0x00010000)
 *
 * Caller-owned arrays are borrowed only for the duration of a call. A pointer
 * may be NULL only when its corresponding element count is zero. Handles are
 * library-owned and must be released exactly once by their matching free
 * function. Do not free a handle while another thread is using it.
 *
 * Immutable Cb2VecModel prediction calls may run concurrently. Every call on
 * a given Cb2VecTrainer, including read-only calls, must be externally
 * serialized. Invalid/dangling handles, double-free, undersized allocations,
 * and racing free calls are caller undefined behavior.
 */

typedef int32_t Cb2VecStatus;
typedef struct Cb2VecTrainer Cb2VecTrainer;
typedef struct Cb2VecModel Cb2VecModel;

#define CB2VEC_ABI_VERSION UINT32_C(0x00010000)

#define CB2VEC_OK INT32_C(0)
#define CB2VEC_ERROR_NULL_POINTER INT32_C(-1)
#define CB2VEC_ERROR_INVALID_ARGUMENT INT32_C(-2)
#define CB2VEC_ERROR_ABI_MISMATCH INT32_C(-3)
#define CB2VEC_ERROR_ARTIFACT INT32_C(-4)
#define CB2VEC_ERROR_MODEL INT32_C(-5)
#define CB2VEC_ERROR_NUMERIC INT32_C(-6)
#define CB2VEC_ERROR_BUFFER_TOO_SMALL INT32_C(-7)
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

#undef CB2VEC_STATIC_ASSERT

#endif /* CB2VEC_H */
