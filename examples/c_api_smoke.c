/*
 * End-to-end C ABI smoke test.
 *
 * Covers the ABI 1.0 trainer path plus every ABI 1.1 addition: incremental
 * search sessions, trainer checkpoints, and version-2 artifacts. Built with
 * -Wall -Wextra -Werror as C11 and compiled again as C++17 in CI.
 */

#include "cb2vec.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition, code)                                                 \
    do {                                                                       \
        if (!(condition)) {                                                    \
            fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__,      \
                    #condition, cb2vec_last_error());                          \
            return (code);                                                     \
        }                                                                      \
    } while (0)

/*
 * Five sites over a nine-token ragged layout:
 *   site 0 -> tokens[0..3)  group 0
 *   site 1 -> tokens[3..5)  group 1
 *   site 2 -> tokens[5..7)  group 1
 *   site 3 -> tokens[7..8)  group 2
 *   site 4 -> tokens[8..9)  group 0
 */
static const uint16_t kTokens[9] = {0, 3, 5, 1, 8, 2, 7, 4, 6};
static const uint32_t kSiteOffsets[6] = {0, 3, 5, 7, 8, 9};
static const uint32_t kSiteGroups[5] = {0, 1, 1, 2, 0};

static int bits_equal(float left, float right) {
    uint32_t left_bits;
    uint32_t right_bits;
    memcpy(&left_bits, &left, sizeof(left_bits));
    memcpy(&right_bits, &right, sizeof(right_bits));
    return left_bits == right_bits;
}

/* Full-rebuild reference score for an arbitrary token vector. */
static int rebuild(const Cb2VecModel *model, const uint16_t *tokens, float *out) {
    return cb2vec_model_predict_v1(model, tokens, 9, kSiteOffsets, kSiteGroups,
                                   5, out);
}

static int run_session_checks(Cb2VecModel *model) {
    Cb2VecSessionConfigV1 session_config;
    Cb2VecSessionInfoV1 session_info;
    Cb2VecSession *session = NULL;
    Cb2VecTokenDeltaV1 frame[2];
    uint16_t mutated[9];
    uint32_t popped = 0;
    float incremental = 0.0f;
    float reference = 0.0f;
    int status;
    int depth;

    CHECK(cb2vec_session_config_default_v1(&session_config) == CB2VEC_OK, 20);
    session_config.max_sites = 8;
    session_config.max_token_slots = 16;
    session_config.max_deltas_per_frame = 4;
    session_config.max_depth = 6;
    CHECK(cb2vec_session_create_v1(model, &session_config, &session) == CB2VEC_OK,
          21);

    /* Scoring before a reset is a state error, not a crash. */
    status = cb2vec_session_predict_v1(session, &incremental);
    if (status != CB2VEC_ERROR_STATE) {
        cb2vec_session_free_v1(session);
        return 22;
    }

    if (cb2vec_session_reset_v1(session, kTokens, 9, kSiteOffsets, kSiteGroups,
                                5) != CB2VEC_OK) {
        fprintf(stderr, "reset failed: %s\n", cb2vec_last_error());
        cb2vec_session_free_v1(session);
        return 23;
    }

    /* The initial position must match the non-incremental path exactly. */
    if (cb2vec_session_predict_v1(session, &incremental) != CB2VEC_OK ||
        rebuild(model, kTokens, &reference) != CB2VEC_OK ||
        !bits_equal(incremental, reference)) {
        fprintf(stderr, "initial score mismatch: %s\n", cb2vec_last_error());
        cb2vec_session_free_v1(session);
        return 24;
    }

    /* One move touching two sites, then the same score by full rebuild. */
    frame[0].site = 0;
    frame[0].lane = 0;
    frame[0].old_token = 0;
    frame[0].new_token = 6;
    frame[1].site = 2;
    frame[1].lane = 1;
    frame[1].old_token = 7;
    frame[1].new_token = 0;
    memcpy(mutated, kTokens, sizeof(kTokens));
    mutated[0] = 6;
    mutated[6] = 0;

    if (cb2vec_session_push_v1(session, frame, 2) != CB2VEC_OK ||
        cb2vec_session_materialize_v1(session) != CB2VEC_OK ||
        cb2vec_session_predict_v1(session, &incremental) != CB2VEC_OK ||
        rebuild(model, mutated, &reference) != CB2VEC_OK ||
        !bits_equal(incremental, reference)) {
        fprintf(stderr, "incremental score mismatch: %s\n", cb2vec_last_error());
        cb2vec_session_free_v1(session);
        return 25;
    }

    /* A delta whose expected old token is wrong must change nothing. */
    frame[0].old_token = 99;
    status = cb2vec_session_push_v1(session, frame, 1);
    if (status != CB2VEC_ERROR_INVALID_ARGUMENT) {
        cb2vec_session_free_v1(session);
        return 26;
    }
    if (cb2vec_session_get_info_v1(session, &session_info) != CB2VEC_OK ||
        session_info.depth != 1 || session_info.site_count != 5 ||
        session_info.token_slots != 9) {
        cb2vec_session_free_v1(session);
        return 27;
    }
    if (cb2vec_session_predict_v1(session, &incremental) != CB2VEC_OK ||
        !bits_equal(incremental, reference)) {
        cb2vec_session_free_v1(session);
        return 28;
    }

    /* Fill the stack, confirm the ceiling, then unwind to the start. */
    for (depth = 1; depth < (int)session_config.max_depth; ++depth) {
        Cb2VecTokenDeltaV1 step;
        step.site = 3;
        step.lane = 0;
        step.old_token = (uint16_t)(depth == 1 ? 4 : depth - 1);
        step.new_token = (uint16_t)depth;
        if (cb2vec_session_push_v1(session, &step, 1) != CB2VEC_OK) {
            fprintf(stderr, "push %d failed: %s\n", depth, cb2vec_last_error());
            cb2vec_session_free_v1(session);
            return 29;
        }
    }
    {
        Cb2VecTokenDeltaV1 overflow;
        overflow.site = 3;
        overflow.lane = 0;
        overflow.old_token = (uint16_t)(session_config.max_depth - 1);
        overflow.new_token = 0;
        status = cb2vec_session_push_v1(session, &overflow, 1);
        if (status != CB2VEC_ERROR_LIMIT_EXCEEDED) {
            cb2vec_session_free_v1(session);
            return 30;
        }
    }
    for (depth = 0; depth < (int)session_config.max_depth; ++depth) {
        if (cb2vec_session_pop_v1(session, &popped) != CB2VEC_OK) {
            cb2vec_session_free_v1(session);
            return 31;
        }
    }
    status = cb2vec_session_pop_v1(session, NULL);
    if (status != CB2VEC_ERROR_STATE) {
        cb2vec_session_free_v1(session);
        return 32;
    }
    if (cb2vec_session_predict_v1(session, &incremental) != CB2VEC_OK ||
        rebuild(model, kTokens, &reference) != CB2VEC_OK ||
        !bits_equal(incremental, reference)) {
        fprintf(stderr, "unwound score mismatch: %s\n", cb2vec_last_error());
        cb2vec_session_free_v1(session);
        return 33;
    }

    return cb2vec_session_free_v1(session) == CB2VEC_OK ? 0 : 34;
}

static int run_checkpoint_checks(Cb2VecTrainer *trainer) {
    Cb2VecTrainer *resumed = NULL;
    uint8_t *bytes = NULL;
    uint32_t required = 0;
    uint32_t declared = 0;
    uint32_t written = 0;
    int status;

    CHECK(cb2vec_trainer_checkpoint_len_v1(trainer, &declared) == CB2VEC_OK, 40);
    status = cb2vec_trainer_write_checkpoint_v1(trainer, NULL, 0, &required);
    CHECK(status == CB2VEC_ERROR_BUFFER_TOO_SMALL, 41);
    CHECK(required == declared && required > 0, 42);

    bytes = (uint8_t *)malloc(required);
    CHECK(bytes != NULL, 43);
    if (cb2vec_trainer_write_checkpoint_v1(trainer, bytes, required, &written) !=
            CB2VEC_OK ||
        written != required) {
        fprintf(stderr, "checkpoint write failed: %s\n", cb2vec_last_error());
        free(bytes);
        return 44;
    }
    if (cb2vec_trainer_load_checkpoint_v1(bytes, required, &resumed) !=
        CB2VEC_OK) {
        fprintf(stderr, "checkpoint load failed: %s\n", cb2vec_last_error());
        free(bytes);
        return 45;
    }
    cb2vec_trainer_free_v1(resumed);
    resumed = NULL;

    /* A single flipped payload bit must be rejected by the checksum. */
    bytes[required - 1] = (uint8_t)(bytes[required - 1] ^ 0x01u);
    status = cb2vec_trainer_load_checkpoint_v1(bytes, required, &resumed);
    free(bytes);
    CHECK(status == CB2VEC_ERROR_CHECKPOINT, 46);
    CHECK(resumed == NULL, 47);
    return 0;
}

int main(void) {
    Cb2VecModelShapeV1 shape;
    Cb2VecTrainerConfigV1 config;
    Cb2VecQuantizationConfigV1 quantization;
    Cb2VecArtifactMetadataV1 metadata;
    Cb2VecArtifactInfoV1 artifact_info;
    Cb2VecModelInfoV1 info;
    Cb2VecTrainer *trainer = NULL;
    Cb2VecModel *model = NULL;
    uint8_t digest[32];
    uint8_t *artifact = NULL;
    uint32_t required = 0;
    int result;

    CHECK(cb2vec_abi_version() == CB2VEC_ABI_VERSION, 1);
    CHECK((cb2vec_abi_version() >> 16) == 1u, 1);

    CHECK(cb2vec_model_shape_default_v1(&shape) == CB2VEC_OK, 2);
    shape.token_count = 9;
    shape.group_count = 3;
    shape.dim = 4;
    shape.fm_rank = 2;
    CHECK(cb2vec_trainer_config_default_v1(&config) == CB2VEC_OK, 2);
    config.pooling = CB2VEC_POOLING_MEAN;
    config.activation = CB2VEC_ACTIVATION_RELU;
    CHECK(cb2vec_trainer_create_v1(&shape, &config, &trainer) == CB2VEC_OK, 2);
    CHECK(cb2vec_trainer_get_info_v1(trainer, &info) == CB2VEC_OK, 2);
    if (info.token_count != shape.token_count || info.dim != shape.dim) {
        cb2vec_trainer_free_v1(trainer);
        return 3;
    }

    /* Version-2 artifact: the recipe travels with the weights. */
    memset(digest, 0x5A, sizeof(digest));
    CHECK(cb2vec_quantization_config_default_v1(&quantization) == CB2VEC_OK, 4);
    metadata.struct_size = (uint32_t)sizeof(metadata);
    metadata.abi_version = CB2VEC_ABI_VERSION;
    metadata.schema_version = 7;
    metadata.flags = 0;
    memset(metadata.schema_digest, 0xC3, sizeof(metadata.schema_digest));

    if (cb2vec_trainer_write_artifact_v2(trainer, &quantization, digest,
                                         &metadata, NULL, 0, &required) !=
        CB2VEC_ERROR_BUFFER_TOO_SMALL) {
        cb2vec_trainer_free_v1(trainer);
        return 5;
    }
    artifact = (uint8_t *)malloc(required);
    if (artifact == NULL) {
        cb2vec_trainer_free_v1(trainer);
        return 6;
    }
    if (cb2vec_trainer_write_artifact_v2(trainer, &quantization, digest,
                                         &metadata, artifact, required,
                                         &required) != CB2VEC_OK) {
        fprintf(stderr, "artifact v2 write failed: %s\n", cb2vec_last_error());
        free(artifact);
        cb2vec_trainer_free_v1(trainer);
        return 7;
    }
    if (cb2vec_artifact_probe_v1(artifact, required, &artifact_info) !=
            CB2VEC_OK ||
        artifact_info.artifact_version != CB2VEC_ARTIFACT_VERSION_V2 ||
        artifact_info.has_inference_config != 1u ||
        artifact_info.activation != CB2VEC_ACTIVATION_RELU ||
        artifact_info.pooling != CB2VEC_POOLING_MEAN ||
        artifact_info.schema_version != 7u ||
        artifact_info.token_count != shape.token_count) {
        free(artifact);
        cb2vec_trainer_free_v1(trainer);
        return 8;
    }

    /* NULL inference means "use the recipe the artifact stored". */
    if (cb2vec_model_load_v2(artifact, required, NULL, &metadata, &model) !=
        CB2VEC_OK) {
        fprintf(stderr, "model load failed: %s\n", cb2vec_last_error());
        free(artifact);
        cb2vec_trainer_free_v1(trainer);
        return 9;
    }
    free(artifact);

    result = run_session_checks(model);
    if (result != 0) {
        cb2vec_model_free_v1(model);
        cb2vec_trainer_free_v1(trainer);
        return result;
    }

    result = run_checkpoint_checks(trainer);
    if (result != 0) {
        cb2vec_model_free_v1(model);
        cb2vec_trainer_free_v1(trainer);
        return result;
    }

    if (cb2vec_model_free_v1(model) != CB2VEC_OK ||
        cb2vec_trainer_free_v1(trainer) != CB2VEC_OK) {
        return 10;
    }
    printf("CB2Vec %s C ABI 0x%08x smoke passed.\n", cb2vec_library_version(),
           cb2vec_abi_version());
    return 0;
}
