#include "cb2vec.h"

#include <stdio.h>

int main(void) {
    Cb2VecModelShapeV1 shape;
    Cb2VecTrainerConfigV1 config;
    Cb2VecModelInfoV1 info;
    Cb2VecTrainer *trainer = NULL;

    if (cb2vec_abi_version() != CB2VEC_ABI_VERSION) {
        return 1;
    }
    if (cb2vec_model_shape_default_v1(&shape) != CB2VEC_OK ||
        cb2vec_trainer_config_default_v1(&config) != CB2VEC_OK ||
        cb2vec_trainer_create_v1(&shape, &config, &trainer) != CB2VEC_OK ||
        cb2vec_trainer_get_info_v1(trainer, &info) != CB2VEC_OK) {
        fprintf(stderr, "CB2Vec error: %s\n", cb2vec_last_error());
        cb2vec_trainer_free_v1(trainer);
        return 2;
    }
    if (info.token_count != shape.token_count || info.dim != shape.dim) {
        cb2vec_trainer_free_v1(trainer);
        return 3;
    }
    return cb2vec_trainer_free_v1(trainer) == CB2VEC_OK ? 0 : 4;
}
