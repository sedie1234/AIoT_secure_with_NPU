#ifndef ENC_STRUCT_H
#define ENC_STRUCT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "onnx.pb-c.h"


typedef struct{
    Onnx__TensorProto__DataType data_type;
    size_t n_dims;
    size_t* dims; // malloc
    size_t n_raw_data; // if tensor is not initializer like input or output, n_raw_data is 0
    uint8_t* raw_data; // malloc
    uint8_t* raw_bit_data; // malloc
} TensorPack;

typedef struct {
    int MAC_w; // set in init_enc_model
    int h;
    int n_nodes; // set in init_enc_model
    int n_inits; // set in init_enc_model
    int n_model_inputs; // set in init_enc_model
    int n_model_outputs; // set in init_enc_model
    TensorPack* inputs; // set in init_enc_model // malloc
    TensorPack* outputs; // set in init_enc_model // malloc
    int lpos;
    int upos;
    int bpos;
    uint8_t* enc_conninfo_mat;
    uint8_t* enc_conninfo_mat_bit_info;
    TensorPack* inits; // set in init_enc_model // malloc
} EncModel;

EncModel init_enc_model(Onnx__ModelProto *model, int mac_w);
void save_enc_model(EncModel enc_model, const char *filename);
void load_enc_model(EncModel *enc_model, const char *filename);
void free_enc_model(EncModel *enc_model);

#endif //ENC_STRUCT_H