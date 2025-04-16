#ifndef ENCRYPT_H
#define ENCRYPT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnx.pb-c.h"
#include "model_utils.h"

void shuffle_nodes(Onnx__ModelProto *model);
void shuffle_initializers(Onnx__ModelProto *model);
// uint32_t** create_conninfo_mat(Onnx__ModelProto *model);
uint32_t* create_conninfo_mat(Onnx__ModelProto *model);
// suppose that w is a multiple of 4
// return value must be deallocated in the end.
// h = (mat_size*4 +w -1) /w
// uint8_t** mat_normalize(uint32_t **mat, Onnx__ModelProto *model, int w);
uint8_t* mat_normalize(uint32_t *mat, Onnx__ModelProto *model, int w);

#endif // ENCRYPT_H