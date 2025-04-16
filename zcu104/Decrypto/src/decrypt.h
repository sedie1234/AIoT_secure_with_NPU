#ifndef DECRYPT_H
#define DECRYPT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnx.pb-c.h"
#include "model_utils.h"

void mat_251to256(uint8_t *mat, uint8_t *mat_256, uint8_t* bit_info, int size);
void table_to_model(Onnx__ModelProto* model, uint8_t* mat);

#endif // DECRYPT_H