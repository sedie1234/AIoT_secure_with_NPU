#ifndef ENC_STRUCT_C
#define ENC_STRUCT_C

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "enc_struct.h"
#include "model_utils.h"
#include "onnx.pb-c.h"
// #include "encrypt.h"
#include "key_utils.h"
#include "matrix.h"
#include "gal251_field.h"

EncModel init_enc_model(Onnx__ModelProto *model, int mac_w){
    EncModel enc_model;
    enc_model.MAC_w = mac_w;

    enc_model.n_nodes = model->graph->n_node;
    enc_model.n_inits = model->graph->n_initializer;
    enc_model.n_model_inputs = model->graph->n_input;
    enc_model.n_model_outputs = model->graph->n_output;
    
    enc_model.inputs = malloc(enc_model.n_model_inputs * sizeof(TensorPack));
    enc_model.outputs = malloc(enc_model.n_model_outputs * sizeof(TensorPack));

    // inputs set
    for(int i=0; i<enc_model.n_model_inputs; i++){
        enc_model.inputs[i].data_type = model->graph->input[i]->type->tensor_type->elem_type;
        enc_model.inputs[i].n_dims = model->graph->input[i]->type->tensor_type->shape->n_dim;
        enc_model.inputs[i].dims = malloc(enc_model.inputs[i].n_dims * sizeof(size_t));
        for(int j=0; j<enc_model.inputs[i].n_dims; j++){
            enc_model.inputs[i].dims[j] = model->graph->input[i]->type->tensor_type->shape->dim[j]->dim_value;
        }
        enc_model.inputs[i].n_raw_data = 0;
        enc_model.inputs[i].raw_data = NULL;
    }

    // outputs set
    for(int i=0; i<enc_model.n_model_outputs; i++){
        enc_model.outputs[i].data_type = model->graph->output[i]->type->tensor_type->elem_type;
        enc_model.outputs[i].n_dims = model->graph->output[i]->type->tensor_type->shape->n_dim;
        enc_model.outputs[i].dims = malloc(enc_model.outputs[i].n_dims * sizeof(size_t));
        for(int j=0; j<enc_model.outputs[i].n_dims; j++){
            enc_model.outputs[i].dims[j] = model->graph->output[i]->type->tensor_type->shape->dim[j]->dim_value;
        }
        enc_model.outputs[i].n_raw_data = 0;
        enc_model.outputs[i].raw_data = NULL;
    }

    // inits set
    enc_model.inits = malloc(enc_model.n_inits * sizeof(TensorPack));

    for (int i = 0; i < enc_model.n_inits; i++) {
        enc_model.inits[i].data_type = model->graph->initializer[i]->data_type;
        enc_model.inits[i].n_dims = model->graph->initializer[i]->n_dims;
        enc_model.inits[i].dims = malloc(enc_model.inits[i].n_dims * sizeof(size_t));
        for (int j = 0; j < enc_model.inits[i].n_dims; j++) {
            enc_model.inits[i].dims[j] = model->graph->initializer[i]->dims[j];
        }
        enc_model.inits[i].n_raw_data = model->graph->initializer[i]->raw_data.len 
                                        * sizeof(model->graph->initializer[i]->raw_data.data[0]);
        int enc_raw_size = (enc_model.inits[i].n_raw_data % enc_model.MAC_w) ? 
                        enc_model.inits[i].n_raw_data + enc_model.MAC_w - enc_model.inits[i].n_raw_data % enc_model.MAC_w : 
                        enc_model.inits[i].n_raw_data;
        enc_model.inits[i].raw_data = malloc(enc_raw_size);
        enc_model.inits[i].raw_bit_data = malloc(enc_raw_size/8 + enc_raw_size%8);
        memset(enc_model.inits[i].raw_data, 0, enc_raw_size);
        memcpy(enc_model.inits[i].raw_data, model->graph->initializer[i]->raw_data.data, model->graph->initializer[i]->raw_data.len);

    }

    return enc_model;
}

void save_enc_model(EncModel enc_model, const char *filename){
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("Failed to open file");
        return;
    }

    fwrite(&enc_model.MAC_w, sizeof(int), 1, fp);
    fwrite(&enc_model.h, sizeof(int), 1, fp);
    fwrite(&enc_model.n_nodes, sizeof(int), 1, fp);
    fwrite(&enc_model.n_inits, sizeof(int), 1, fp);
    fwrite(&enc_model.n_model_inputs, sizeof(int), 1, fp);
    fwrite(&enc_model.n_model_outputs, sizeof(int), 1, fp);
    
    for(int i=0; i<enc_model.n_model_inputs; i++){
        fwrite(&enc_model.inputs[i].data_type, sizeof(Onnx__TensorProto__DataType), 1, fp);
        fwrite(&enc_model.inputs[i].n_dims, sizeof(size_t), 1, fp);
        fwrite(enc_model.inputs[i].dims, sizeof(size_t), enc_model.inputs[i].n_dims, fp);
    }

    for(int i=0; i<enc_model.n_model_outputs; i++){
        fwrite(&enc_model.outputs[i].data_type, sizeof(Onnx__TensorProto__DataType), 1, fp);
        fwrite(&enc_model.outputs[i].n_dims, sizeof(size_t), 1, fp);
        fwrite(enc_model.outputs[i].dims, sizeof(size_t), enc_model.outputs[i].n_dims, fp);
    }

    fwrite(&enc_model.lpos, sizeof(int), 1, fp);
    fwrite(&enc_model.upos, sizeof(int), 1, fp);
    fwrite(&enc_model.bpos, sizeof(int), 1, fp);

    fwrite(enc_model.enc_conninfo_mat, sizeof(uint8_t), enc_model.h*enc_model.MAC_w, fp);

    fwrite(enc_model.enc_conninfo_mat_bit_info, sizeof(uint8_t), 
            enc_model.h*enc_model.MAC_w/8 + enc_model.h*enc_model.MAC_w%8, fp);

    for(int i=0; i<enc_model.n_inits; i++){
        fwrite(&enc_model.inits[i].data_type, sizeof(Onnx__TensorProto__DataType), 1, fp);
        fwrite(&enc_model.inits[i].n_dims, sizeof(size_t), 1, fp);
        fwrite(enc_model.inits[i].dims, sizeof(size_t), enc_model.inits[i].n_dims, fp);
        fwrite(&enc_model.inits[i].n_raw_data, sizeof(size_t), 1, fp);
        fwrite(enc_model.inits[i].raw_data, sizeof(uint8_t), enc_model.inits[i].n_raw_data, fp);
    }

    fclose(fp);

    return;
}

void load_enc_model(EncModel *enc_model, const char *filename){
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("Failed to open file");
        return;
    }

    fread(&enc_model->MAC_w, sizeof(int), 1, fp);
    fread(&enc_model->h, sizeof(int), 1, fp);
    fread(&enc_model->n_nodes, sizeof(int), 1, fp);
    fread(&enc_model->n_inits, sizeof(int), 1, fp);
    fread(&enc_model->n_model_inputs, sizeof(int), 1, fp);
    fread(&enc_model->n_model_outputs, sizeof(int), 1, fp);

    enc_model->inputs = malloc(enc_model->n_model_inputs * sizeof(TensorPack));
    enc_model->outputs = malloc(enc_model->n_model_outputs * sizeof(TensorPack));

    for(int i=0; i<enc_model->n_model_inputs; i++){
        fread(&enc_model->inputs[i].data_type, sizeof(Onnx__TensorProto__DataType), 1, fp);
        fread(&enc_model->inputs[i].n_dims, sizeof(size_t), 1, fp);
        enc_model->inputs[i].dims = malloc(enc_model->inputs[i].n_dims * sizeof(size_t));
        fread(enc_model->inputs[i].dims, sizeof(size_t), enc_model->inputs[i].n_dims, fp);
    }

    for(int i=0; i<enc_model->n_model_outputs; i++){
        fread(&enc_model->outputs[i].data_type, sizeof(Onnx__TensorProto__DataType), 1, fp);
        fread(&enc_model->outputs[i].n_dims, sizeof(size_t), 1, fp);
        enc_model->outputs[i].dims = malloc(enc_model->outputs[i].n_dims * sizeof(size_t));
        fread(enc_model->outputs[i].dims, sizeof(size_t), enc_model->outputs[i].n_dims, fp);
    }

    fread(&enc_model->lpos, sizeof(int), 1, fp);
    fread(&enc_model->upos, sizeof(int), 1, fp);
    fread(&enc_model->bpos, sizeof(int), 1, fp);

    enc_model->enc_conninfo_mat = malloc(enc_model->h*enc_model->MAC_w);
    fread(enc_model->enc_conninfo_mat, sizeof(uint8_t), enc_model->h*enc_model->MAC_w, fp);

    enc_model->enc_conninfo_mat_bit_info = malloc(enc_model->h*enc_model->MAC_w/8 + enc_model->h*enc_model->MAC_w%8);
    fread(enc_model->enc_conninfo_mat_bit_info, sizeof(uint8_t), 
            enc_model->h*enc_model->MAC_w/8 + enc_model->h*enc_model->MAC_w%8, fp);
    
    enc_model->inits = malloc(enc_model->n_inits * sizeof(TensorPack));
    fread(enc_model->inits, sizeof(TensorPack), enc_model->n_inits, fp);

}

void free_enc_model(EncModel *enc_model){
    for(int i=0; i<enc_model->n_model_inputs; i++){
        free(enc_model->inputs[i].dims);
    }
    free(enc_model->inputs);

    for(int i=0; i<enc_model->n_model_outputs; i++){
        free(enc_model->outputs[i].dims);
    }
    free(enc_model->outputs);

    // free(enc_model->enc_conninfo_mat);
    // free(enc_model->enc_conninfo_mat_bit_info);

    for(int i=0; i<enc_model->n_inits; i++){
        free(enc_model->inits[i].dims);
        free(enc_model->inits[i].raw_data);
        free(enc_model->inits[i].raw_bit_data);
    }
    free(enc_model->inits);

    return;
}
#endif //ENC_STRUCT_C