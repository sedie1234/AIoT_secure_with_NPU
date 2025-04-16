#ifndef DECRYPT_C
#define DECRYPT_C

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "onnx.pb-c.h"
#include "model_utils.h"
#include "decrypt.h"

void mat_251to256(uint8_t *mat, uint8_t *mat_256, uint8_t* bit_info, int size){
    int bit_index = 0;
    int bit_offset = 0;
    for(int i=0; i<size; i++){
        mat_256[i] = mat[i];
        mat_256[i] = mat_256[i] | ((bit_info[bit_index] >> bit_offset) & 1);
        bit_offset++;
        if(bit_offset == 8){
            bit_offset = 0;
            bit_index++;
        }
    }
}

void table_to_model(Onnx__ModelProto* model, uint8_t* mat){
    int node_count = 0;
    int info_count = 0;
    int node_index_max = model->graph->n_node;
    int init_index_max = model->graph->n_initializer;
    int model_out_index_max = model->graph->n_output;
    int model_in_index_max = model->graph->n_input;

    char* node_name = "node";
    for(int i=0; i<model->graph->n_node; i++){
        info_count += model->graph->node[i]->n_input;
        info_count += model->graph->node[i]->n_output;
    }

    int node_info_count = 0;
    for(int i=0; i<info_count; i++){
        Onnx__NodeProto *node = model->graph->node[i];
        static int input_count;
        static int output_count;
        int* _mat = (int*)(mat+4*i);
        if((*_mat)&0x80000000){
            //output info
            int output_pos = (*_mat)&0x00ffffff;
            int output_index = (*_mat)&0x7f000000 >> 24;
            if(output_pos == 0x00ffffff){
                //output node
                model->graph->node[node_count]->output[output_index] 
                    = model->graph->n_output==1 ? "output" : "output" + output_pos - node_index_max - init_index_max;
            }else{
                //////////////////////////////
            }
        }else{
            //input info
        }

        node_info_count ++;
        if(model->graph->node[node_count]->n_input + model->graph->node[node_count]->n_output == node_info_count){
            node_count++;
            node_info_count = 0;
            input_count = 0;
            output_count = 0;
        }
    }
}

#endif  // DECRYPT_C