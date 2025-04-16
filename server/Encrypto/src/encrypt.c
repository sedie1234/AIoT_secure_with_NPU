#ifndef ENCRIPT_C
#define ENCRIPT_C

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "onnx.pb-c.h"
#include "model_utils.h"
#include "encrypt.h"

void shuffle_nodes(Onnx__ModelProto *model) {
    if (!model->graph || model->graph->n_node <= 1) {
        return; 
    }

    srand(time(NULL)); 

    for (size_t i = model->graph->n_node - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);

        Onnx__NodeProto *temp = model->graph->node[i];
        model->graph->node[i] = model->graph->node[j];
        model->graph->node[j] = temp;
    }
}

void shuffle_initializers(Onnx__ModelProto *model){
    if (!model->graph || model->graph->n_initializer <= 1) {
        return;
    }

    srand(time(NULL)); 

    for (size_t i = model->graph->n_initializer - 1; i > 0; i--) {
        size_t j = rand() % (i + 1); 

        Onnx__TensorProto *temp = model->graph->initializer[i];
        model->graph->initializer[i] = model->graph->initializer[j];
        model->graph->initializer[j] = temp;
    }
}

// uint32_t** create_conninfo_mat(Onnx__ModelProto *model){

//     int64_t n_nodes = get_node_size(model);
//     if (n_nodes == -1) {
//         return NULL;
//     }
    
//     int64_t n_inits = get_initializer_size(model);
//     if (n_inits == -1) {
//         return NULL;
//     }

//     uint32_t **conninfo_mat = malloc(n_nodes * sizeof(uint32_t *));

//     for (int i = 0; i < n_nodes; i++) {
//         int64_t n_inputs = model->graph->node[i]->n_input;
//         int64_t n_outputs = model->graph->node[i]->n_output;
//         int64_t n_model_inputs = get_model_input_size(model);
//         int64_t n_model_outputs = get_model_output_size(model);
//         conninfo_mat[i] = malloc((n_inputs + n_outputs) * sizeof(uint32_t));
//         if (!conninfo_mat[i]) {
//             perror("Memory allocation failed");
//             for (int j = 0; j < i; j++) {
//                 free(conninfo_mat[j]);
//             }
//             free(conninfo_mat);
//             return NULL;
//         }

//         int64_t arr_index = 0;

//         // set output connection info
//         // if the node is output node, set 31st bit as 1
//         // if the node is output node, set 24-30 bits as output index
//         // if the node is not output node, set 0-23 bits as 0xffffff, else set 0-23 bits as node index
//         for (int j = 0; j < n_outputs; j++) {
//             char* output_name = model->graph->node[i]->output[j];

//             conninfo_mat[i][arr_index] = 0x80000000; // 31st bit is 1 for output
//             conninfo_mat[i][arr_index] |= j << 24; // 24-30 bits are for output index (7bit)
//             conninfo_mat[i][arr_index] |= 0x00ffffff; // 0-23 bits are for node index default=0xff_ffff(24bit)    
            
//             for (int k = 0; k < n_model_outputs; k++){
//                 if(strcmp(model->graph->output[k]->name, output_name) == 0){
//                     conninfo_mat[i][arr_index] &= (k|0xff000000); // 0-23 bits are for output index (24bit)
//                     break;
//                 }
                
//             }
//             arr_index++;
//         }

//         // set input connection info
//         // if the node is input node, set 31st bit as 0
//         // if the node is input node, set 24-30 bits as input index
//         // if the node is not input node, set 0-23 bits as 0xffffff, else set 0-23 bits as node index
//         // if the input is initializer, index added by n_nodes
//         // if the input is model input, index added by n_nodes + n_inits
//         for (int j = 0; j < n_inputs; j++) {
//             char* input_name = model->graph->node[i]->input[j];

//             conninfo_mat[i][arr_index] = 0x00000000; // 31st bit is 0 for input
//             conninfo_mat[i][arr_index] |= j << 24; // 24-30 bits are for input index (7bit)
//             conninfo_mat[i][arr_index] |= 0x00ffffff; // 0-23 bits are for node index default=0xff_ffff(24bit)    
            
//             for (int k = 0; k < n_nodes; k++){
//                 for(int l = 0; l < model->graph->node[k]->n_output; l++){
//                     if(strcmp(model->graph->node[k]->output[l], input_name) == 0){
//                         conninfo_mat[i][arr_index] &= (k|0xff000000); // 0-23 bits are for input index (24bit)
//                         break;
//                     }
//                 }
//             }

//             for (int k=0; k<n_inits; k++){
//                 if(strcmp(model->graph->initializer[k]->name, input_name) == 0){
//                     conninfo_mat[i][arr_index] &= ((k+n_nodes)|0xff000000); // 0-23 bits are for input index (24bit)
//                     break;
//                 }
//             }

//             for (int k=0; k<n_model_inputs; k++){
//                 if(strcmp(model->graph->input[k]->name, input_name) == 0){
//                     conninfo_mat[i][arr_index] &= ((k+n_nodes+n_inits)|0xff000000); // 0-23 bits are for input index (24bit)
//                     break;
//                 }
//             }

//             arr_index++;
//             if(arr_index > n_inputs + n_outputs){
//                 printf("[ERROR] Connection info matrix is not enough\n");
//                 return NULL;
//             }
//         }
//     }

//     return conninfo_mat;
// }

uint32_t* create_conninfo_mat(Onnx__ModelProto *model){
    
    int64_t n_nodes = get_node_size(model);
    if (n_nodes == -1) {
        return NULL;
    }
    
    int64_t n_inits = get_initializer_size(model);
    if (n_inits == -1) {
        return NULL;
    }

    int64_t mat_size=0;

    for(int i=0; i<n_nodes; i++){
        int64_t n_inputs = model->graph->node[i]->n_input;
        int64_t n_outputs = model->graph->node[i]->n_output;
        mat_size += n_inputs + n_outputs;
    }

    uint32_t *conninfo_mat = malloc(mat_size * sizeof(uint32_t));

    int64_t n_model_inputs = get_model_input_size(model);
    int64_t n_model_outputs = get_model_output_size(model);
    int64_t arr_index = 0;

    for (int i=0; i<n_nodes; i++){
        int64_t n_inputs = model->graph->node[i]->n_input;
        int64_t n_outputs = model->graph->node[i]->n_output;
        

        // set output connection info
        // if the node is output node, set 31st bit as 1
        // if the node is output node, set 24-30 bits as output index
        // if the node is not output node, set 0-23 bits as 0xffffff, else set 0-23 bits as node index
        for (int j = 0; j < n_outputs; j++) {
            char* output_name = model->graph->node[i]->output[j];

            conninfo_mat[arr_index] = 0x80000000; // 31st bit is 1 for output
            conninfo_mat[arr_index] |= j << 24; // 24-30 bits are for output index (7bit)
            conninfo_mat[arr_index] |= 0x00ffffff; // 0-23 bits are for node index default=0xff_ffff(24bit)    
            
            for (int k = 0; k < n_model_outputs; k++){
                if(strcmp(model->graph->output[k]->name, output_name) == 0){
                    conninfo_mat[arr_index] &= ((k+n_nodes+n_inits)|0xff000000); // 0-23 bits are for output index (24bit)
                    break;
                }
                
            }

            for (int k = 0; k < n_nodes; k++){
                for(int l = 0; l < model->graph->node[k]->n_input; l++){
                    if(strcmp(model->graph->node[k]->input[l], output_name) == 0){
                        conninfo_mat[arr_index] &= (k|0xff000000); // 0-23 bits are for output index (24bit)
                        break;
                    }
                }
            }

            arr_index++;
        }

        // set input connection info
        // if the node is input node, set 31st bit as 0
        // if the node is input node, set 24-30 bits as input index
        // if the node is not input node, set 0-23 bits as 0xffffff, else set 0-23 bits as node index
        // if the input is initializer, index added by n_nodes
        // if the input is model input, index added by n_nodes + n_inits
        for (int j = 0; j < n_inputs; j++) {
            char* input_name = model->graph->node[i]->input[j];

            conninfo_mat[arr_index] = 0x00000000; // 31st bit is 0 for input
            conninfo_mat[arr_index] |= j << 24; // 24-30 bits are for input index (7bit)
            conninfo_mat[arr_index] |= 0x00ffffff; // 0-23 bits are for node index default=0xff_ffff(24bit)

            for (int k = 0; k < n_nodes; k++){
                for(int l = 0; l < model->graph->node[k]->n_output; l++){
                    if(strcmp(model->graph->node[k]->output[l], input_name) == 0){
                        conninfo_mat[arr_index] &= (k|0xff000000); // 0-23 bits are for input index (24bit)
                        break;
                    }
                }
            }

            for (int k=0; k<n_inits; k++){
                if(strcmp(model->graph->initializer[k]->name, input_name) == 0){
                    conninfo_mat[arr_index] &= ((k+n_nodes)|0xff000000); // 0-23 bits are for input index (24bit)
                    break;
                }
            }

            for (int k=0; k<n_model_inputs; k++){
                if(strcmp(model->graph->input[k]->name, input_name) == 0){
                    conninfo_mat[arr_index] &= ((k+n_nodes+n_inits)|0xff000000); // 0-23 bits are for input index (24bit)
                    break;
                }
            }

            arr_index++;
            if(arr_index > mat_size){
                printf("[ERROR] Connection info matrix is not enough\n");
                return NULL;
            }
        }
    }

    return conninfo_mat;
}

// uint8_t** mat_normalize(uint32_t **mat, Onnx__ModelProto *model, int w){
//     int64_t n_nodes = get_node_size(model);
//     int64_t n_inits = get_initializer_size(model);

//     int64_t mat_size;

//     for(int i=0; i<n_nodes; i++){
//         int64_t n_inputs = model->graph->node[i]->n_input;
//         int64_t n_outputs = model->graph->node[i]->n_output;
        
//         for(int j=0; j<n_inputs+n_outputs; j++){
//             mat_size++;
//         }
//     }

//     printf("mat_size : %ld\n", mat_size);
//     int h = (mat_size * 4 + w - 1) / w;

//     uint8_t** array = (uint8_t**)malloc(sizeof(uint8_t*) * h);

//     for(int i=0; i<h; i++){
//         array[i] = (uint8_t*)malloc(sizeof(uint8_t)*w);
//     }
    
//     int array_index  = 0;

//     for(int i=0; i<n_nodes; i++){
//         int64_t n_inputs = model->graph->node[i]->n_input;
//         int64_t n_outputs = model->graph->node[i]->n_output;
        
//         for(int j=0; j<n_inputs+n_outputs; j++){
            
//             int ptr_offset = array_index / w;

//             *(*(array+ptr_offset) + array_index%w) = (mat[i][j] & 0xff000000) >> 24;
//             *(*(array+ptr_offset) + array_index%w +1) = (mat[i][j] & 0x00ff0000) >> 16;
//             *(*(array+ptr_offset) + array_index%w +2) = (mat[i][j] & 0x00ff0000) >> 8;
//             *(*(array+ptr_offset) + array_index%w +3) = mat[i][j] & 0x000000ff;

//             array_index+=4;
//         }
        
//     }

//     return array;
// }

uint8_t* mat_normalize(uint32_t *mat, Onnx__ModelProto *model, int w){
    int64_t n_nodes = get_node_size(model);
    int64_t n_inits = get_initializer_size(model);

    int64_t mat_size = 0;

    for(int i=0; i<n_nodes; i++){
        int64_t n_inputs = model->graph->node[i]->n_input;
        int64_t n_outputs = model->graph->node[i]->n_output;
        
        mat_size += n_inputs + n_outputs;
    }

    int h = (mat_size * 4 + w - 1) / w;

    uint8_t* array = (uint8_t*)malloc(sizeof(uint8_t) * h * w);

    int array_index  = 0;

    for(int i=0; i<n_nodes; i++){
        int64_t n_inputs = model->graph->node[i]->n_input;
        int64_t n_outputs = model->graph->node[i]->n_output;
        
        for(int j=0; j<n_inputs+n_outputs; j++){
            
            int ptr_offset = array_index / w;

            *(array + array_index) = (mat[i] & 0xff000000) >> 24;
            *(array + array_index + 1) = (mat[i] & 0x00ff0000) >> 16;
            *(array + array_index + 2) = (mat[i] & 0x0000ff00) >> 8;
            *(array + array_index + 3) = mat[i] & 0x000000ff;

            array_index+=4;
        }
        
    }

    return array;
}

#endif  // ENCRIPT_C