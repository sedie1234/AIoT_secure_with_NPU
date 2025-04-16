#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "model_utils.h"
#include "onnx.pb-c.h"
#include "decrypt.h"
#include "key_utils.h"
#include "matrix.h"
#include "gal251_field.h"
#include "enc_struct.h"

int get_random_int(int min, int max);

Onnx__AttributeProto*** load_all_attributes(const char *filename, size_t *n_nodes, size_t **n_attributes, 
                            char** op_type, size_t* n_input, size_t* n_output);
void* free_loaded_attributes(Onnx__AttributeProto ***all_attrs, size_t n_nodes, size_t *n_attributes);

int main(int argc, char *argv[]) {

    // 1. argument parsing stage
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <output_onnx> <input.em> <input.et> <key_path>\n", argv[0]);
        return 1;
    }

    char* out_model_path = argv[1];
    char* enc_model_path = argv[2];
    char* table_filename = argv[3];
    char* key_filename = argv[4];

    // 2. model loading stage : em, ek, et

    //em load
    EncModel enc_model;
    load_enc_model(&enc_model, enc_model_path);

    //et load
    size_t __n_nodes;
    size_t* __n_attributes;
    char** op_types;
    int* node_n_inputs;
    int* node_n_outputs;
    Onnx__AttributeProto*** all_attrs = load_all_attributes(table_filename, &__n_nodes, &__n_attributes
                                                            , &op_types, &node_n_inputs, &node_n_outputs);
    if (!all_attrs) {
        return 1;
    }
    
    //ek load
    KeyInfo key_info;
    memset(&key_info, 0, sizeof(key_info));
    int ret = read_key(key_filename, &key_info);

    // 3. decrypt stage

    //make L, U, B
    uint8_t* U = create_upper_triangular_matrix(key_info, enc_model.upos, enc_model.MAC_w);
    uint8_t* L = create_lower_triangular_matrix(key_info, enc_model.lpos, enc_model.MAC_w);
    uint8_t* B = create_bias_matrix(key_info, enc_model.bpos, enc_model.h, enc_model.MAC_w);

    uint8_t* LU = allocate_matrix_uint8(enc_model.MAC_w, enc_model.MAC_w);

    //model conn decrypt
    gal251_matrix_multiply(L, U, LU, enc_model.MAC_w, enc_model.MAC_w, enc_model.MAC_w);
    gal251_matrix_sub(enc_model.enc_conninfo_mat, B, enc_model.enc_conninfo_mat, enc_model.h, enc_model.MAC_w);
    gal251_matrix_multiply(enc_model.enc_conninfo_mat, LU, enc_model.enc_conninfo_mat, enc_model.h, enc_model.MAC_w, enc_model.MAC_w);
    mat_251to256(enc_model.enc_conninfo_mat, enc_model.enc_conninfo_mat, enc_model.enc_conninfo_mat_bit_info, enc_model.h*enc_model.MAC_w);

    //inits decrypt
    for(int i=0; i<enc_model.n_inits; i++){
        int init_h = enc_model.inits[i].n_raw_data / enc_model.MAC_w;
        int bias_index = 0;
        for(int j=0; j<enc_model.inits[i].n_raw_data; j++){
            enc_model.inits[i].raw_data[j] = gal251_sub(enc_model.inits[i].raw_data[j], key_info.key[(enc_model.bpos+bias_index)%key_info.size]);
            bias_index++;   
        }
        gal251_matrix_multiply(enc_model.inits[i].raw_data, LU, enc_model.inits[i].raw_data, init_h, enc_model.MAC_w, enc_model.MAC_w);
        mat_251to256(enc_model.inits[i].raw_data, enc_model.inits[i].raw_data, enc_model.inits[i].raw_bit_data, enc_model.inits[i].n_raw_data);
    }

    // 4. model modification stage

    char* node_name = "node";

    Onnx__ModelProto model = ONNX__MODEL_PROTO__INIT;
    model.ir_version=12;
    model.producer_name="onnx-decryptor";
    model.producer_version="0.0.1";
    model.domain="ai.onnx";
    model.model_version=1;
    model.doc_string="Decrypted model";

    Onnx__GraphProto graph = ONNX__GRAPH_PROTO__INIT;
    graph.name="graph";
    graph.n_node = enc_model.n_nodes;
    graph.node = malloc(enc_model.n_nodes * sizeof(Onnx__NodeProto*));
    graph.n_initializer = enc_model.n_inits;
    graph.initializer = malloc(enc_model.n_inits * sizeof(Onnx__TensorProto*));
    graph.n_input = enc_model.n_model_inputs;
    graph.input = malloc(enc_model.n_model_inputs * sizeof(Onnx__ValueInfoProto*));
    graph.n_output = enc_model.n_model_outputs;
    graph.output = malloc(enc_model.n_model_outputs * sizeof(Onnx__ValueInfoProto*));

    for(int i=0; i<enc_model.n_model_inputs; i++){
        char* input_name = enc_model.inputs[i].dims[0] == 1 ? "input" : "input" + i;
        graph.input[i] = create_new_tensor(input_name, enc_model.inputs[i].data_type, enc_model.inputs[i].dims, enc_model.inputs[i].n_dims);
    }

    for(int i=0; i<enc_model.n_model_outputs; i++){
        char* output_name = enc_model.outputs[i].dims[0] == 1 ? "output" : "output" + i;
        graph.output[i] = create_new_tensor(output_name, enc_model.outputs[i].data_type, enc_model.outputs[i].dims, enc_model.outputs[i].n_dims);
    }

    for(int i=0; i<enc_model.n_inits; i++){
        char* init_name = enc_model.inits[i].dims[0] == 1 ? "init" : "init" + i;
        graph.initializer[i] = create_new_initializer(init_name, enc_model.inits[i].data_type, enc_model.inits[i].dims, enc_model.inits[i].n_dims, enc_model.inits[i].raw_data, enc_model.inits[i].n_raw_data);
    }

    for(int i=0; i<enc_model.n_nodes; i++){
        graph.node[i] = create_new_node(node_name+i, op_types[i], model.domain, NULL, 
                                        all_attrs[i], __n_attributes[i], NULL);
        graph.node[i]->n_input = node_n_inputs[i];
        graph.node[i]->n_output = node_n_outputs[i];
        graph.node[i]->input = malloc(node_n_inputs[i] * sizeof(char*));
        graph.node[i]->output = malloc(node_n_outputs[i] * sizeof(char*));
        
    }

    model.graph = &graph;



    // 5. model saving stage

    // 6. cleanup stage
    free_matrix_uint8(U, enc_model.MAC_w);
    free_matrix_uint8(L, enc_model.MAC_w);
    free_matrix_uint8(LU, enc_model.MAC_w);
    free_matrix_uint8(B, enc_model.h);

    free_loaded_attributes(all_attrs, __n_nodes, __n_attributes);
    free_enc_model(&enc_model);
    key_free(&key_info);

    for(int i=0; i<model.graph->n_node; i++){
        free_node(model.graph->node[i]);
    }

    // free(graph.node);
    free(graph.initializer);
    free(graph.input);
    free(graph.output);
    free(op_types);
    free(node_n_inputs);
    free(node_n_outputs);
    

}

Onnx__AttributeProto*** load_all_attributes(const char *filename, size_t *n_nodes, size_t **n_attributes, 
                            char** op_type, size_t* n_input, size_t* n_output) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file");
        return NULL;
    }

    fread(n_nodes, sizeof(size_t), 1, file);  // 노드 개수 읽기
    Onnx__AttributeProto ***all_attrs = malloc((*n_nodes) * sizeof(Onnx__AttributeProto**));

    // **n_attributes에 메모리 할당
    *n_attributes = malloc((*n_nodes) * sizeof(size_t));
    op_type = malloc((*n_nodes) * sizeof(char*));

    for (size_t i = 0; i < *n_nodes; i++) {
        int op_type_len;
        fread(&op_type_len, sizeof(int), 1, file);  // 노드 타입 길이 저장
        op_type[i] = malloc(op_type_len);
        fread(op_type[i], sizeof(char), op_type_len, file);  // 노드 타입 저장
        fread(n_input[i], sizeof(size_t), 1, file);  // 입력 개수 저장
        fread(n_output[i], sizeof(size_t), 1, file);  // 출력 개수 저장
        fread(&((*n_attributes)[i]), sizeof(size_t), 1, file);  // 노드별 속성 개수 읽기

        all_attrs[i] = malloc((*n_attributes)[i] * sizeof(Onnx__AttributeProto*));

        for (size_t j = 0; j < (*n_attributes)[i]; j++) {
            size_t attr_size;
            fread(&attr_size, sizeof(size_t), 1, file);  // 속성 크기 읽기

            uint8_t *buffer = malloc(attr_size);
            fread(buffer, attr_size, 1, file);

            all_attrs[i][j] = onnx__attribute_proto__unpack(NULL, attr_size, buffer);
            free(buffer);
        }
    }

    fclose(file);
    printf("All attributes loaded from %s\n", filename);
    return all_attrs;
}

void* free_loaded_attributes(Onnx__AttributeProto ***all_attrs, size_t n_nodes, size_t *n_attributes) {
    for (size_t i = 0; i < n_nodes; i++) {
        for (size_t j = 0; j < n_attributes[i]; j++) {
            free_attribute(all_attrs[i][j]);
        }
        free(all_attrs[i]);
    }
    free(all_attrs);
    free(n_attributes);
    return NULL;
}