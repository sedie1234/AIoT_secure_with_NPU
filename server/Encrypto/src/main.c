#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "model_utils.h"
#include "onnx.pb-c.h"
#include "encrypt.h"
#include "key_utils.h"
#include "matrix.h"
#include "gal251_field.h"
#include "enc_struct.h"

int get_random_int(int min, int max);

void save_all_attributes(Onnx__ModelProto *model, const char *filename);
Onnx__AttributeProto*** load_all_attributes(const char *filename, size_t *n_nodes, size_t **n_attributes);
void* free_loaded_attributes(Onnx__AttributeProto ***all_attrs, size_t n_nodes, size_t *n_attributes);

int main(int argc, char *argv[]) {

    // 1. argument parsing stage
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <input_onnx> <output.em> <output.et> <key_path>\n", argv[0]);
        return 1;
    }

    char* model_path = argv[1];
    char* out_model_path = argv[2];
    char* table_filename = argv[3];
    char* key_filename = argv[4];

    // 2. model loading stage
    printf("Loading ONNX model: %s\n", model_path);
    Onnx__ModelProto *model = load_onnx_model(model_path);
    if (!model) {
        return 1;
    }

    // 3. model modification stage

    // read key file
    KeyInfo key_info;
    memset(&key_info, 0, sizeof(key_info));
    int ret = read_key(key_filename, &key_info);    

    // set variables
    int w = 32;
    if (key_info.size < w*w) {
        printf("[Error] Key size is too small\n");
        return 1;
    }
    int upos = get_random_int(0, key_info.size-w*w-1);
    int lpos = get_random_int(0, key_info.size-w*w-1);
    int bpos;

    // shuffle order of node, initializer
    shuffle_nodes(model);
    shuffle_initializers(model);

    // create connection info matrix : int array
    // normalize : fixed # of col, unsigned 8bit integer
    uint32_t* mat = create_conninfo_mat(model);    
    uint8_t* norm_mat = mat_normalize(mat, model, w);
    uint8_t* norm_mat_bit_info;

    int64_t mat_size=0;
    int64_t n_nodes = get_node_size(model);
    int64_t n_inits = get_initializer_size(model);

    free_matrix_int(mat, n_nodes);

    //get matrix size
    for(int i=0; i<n_nodes; i++){
        int64_t n_inputs = model->graph->node[i]->n_input;
        int64_t n_outputs = model->graph->node[i]->n_output;
        for(int j=0; j<n_inputs+n_outputs; j++){
            mat_size++;
        }
    }

    //get height of matrix
    int h = (mat_size*4 +w -1) /w;

    //get random bias position
    if (key_info.size < w*h) {
        printf("[Error] Key size is too small\n");
        return 1;
    }

    norm_mat_bit_info = (uint8_t*)malloc(sizeof(uint8_t)*h*w/8 + h*w%8);

    bpos = get_random_int(0, key_info.size-w*h-1);

    //get key array : U_inv, L_inv
    uint8_t* U = create_upper_triangular_matrix(key_info, upos, w);
    uint8_t* L = create_lower_triangular_matrix(key_info, lpos, w);
    uint8_t* B = create_bias_matrix(key_info, bpos, h, w);

    uint8_t* U_inv = allocate_matrix_uint8(w, w);
    uint8_t* L_inv = allocate_matrix_uint8(w, w);
    uint8_t* UinvLinv = allocate_matrix_uint8(w, w);
    uint8_t* Y = allocate_matrix_uint8(h, w);

    // matrix into gal251 field
    int bit_index = 0;
    int bit_offset = 0;

    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            norm_mat[i*w+j] = into_gal251(norm_mat[i*w+j]).value;
            norm_mat_bit_info[bit_index] |= (norm_mat[i*w+j] << bit_offset);
            bit_offset++;
            if(bit_offset == 8){
                bit_offset = 0;
                bit_index++;
            }
        }
    }

    for(int i=0; i<w; i++){
        for(int j=0; j<w; j++){
            U[i*w+j] = into_gal251(U[i*w+j]).value;
            L[i*w+j] = into_gal251(L[i*w+j]).value;
        }
    }

    if(!(gal251_inverse_upper_triangular_matrix(U, U_inv, w))){
        printf("[Error] Can't get inverse\n");
        return 1;
    }

    if(!(gal251_inverse_lower_triangular_matrix(L, L_inv, w))){
        printf("[Error] Can't get inverse\n");
        return 1;
    }

    // matrix encryption
    gal251_matrix_multiply(U_inv, L_inv, UinvLinv, w, w, w);
    gal251_matrix_multiply(norm_mat, UinvLinv, Y, h, w, w);
    gal251_matrix_add(Y, B, Y, h, w);



    //matrix buffer free
    free_matrix_uint8(norm_mat, h);
    free_matrix_uint8(U, w);
    free_matrix_uint8(L, w);
    free_matrix_uint8(B, h);
    
    free_matrix_uint8(U_inv, h);
    free_matrix_uint8(L_inv, h);
    


    EncModel enc_model = init_enc_model(model, w);
    
    enc_model.h = h;
    enc_model.lpos = lpos;
    enc_model.upos = upos;
    enc_model.bpos = bpos;
    enc_model.enc_conninfo_mat = Y;
    enc_model.enc_conninfo_mat_bit_info = norm_mat_bit_info;

    

    // initialzier encryption
    for (size_t i = 0; i < n_inits; i++) {

        int init_h = enc_model.inits[i].n_raw_data/w;

        Onnx__TensorProto *initializer = model->graph->initializer[i];
        size_t size = initializer->raw_data.len * sizeof(initializer->raw_data.data[0]);
        size_t remain = size;
        bit_index = 0;
        bit_offset = 0;

        memcpy(enc_model.inits[i].raw_data, initializer->raw_data.data, size);

        for(int j=0; j<enc_model.inits[i].n_raw_data; j++) {
            Gal251Field val = into_gal251(enc_model.inits[i].raw_data[j]);
            enc_model.inits[i].raw_data[j] = val.value;
            enc_model.inits[i].raw_bit_data[j/8] |= (val.m_bit << (j%8));
        }

        gal251_matrix_multiply(enc_model.inits[i].raw_data, UinvLinv, enc_model.inits[i].raw_data, init_h, enc_model.MAC_w, enc_model.MAC_w);
        int bias_index;

        for(int j=0; j<enc_model.inits[i].n_raw_data; j++){
            enc_model.inits[i].raw_data[i] = gal251_add(enc_model.inits[i].raw_data[i], key_info.key[(bpos+bias_index)%key_info.size]);
            bias_index++;   
        }
    }

    // save enc_model
    save_enc_model(enc_model, out_model_path);
    free_enc_model(&enc_model);
    //key buffer free
    

    // 4. model saving stage
    // printf("Saving modified ONNX model: %s\n", out_model_path);
    // save_onnx_model(out_model_path, model);

    // 5. cleanup stage
    // onnx__model_proto__free_unpacked(model, NULL);

    printf("Done!\n");

    // 6. model table creation stage

    save_all_attributes(model, table_filename);
    
    // 7. free stage
    key_free(&key_info);
    free_matrix_uint8(UinvLinv, h);
    free_matrix_uint8(Y, h);
    free(norm_mat_bit_info);

    //=====This code for Decrypting the model=====
    // int __n_nodes;
    // size_t* __n_attributes;
    // Onnx__AttributeProto*** all_attrs = load_all_attributes(table_filename, &__n_nodes, &__n_attributes);
    // if (!all_attrs) {
    //     return 1;
    // }
    
    // free_loaded_attributes(all_attrs, __n_nodes, __n_attributes);
    return 0;
}

int get_random_int(int min, int max) {
    return min + rand() % (max - min + 1);
}

void save_all_attributes(Onnx__ModelProto *model, const char *filename) {
    if (!model || !model->graph) {
        printf("Invalid ONNX model.\n");
        return;
    }

    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open file");
        return;
    }

    size_t n_nodes = model->graph->n_node;
    fwrite(&n_nodes, sizeof(size_t), 1, file);  // 노드 개수 저장

    for (size_t i = 0; i < n_nodes; i++) {
        Onnx__NodeProto *node = model->graph->node[i];
        int op_type_len = strlen(node->op_type);
        fwrite(&op_type_len, sizeof(int), 1, file);  // 노드 타입 길이 저장
        fwrite(&node->op_type, sizeof(char), strlen(node->op_type), file);  // 노드 타입 저장
        fwrite(&node->n_input, sizeof(size_t), 1, file);  // 입력 개수 저장
        fwrite(&node->n_output, sizeof(size_t), 1, file);  // 출력 개수 저장

        size_t n_attributes = node->n_attribute;
        fwrite(&n_attributes, sizeof(size_t), 1, file);  // 각 노드의 속성 개수 저장

        for (size_t j = 0; j < n_attributes; j++) {
            size_t attr_size = onnx__attribute_proto__get_packed_size(node->attribute[j]);
            uint8_t *buffer = malloc(attr_size);

            onnx__attribute_proto__pack(node->attribute[j], buffer);  // 직렬화
            fwrite(&attr_size, sizeof(size_t), 1, file);  // 크기 저장
            fwrite(buffer, attr_size, 1, file);  // 데이터 저장

            free(buffer);
        }
    }

    fclose(file);
    printf("All attributes saved to %s\n", filename);
}

Onnx__AttributeProto*** load_all_attributes(const char *filename, size_t *n_nodes, size_t **n_attributes) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file");
        return NULL;
    }

    fread(n_nodes, sizeof(size_t), 1, file);  // 노드 개수 읽기
    Onnx__AttributeProto ***all_attrs = malloc((*n_nodes) * sizeof(Onnx__AttributeProto**));

    // **n_attributes에 메모리 할당
    *n_attributes = malloc((*n_nodes) * sizeof(size_t));

    for (size_t i = 0; i < *n_nodes; i++) {
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