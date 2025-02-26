#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model_utils.h"
#include "onnx.pb-c.h"
#include "encrypt.h"
#include "key_utils.h"
#include "matrix.h"
#include "gal251_field.h"

int main(int argc, char *argv[]) {

    // 1. argument parsing stage
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input_onnx> <output_onnx> <key_path>\n", argv[0]);
        return 1;
    }

    char* model_path = argv[1];
    char* out_model_path = argv[2];
    char* key_filename = argv[3];

    // 2. model loading stage
    printf("Loading ONNX model: %s\n", model_path);
    Onnx__ModelProto *model = load_onnx_model(model_path);
    if (!model) {
        return 1;
    }

    // 3. model modification stage
    int w = 32;

    // shuffle order of node, initializer
    shuffle_nodes(model);
    shuffle_initializers(model);

    // create connection info matrix : int array
    // normalize : fixed # of col, unsigned 8bit integer
    uint32_t** mat = create_conninfo_mat(model);    
    uint8_t** norm_mat = mat_normalize(mat, model, w);

    int64_t mat_size=0;
    int64_t n_nodes = get_node_size(model);
    int64_t n_inits = get_initializer_size(model);

    free_matrix_int(mat, n_nodes);

    for(int i=0; i<n_nodes; i++){
        int64_t n_inputs = model->graph->node[i]->n_input;
        int64_t n_outputs = model->graph->node[i]->n_output;
        for(int j=0; j<n_inputs+n_outputs; j++){
            mat_size++;
        }
    }

    int h = (mat_size*4 +w -1) /w;

    uint8_t** U_inv = allocate_matrix_uint8(w, w);
    uint8_t** L_inv = allocate_matrix_uint8(w, w);
    
    

    KeyInfo key_info;
    memset(&key_info, 0, sizeof(key_info));
    int ret = read_key(key_filename, &key_info);    

    //=============================================
    // key print test
    //=============================================
    // for(int i=0; i<10; i++){
    //     for(int j=0; j<10; j++){
    //         printf("%02X ", *(key_info.key + i*10 + j));
    //     }
    //     printf("\n");
    // }
    //=============================================

    //=============================================
    // matrix print test
    //=============================================
    // for(int i=0; i<h; i++){
    //     for(int j=0; j<w; j++){
    //         printf("%02X ",norm_mat[i][j]);
    //     }
    //     printf("\n");
    // }
    //=============================================

    uint8_t** U = create_upper_triangular_matrix(key_info.key, key_info.size, w);
    uint8_t** L = create_lower_triangular_matrix(key_info.key, key_info.size, w);

    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            norm_mat[i][j] = into_gal251(norm_mat[i][j]).value;
        }
    }

    for(int i=0; i<w; i++){
        for(int j=0; j<w; j++){
            U[i][j] = into_gal251(U[i][j]).value;
            L[i][j] = into_gal251(L[i][j]).value;
        }
    }

    if(!(gal251_inverse_upper_triangular_matrix(U, U_inv, w))){
        printf("[Error] Can't get inverse\n");
        return;
    }

    if(!(gal251_inverse_lower_triangular_matrix(L, L_inv, w))){
        printf("[Error] Can't get inverse\n");
        return;
    }


    uint8_t** I = allocate_matrix_uint8(w, w);
    uint8_t** LU = allocate_matrix_uint8(w, w);
    uint8_t** UinvLinv = allocate_matrix_uint8(w, w);
    uint8_t** X = allocate_matrix_uint8(h, w);
    uint8_t** Y = allocate_matrix_uint8(h, w);


    gal251_matrix_multiply(L, U, LU, w, w, w);
    gal251_matrix_multiply(U_inv, L_inv, UinvLinv, w, w, w);

    gal251_matrix_multiply(norm_mat, LU, Y, h, w, w);
    gal251_matrix_multiply(Y, UinvLinv, X, h, w, w);

    // gal251_matrix_multiply(norm_mat, U, Y, h, w, w);
    // gal251_matrix_multiply(Y, U_inv, X, h, w, w);

    gal251_matrix_multiply(U, U_inv, I, w, w, w);

    printf("[I]\n");
    for(int i=0; i<w; i++){
        for(int j=0; j<w; j++){
            printf("%02X ",I[i][j]);
        }
        printf("\n");
    }

    printf("[prev L]\n");
    for(int i=0; i<w; i++){
        for(int j=0; j<w; j++){
            printf("%02X ",U[i][j]);
        }
        printf("\n");
    }

    printf("[prev L_inv]\n");
    for(int i=0; i<w; i++){
        for(int j=0; j<w; j++){
            printf("%02X ",U_inv[i][j]);
        }
        printf("\n");
    }


    //=============================================
    // matrix print test
    //=============================================
    printf("[prev X]\n");
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            printf("%02X ",norm_mat[i][j]);
        }
        printf("\n");
    }

    printf("\n[Y]\n");
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            printf("%02X ",Y[i][j]);
        }
        printf("\n");
    }

    printf("\n[after X]\n");
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            printf("%02X ",X[i][j]);
        }
        printf("\n");
    }
    //=============================================
    

    //key buffer free
    key_free(&key_info);

    //matrix buffer free
    
    free_matrix_uint8(norm_mat, h);
    free_matrix_uint8(U, w);
    free_matrix_uint8(L, w);
    free_matrix_uint8(I, w);
    free_matrix_uint8(U_inv, h);
    free_matrix_uint8(L_inv, h);
    free_matrix_uint8(X, h);
    free_matrix_uint8(LU, h);
    free_matrix_uint8(UinvLinv, h);
    free_matrix_uint8(Y, h);

    // 4. model saving stage
    // printf("Saving modified ONNX model: %s\n", out_model_path);
    // save_onnx_model(out_model_path, model);

    // 5. cleanup stage
    // onnx__model_proto__free_unpacked(model, NULL);
    printf("Done!\n");

    return 0;
}