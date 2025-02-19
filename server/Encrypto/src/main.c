#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model_utils.h"
#include "onnx.pb-c.h"
#include "encrypt.h"

int main(int argc, char *argv[]) {

    // 1. argument parsing stage
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_onnx> <output_onnx>\n", argv[0]);
        return 1;
    }

    char* model_path = argv[1];
    char* out_model_path = argv[2];

    // 2. model loading stage
    printf("Loading ONNX model: %s\n", model_path);
    Onnx__ModelProto *model = load_onnx_model(model_path);
    if (!model) {
        return 1;
    }

    // 3. model modification stage
    int w = 32;
    shuffle_nodes(model);
    shuffle_initializers(model);
    uint32_t** mat = create_conninfo_mat(model);    
    uint8_t** norm_mat = mat_normalize(mat, model, w);

    int64_t mat_size=0;
    int64_t n_nodes = get_node_size(model);
    int64_t n_inits = get_initializer_size(model);

    for(int i=0; i<n_nodes; i++){
        int64_t n_inputs = model->graph->node[i]->n_input;
        int64_t n_outputs = model->graph->node[i]->n_output;
        for(int j=0; j<n_inputs+n_outputs; j++){
            mat_size++;
        }
    }

    int h = (mat_size*4 +w -1) /w;

    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            printf("%02X ",norm_mat[i][j]);
        }
        printf("\n");
    }
    free(mat);
    free(norm_mat);

    // 4. model saving stage
    // printf("Saving modified ONNX model: %s\n", out_model_path);
    // save_onnx_model(out_model_path, model);

    // 5. cleanup stage
    onnx__model_proto__free_unpacked(model, NULL);
    printf("Done!\n");

    return 0;
}