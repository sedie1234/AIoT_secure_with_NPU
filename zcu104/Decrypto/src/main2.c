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

void printBuffer(uint8_t* buffer, int length, int line_length);

int main(int argc, char *argv[]) {

    // 1. argument parsing stage
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <enc_file.enc> <key_path>\n", argv[0]);
        return 1;
    }

    char* enc_file_path = argv[1];
    char* key_filename = argv[2];

    // 2. load enc file
    EncFile enc_file;
    memset(&enc_file, 0, sizeof(enc_file));
    FILE* enc_file_ptr = fopen(enc_file_path, "rb");
    if (enc_file_ptr == NULL) {
        fprintf(stderr, "Failed to open enc file: %s\n", enc_file_path);
        return 1;
    }
    fread(&enc_file.size, sizeof(int), 1, enc_file_ptr);
    fread(&enc_file.w, sizeof(int), 1, enc_file_ptr);
    fread(&enc_file.h, sizeof(int), 1, enc_file_ptr);
    fread(&enc_file.upos, sizeof(int), 1, enc_file_ptr);
    fread(&enc_file.lpos, sizeof(int), 1, enc_file_ptr);
    fread(&enc_file.bpos, sizeof(int), 1, enc_file_ptr);
    fread(&enc_file.bit_buffer_size, sizeof(int), 1, enc_file_ptr);
    enc_file.enc_data_val = malloc(enc_file.size);
    enc_file.enc_data_bit = malloc(enc_file.bit_buffer_size);
    if (enc_file.enc_data_val == NULL || enc_file.enc_data_bit == NULL) {
        fprintf(stderr, "Failed to allocate memory for enc data\n");
        fclose(enc_file_ptr);
        return 1;
    }
    fread(enc_file.enc_data_val, sizeof(uint8_t), enc_file.size, enc_file_ptr);
    fread(enc_file.enc_data_bit, sizeof(uint8_t), enc_file.bit_buffer_size, enc_file_ptr);
    fclose(enc_file_ptr);
    printf("Loaded enc file: %s\n", enc_file_path);
    printf("Enc file size: %d\n", enc_file.size);
    
    // 3. load key file
    KeyInfo key_info;
    memset(&key_info, 0, sizeof(key_info));
    int ret = read_key(key_filename, &key_info);
    if (ret == -1) {
        fprintf(stderr, "Failed to read key file: %s\n", key_filename);
        free(enc_file.enc_data_val);
        free(enc_file.enc_data_bit);
        return 1;
    }
    if (key_info.size < enc_file.w * enc_file.w) {
        printf("[Error] Key size is too small\n");
        free(enc_file.enc_data_val);
        free(enc_file.enc_data_bit);
        return 1;
    }

    // 4. create L, U, B
    uint8_t* U = create_upper_triangular_matrix(key_info, enc_file.upos, enc_file.w);
    uint8_t* L = create_lower_triangular_matrix(key_info, enc_file.lpos, enc_file.w);
    uint8_t* B = create_bias_matrix(key_info, enc_file.bpos, enc_file.h, enc_file.w);
    uint8_t* LU = allocate_matrix_uint8(enc_file.w, enc_file.w);
    uint8_t* X = allocate_matrix_uint8(enc_file.h, enc_file.w);
    uint8_t* X_ = allocate_matrix_uint8(enc_file.h, enc_file.w);

    
    printf("Y\n");
    printBuffer(enc_file.enc_data_val, 160, 16);

    // 5. decrypt stage
    gal251_matrix_multiply(L, U, LU, enc_file.w, enc_file.w, enc_file.w);
    gal251_matrix_sub(enc_file.enc_data_val, B, X_, enc_file.h, enc_file.w);
    
    printf("Y-B\n");
    printBuffer(X_, 160, 16);

    gal251_matrix_multiply(X_, LU, X, enc_file.h, enc_file.w, enc_file.w);
    printf("X = (Y-B)LU\n");
    printBuffer(X, 160, 16);
    
    uint8_t* decrypt_data;
    gal251_to_256(&decrypt_data, X, enc_file.enc_data_bit, enc_file.size);

    // 6. print decrypted data
    printf("Decrypted data:\n");
    printBuffer(decrypt_data, 160, 16);

    // 7. free stage
    free(enc_file.enc_data_val);
    free(enc_file.enc_data_bit);
    free(decrypt_data);
    free_matrix_uint8(U, enc_file.w);
    free_matrix_uint8(L, enc_file.w);
    free_matrix_uint8(B, enc_file.h);
    free_matrix_uint8(LU, enc_file.w);
    free_matrix_uint8(X, enc_file.h);
    key_free(&key_info);

    return 0;
}


void printBuffer(uint8_t* buffer, int length, int line_length) {
    for (int i = 0; i < length; i++) {
        printf("%02X ", buffer[i]);
        if (i%line_length == line_length-1) {
            printf("\n");
        }
    }
    printf("\n");
}
