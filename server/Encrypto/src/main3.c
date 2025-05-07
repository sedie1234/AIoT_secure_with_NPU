#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

//#include "model_utils.h"
//#include "onnx.pb-c.h"
//#include "encrypt.h"
#include "key_utils.h"
#include "matrix.h"
#include "gal251_field.h"
#include "enc_struct.h"

#define MAC_W 32

int get_random_int(int min, int max);
void printBuffer(uint8_t* buffer, int length, int line_length);

int main(int argc, char *argv[]) {
    srand(time(NULL));
    // 1. argument parsing stage
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <file to encrypt> <output file.enc> <key_path>\n", argv[0]);
        return 1;
    }

    char* filename = argv[1];
    char* out_file_path = argv[2];
    char* key_filename = argv[3];

    FILE* fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return 1;
    }

    fseek(fp, 0, SEEK_END);
    int data_length = ftell(fp);
    rewind(fp);

    FILE* out_file = fopen(out_file_path, "wb");
    if (out_file == NULL) {
        fprintf(stderr, "Failed to open output file: %s\n", out_file_path);
        fclose(fp);
        fclose(out_file);
        return 1;
    }
    
    EncFile enc_file;
    memset(&enc_file, 0, sizeof(enc_file));

   
    enc_file.w = MAC_W;

    // 2. create random data
    uint8_t* data = malloc(data_length);
    if (data == NULL) {
        fprintf(stderr, "Failed to allocate memory for data\n");
        fclose(fp);
        fclose(out_file);
        return 1;
    }

    size_t bytes_read = fread(data, 1, data_length, fp);
    if (bytes_read != data_length) {
        fprintf(stderr, "Failed to read data from file: %s\n", filename);
        free(data);
        fclose(fp);
        fclose(out_file);
        return 1;
    }
    fclose(fp);

    enc_file.file_size = data_length;

    // original data
    printf("Generated random data of length %d\n", data_length);
    printf("[originaldata]\n");
    printBuffer(data, 160, 16);

    // 2.1 into gal251 field
    Gal251Field* gal251_data = into_gal251_buffer(data, data_length);
    uint8_t* gal251_data_bit_buffer;
    int bit_buffer_size = alloc_bit_buffer(&gal251_data_bit_buffer, data_length);
    uint8_t* gal251_data_val_buffer = malloc(data_length);
    set_gal251_buffer(gal251_data, gal251_data_val_buffer, gal251_data_bit_buffer, data_length);

    // gal251 feild data
    printf("[gal251 data]\n");
    printBuffer(gal251_data_val_buffer, 160, 16);

    // 3. load key file
    KeyInfo key_info;
    memset(&key_info, 0, sizeof(key_info));
    int ret = read_key(key_filename, &key_info);
    if (ret == -1) {
        fprintf(stderr, "Failed to read key file: %s\n", key_filename);
        free(data);
        return 1;
    }
    if(key_info.size < MAC_W*MAC_W){
        printf("[Error] Key size is too small\n");
        free(data);
        return 1;
    }

    int upos = get_random_int(0, key_info.size-MAC_W*MAC_W-1);
    int lpos = get_random_int(0, key_info.size-MAC_W*MAC_W-1);
    int bpos = get_random_int(0, key_info.size-1);

    enc_file.upos = upos;
    enc_file.lpos = lpos;
    enc_file.bpos = bpos;
    enc_file.bit_buffer_size = bit_buffer_size;

    // height of matrix
    int h = (data_length * 4 + MAC_W - 1) / MAC_W;
    enc_file.h = h;

    enc_file.size = MAC_W * h;

    // 4. encryption stage
    // get key array : U_inv, L_inv
    uint8_t* U = create_upper_triangular_matrix(key_info, upos, MAC_W);
    uint8_t* L = create_lower_triangular_matrix(key_info, lpos, MAC_W);
    uint8_t* B = create_bias_matrix(key_info, bpos, h, MAC_W);

    uint8_t* U_inv = allocate_matrix_uint8(MAC_W, MAC_W);
    uint8_t* L_inv = allocate_matrix_uint8(MAC_W, MAC_W);
    uint8_t* UinvLinv = allocate_matrix_uint8(MAC_W, MAC_W);
    uint8_t* Y = allocate_matrix_uint8(h, MAC_W);
    uint8_t* Y_ = allocate_matrix_uint8(h, MAC_W);
    
    uint8_t* B_ = allocate_matrix_uint8(h, MAC_W);



    if(!(gal251_inverse_upper_triangular_matrix(U, U_inv, MAC_W))){
        printf("[Error] Can't get inverse\n");
        return 1;
    }

    if(!(gal251_inverse_lower_triangular_matrix(L, L_inv, MAC_W))){
        printf("[Error] Can't get inverse\n");
        return 1;
    }    
    
    // matrix encryption
    gal251_matrix_multiply(U_inv, L_inv, UinvLinv, MAC_W, MAC_W, MAC_W);
    gal251_matrix_multiply(gal251_data_val_buffer, UinvLinv, Y_, h, MAC_W, MAC_W);

    // gal251_matrix_multiply(gal251_data_val_buffer, L_inv, Y_, h, MAC_W, MAC_W);
    // gal251_matrix_multiply(B_, L_inv, Y_, h, MAC_W, MAC_W);

    printf("Y=XUinvLinv\n");
    printBuffer(Y_, 160, 16);


    gal251_matrix_add(Y_, B, Y, h, MAC_W);
    printf("Y=XUinvLinv+B\n");
    printBuffer(Y, 160, 16);

    enc_file.enc_data_val = Y;
    enc_file.enc_data_bit = gal251_data_bit_buffer;

    // // test stage
    // gal251_matrix_sub(Y, B, B_, h, MAC_W);
    // printf("Y-B\n");
    // printBuffer(B_, 160, 16);

    // gal251_matrix_multiply(B_, L, Y_, h, MAC_W, MAC_W);
    // gal251_matrix_multiply(Y_, U, Y, h, MAC_W, MAC_W);
    // printf("X = (Y-B)LU\n");
    // printBuffer(Y, 160, 16);


    // 5. save encrypted data
    // fwrite(&enc_file, sizeof(int), 7, out_file); // Save data size
    fwrite(&enc_file.file_size, sizeof(int), 1, out_file); // Save file size
    fwrite(&enc_file.size, sizeof(int), 1, out_file); // Save data size
    fwrite(&enc_file.w, sizeof(int), 1, out_file); // Save MAC_W
    fwrite(&enc_file.h, sizeof(int), 1, out_file); // Save height
    fwrite(&enc_file.upos, sizeof(int), 1, out_file); // Save upos
    fwrite(&enc_file.lpos, sizeof(int), 1, out_file); // Save lpos
    fwrite(&enc_file.bpos, sizeof(int), 1, out_file); // Save bpos
    fwrite(&enc_file.bit_buffer_size, sizeof(int), 1, out_file); // Save bit buffer size
    fwrite(enc_file.enc_data_val, sizeof(uint8_t), enc_file.size, out_file); // Save bit buffer
    fwrite(enc_file.enc_data_bit, sizeof(uint8_t), enc_file.bit_buffer_size, out_file); // Save encrypted data

    // print encrypted data
    printf("Encrypted data of length 160\n");
    printBuffer(enc_file.enc_data_val, 160, 16);

    // 6. free stage
    fclose(out_file);
    printf("Encrypted data saved to %s\n", out_file_path);
    printf("Encrypted data of length %d\n", enc_file.size);

    free(data);
    free(gal251_data);
    free(gal251_data_val_buffer);
    free(gal251_data_bit_buffer);
    free_matrix_uint8(U, MAC_W);
    free_matrix_uint8(L, MAC_W);
    free_matrix_uint8(B, h);
    free_matrix_uint8(U_inv, MAC_W);
    free_matrix_uint8(L_inv, MAC_W);
    free_matrix_uint8(UinvLinv, MAC_W);
    free_matrix_uint8(Y, h);
    free_matrix_uint8(Y_, h);
    free_matrix_uint8(B_, h);
    key_free(&key_info);

    return 0;
}

int get_random_int(int min, int max) {
    
    return min + rand() % (max - min + 1);
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
