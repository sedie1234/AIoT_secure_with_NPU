#ifndef MATRIX_C
#define MATRIX_C

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "matrix.h"
#include "key_utils.h"

// void print_matrix_uint8(uint8_t **matrix, int rows, int cols) {
//     for (int i = 0; i < rows; i++) {
//         for (int j = 0; j < cols; j++) {
//             printf("%d ", matrix[i][j]);
//         }
//         printf("\n");
//     }
// }

// void print_matrix_int(int **matrix, int rows, int cols) {
//     for (int i = 0; i < rows; i++) {
//         for (int j = 0; j < cols; j++) {
//             printf("%d ", matrix[i][j]);
//         }
//         printf("\n");
//     }
// }

// uint8_t **allocate_matrix_uint8(int rows, int cols) {
//     uint8_t **matrix = (uint8_t**)malloc(rows * sizeof(uint8_t *));
//     for (int i = 0; i < rows; i++) {
//         *(matrix + i) = (uint8_t*)malloc(cols * sizeof(uint8_t));
//     }
//     return matrix;
// }

// int **allocate_matrix_int(int rows, int cols) {
//     int **matrix = malloc(rows * sizeof(int *));
//     for (int i = 0; i < rows; i++) {
//         matrix[i] = malloc(cols * sizeof(int));
//     }
//     return matrix;
// }

// void free_matrix_uint8(uint8_t **matrix, int rows) {
//     for (int i = 0; i < rows; i++) {
//         free(matrix[i]);
//         matrix[i] = NULL; // prevent double free
//     }
//     free(matrix);
//     matrix = NULL; // prevent double free
// }

// void free_matrix_int(int **matrix, int rows){
//     for (int i = 0; i < rows; i++) {
//         free(matrix[i]);
//         matrix[i] = NULL; // prevent double free
//     }
//     free(matrix);
//     matrix = NULL; // prevent double free
// }

// uint8_t** create_upper_triangular_matrix(KeyInfo key, int pos, int n){

//     if(key.size < (n*n - n)/2 + n/2 + n%2){
//         printf("[Error] key size is not enough\n");
//         printf("size : %ld\n", key.size);
//         printf("n : %d\n", n);
//         return NULL;
//     }
//     uint8_t** _A = allocate_matrix_uint8(n, n);

//     int index = pos;
//     for(int i=0; i<n; i++){
//         for(int j=0; j<n; j++){
//             if(i>j){
//                 _A[i][j] = 0;
//             }else{
//                 _A[i][j] = *(key.key+index);
//                 index++;
//                 if(index > key.size){
//                     return NULL;
//                 }
//             }
//         }
//     }

//     return _A;
// }

// uint8_t** create_lower_triangular_matrix(KeyInfo key, int pos, int n){

//     if(key.size < (n*n - n)/2 + n/2 + n%2){
//         printf("[Error] key size is not enough\n");
//         printf("size : %ld\n", key.size);
//         printf("n : %d\n", n);
//         return NULL;
//     }
//     uint8_t** _A = allocate_matrix_uint8(n, n);

//     int index = pos;
//     for(int i=0; i<n; i++){
//         for(int j=0; j<n; j++){
//             if(i<j){
//                 _A[i][j] = 0;
//             }else{
//                 _A[i][j] = *(key.key+index);
//                 index++;
//                 if(index > key.size){
//                     return NULL;
//                 }
//             }
//         }
//     }

//     return _A;
// }

// uint8_t** create_bias_matrix(KeyInfo key, int pos, int h, int w){
//     if(key.size < h*w){
//         printf("[Error] key size is not enough\n");
//         printf("size : %ld\n", key.size);
//         printf("h : %d\n", h);
//         printf("w : %d\n", w);
//         return NULL;
//     }

//     uint8_t** _A = allocate_matrix_uint8(h, w);

//     int index = pos;
//     for(int i=0; i<h; i++){
//         for(int j=0; j<w; j++){
//             _A[i][j] = *(key.key+index);
//             index++;
//             if(index > key.size){
//                 return NULL;
//             }
//         }
//     }

//     return _A;
// }

//=========================================================================

void print_matrix_uint8(uint8_t *matrix, int rows, int cols){
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            printf("%d ", matrix[i*cols+j]);
        }
        printf("\n");
    }
}

void print_matrix_int(int *matrix, int rows, int cols){
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            printf("%d ", matrix[i*cols+j]);
        }
        printf("\n");
    }
}

uint8_t* allocate_matrix_uint8(int rows, int cols){
    uint8_t* matrix = (uint8_t*)malloc(rows * cols * sizeof(uint8_t));
    return matrix;
}

int* allocate_matrix_int(int rows, int cols){
    int* matrix = (int*)malloc(rows * cols * sizeof(int));
    return matrix;
}

void free_matrix_uint8(uint8_t *matrix, int rows){
    free(matrix);
    matrix = NULL; // prevent double free
}

void free_matrix_int(int *matrix, int rows){
    free(matrix);
    matrix = NULL; // prevent double free
}

uint8_t* create_upper_triangular_matrix(KeyInfo key, int pos, int n){
    if(key.size < (n*n - n)/2 + n/2 + n%2){
        printf("[Error] key size is not enough\n");
        printf("size : %ld\n", key.size);
        printf("n : %d\n", n);
        return NULL;
    }
    uint8_t* _A = allocate_matrix_uint8(n, n);

    int index = pos;
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            if(i>j){
                _A[i*n+j] = 0;
            }else{
                _A[i*n+j] = *(key.key+index);
                index++;
                if(index > key.size){
                    return NULL;
                }
            }
        }
    }

    return _A;
}

uint8_t* create_lower_triangular_matrix(KeyInfo key, int pos, int n){
    if(key.size < (n*n - n)/2 + n/2 + n%2){
        printf("[Error] key size is not enough\n");
        printf("size : %ld\n", key.size);
        printf("n : %d\n", n);
        return NULL;
    }
    uint8_t* _A = allocate_matrix_uint8(n, n);

    int index = pos;
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            if(i<j){
                _A[i*n+j] = 0;
            }else{
                _A[i*n+j] = *(key.key+index);
                index++;
                if(index > key.size){
                    return NULL;
                }
            }
        }
    }

    return _A;
}

uint8_t* create_bias_matrix(KeyInfo key, int pos, int h, int w){
    if(key.size < h*w){
        printf("[Error] key size is not enough\n");
        printf("size : %ld\n", key.size);
        printf("h : %d\n", h);
        printf("w : %d\n", w);
        return NULL;
    }

    uint8_t* _A = allocate_matrix_uint8(h, w);

    int index = pos;
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            _A[i*w+j] = *(key.key+index);
            index++;
            if(index > key.size){
                return NULL;
            }
        }
    }

    return _A;
}

#endif //MATRIX_C