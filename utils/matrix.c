#ifndef MATRIX_C
#define MATRIX_C

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "matrix.h"

void print_matrix_uint8(uint8_t **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

void print_matrix_int(int **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

uint8_t **allocate_matrix_uint8(int rows, int cols) {
    uint8_t **matrix = (uint8_t**)malloc(rows * sizeof(uint8_t *));
    for (int i = 0; i < rows; i++) {
        *(matrix + i) = (uint8_t*)malloc(cols * sizeof(uint8_t));
    }
    return matrix;
}

int **allocate_matrix_int(int rows, int cols) {
    int **matrix = malloc(rows * sizeof(int *));
    for (int i = 0; i < rows; i++) {
        matrix[i] = malloc(cols * sizeof(int));
    }
    return matrix;
}

void free_matrix_uint8(uint8_t **matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
        matrix[i] = NULL; // prevent double free
    }
    free(matrix);
    matrix = NULL; // prevent double free
}

void free_matrix_int(int **matrix, int rows){
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
        matrix[i] = NULL; // prevent double free
    }
    free(matrix);
    matrix = NULL; // prevent double free
}

uint8_t** create_upper_triangular_matrix(uint8_t* A, int64_t size, int n){

    if(size < (n*n - n)/2 + n/2 + n%2){
        printf("[Error] key size is not enough\n");
        printf("size : %ld\n", size);
        printf("n : %d\n", n);
        return NULL;
    }
    uint8_t** _A = allocate_matrix_uint8(n, n);

    int index = 0;
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            if(i>j){
                _A[i][j] = 0;
            }else{
                _A[i][j] = *(A+index);
                index++;
                if(index > size){
                    return NULL;
                }
            }
        }
    }

    return _A;
}

uint8_t** create_lower_triangular_matrix(uint8_t* A, int64_t size, int n){

    if(size < (n*n - n)/2 + n/2 + n%2){
        printf("[Error] key size is not enough\n");
        printf("size : %ld\n", size);
        printf("n : %d\n", n);
        return NULL;
    }
    uint8_t** _A = allocate_matrix_uint8(n, n);

    int index = 0;
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            if(i<j){
                _A[i][j] = 0;
            }else{
                _A[i][j] = *(A+index);
                index++;
                if(index > size){
                    return NULL;
                }
            }
        }
    }

    return _A;

}

#endif //MATRIX_C