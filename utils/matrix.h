#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "key_utils.h"

// void print_matrix_uint8(uint8_t **matrix, int rows, int cols);
// void print_matrix_int(int **matrix, int rows, int cols);

// in the end of usage, must call the free function
// uint8_t** allocate_matrix_uint8(int rows, int cols);
// int** allocate_matrix_int(int rows, int cols);

// void free_matrix_uint8(uint8_t **matrix, int rows);
// void free_matrix_int(int **matrix, int rows);

// in the end of usage, must call the free function
// uint8_t** create_upper_triangular_matrix(KeyInfo key, int pos, int n);
// uint8_t** create_lower_triangular_matrix(KeyInfo key, int pos, int n);

// in the end of usage, must call the free function
// uint8_t** create_bias_matrix(KeyInfo key, int pos, int h, int w);

//=========================================================================
void print_matrix_uint8(uint8_t *matrix, int rows, int cols);
void print_matrix_int(int *matrix, int rows, int cols);

// in the end of usage, must call the free function
uint8_t* allocate_matrix_uint8(int rows, int cols);
int* allocate_matrix_int(int rows, int cols);

void free_matrix_uint8(uint8_t *matrix, int rows);
void free_matrix_int(int *matrix, int rows);

// in the end of usage, must call the free function
uint8_t* create_upper_triangular_matrix(KeyInfo key, int pos, int n);
uint8_t* create_lower_triangular_matrix(KeyInfo key, int pos, int n);

// in the end of usage, must call the free function
uint8_t* create_bias_matrix(KeyInfo key, int pos, int h, int w);

#endif //MATRIX_H