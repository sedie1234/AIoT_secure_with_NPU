#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void print_matrix_uint8(uint8_t **matrix, int rows, int cols);
void print_matrix_int(int **matrix, int rows, int cols);

uint8_t** allocate_matrix_uint8(int rows, int cols);
int** allocate_matrix_int(int rows, int cols);

void free_matrix_uint8(uint8_t **matrix, int rows);
void free_matrix_int(int **matrix, int rows);

uint8_t** create_upper_triangular_matrix(uint8_t* A, int64_t size, int n);
uint8_t** create_lower_triangular_matrix(uint8_t* A, int64_t size, int n);

#endif //MATRIX_H