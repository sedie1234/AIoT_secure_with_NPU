#ifndef GAL251_FIELD_H
#define GAL251_FIELD_H

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    uint8_t value;
    bool m_bit;
}Gal251Field;

typedef struct {
    uint8_t (*gal251_add)(uint8_t, uint8_t);
    uint8_t (*gal251_sub)(uint8_t, uint8_t);
    uint8_t (*gal251_mult)(uint8_t, uint8_t);
    uint8_t (*gal251_div)(uint8_t, uint8_t);
}Gal251Op;

Gal251Op init_gal251_ops();

Gal251Field into_gal251(uint8_t num);
Gal251Field* into_gal251_buffer(uint8_t* num, int size); // malloc
int alloc_bit_buffer(uint8_t** bit_buffer, int size); // malloc
void set_gal251_buffer(Gal251Field *val, uint8_t* val_buffer, uint8_t* bit_buffer, int size);
void gal251_to_256(uint8_t** val, uint8_t* val_buffer, uint8_t* bit_buffer, int size);  // malloc

uint8_t gal251_add(uint8_t num1, uint8_t num2);
uint8_t gal251_sub(uint8_t num1, uint8_t num2);
uint8_t gal251_mult(uint8_t num1, uint8_t num2);
uint8_t gal251_div(uint8_t num1, uint8_t num2);

int gal251_extended_gcd(int a, int p, int *x, int *y);
uint8_t gal251_mult_inverse(uint8_t num);

void gal251_matrix_multiply(uint8_t *A, uint8_t *B, uint8_t *C, int m, int n, int p);
void gal251_matrix_add(uint8_t *A, uint8_t *B, uint8_t *C, int m, int n);
void gal251_matrix_sub(uint8_t *A, uint8_t *B, uint8_t *C, int m, int n);
void gal251_vector_add(uint8_t *A, uint8_t *B, uint8_t *C, int n);
void gal251_vector_sub(uint8_t *A, uint8_t *B, uint8_t *C, int n);

int gal251_inverse_upper_triangular_matrix(uint8_t *A, uint8_t *A_inv, int n);
int gal251_inverse_lower_triangular_matrix(uint8_t *A, uint8_t *A_inv, int n);
#endif //GAL251_FIELD_H
