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

Gal251Field into_gal251(uint8_t num);

uint8_t gal251_add(uint8_t num1, uint8_t num2);
uint8_t gal251_sub(uint8_t num1, uint8_t num2);
uint8_t gal251_mult(uint8_t num1, uint8_t num2);
uint8_t gal251_div(uint8_t num1, uint8_t num2);

int gal251_extended_gcd(uint8_t num);
uint8_t gal251_mult_inverse(uint8_t num);

void gal251_matrix_multiply(uint8_t **A, uint8_t **B, uint8_t **C, int m, int n, int p);
int gal251_inverse_upper_triangular_matrix(uint8_t **A, uint8_t **A_inv, int n);
int gal251_inverse_lower_triangular_matrix(uint8_t **A, uint8_t **A_inv, int n);
#endif //GAL251_FIELD_H