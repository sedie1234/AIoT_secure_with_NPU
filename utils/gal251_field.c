#ifndef GAL251_FIELD_C
#define GAL251_FIELD_C

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#include "gal251_field.h"
#include "matrix.h"

Gal251Field into_gal251(uint8_t num){
    Gal251Field val;
    if(num > 250){
        val.value = num - 128;
        val.m_bit = 1;
    }else{
        val.value = num;
        val.m_bit = 0;
    }

    return val;
}

uint8_t gal251_add(uint8_t num1, uint8_t num2){
    int _num1 = num1;
    int _num2 = num2;
    return (_num1 + _num2) % 251;
}

uint8_t gal251_sub(uint8_t num1, uint8_t num2){
    int _num1 = num1;
    int _num2 = num2;
    return (_num1 - _num2) % 251;
}

uint8_t gal251_mult(uint8_t num1, uint8_t num2){
    int _num1 = num1;
    int _num2 = num2;
    return (_num1 * _num2) % 251;
}

uint8_t gal251_div(uint8_t num1, uint8_t num2){
    int num2_inv = gal251_mult_inverse(num2);
    int val = num2_inv * num1;
    return val % 251;
}

int gal251_extended_gcd(uint8_t num){
    int old_r = num;
    int r = 251;
    int old_s = 1;
    int s = 0;

    while(r != 0){
        int quotient = old_r/r;
        int temp_r = r;
        int temp_s = s;
        r = old_r - quotient * r;
        old_r = temp_r;

        s = old_s - quotient * s;
        old_s  = temp_s;
    }

    if (old_r != 1){
        printf("[Error] number %d has no inverse in gal251 field\n");
        return -1;
    }

    return old_s;
}

uint8_t gal251_mult_inverse(uint8_t num){
    int val = gal251_extended_gcd(num);
     return val % 251;
}

void gal251_matrix_multiply(uint8_t **A, uint8_t **B, uint8_t **C, int m, int n, int p){
    int** _C = allocate_matrix_int(m, p);
    Gal251Op field_operator;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            _C[i][j] = 0;
            for (int k = 0; k < n; k++) {
                _C[i][j] += field_operator.gal251_mult(A[i][k] , B[k][j]);
                if(_C[i][j] > 0x0fffff00)
                    _C[i][j] = _C[i][j] % 251;
            }
            C[i][j] = _C[i][j] % 251;
        }
    }
    free_matrix_int(_C, m);
}

int gal251_inverse_upper_triangular_matrix(uint8_t **A, uint8_t **A_inv, int n){

    Gal251Op Operator;

    //init inverse matrix : unit matrix
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            A_inv[i][j] = (i == j) ? 1 : 0;
        }
    }

    // backward substitution
    for(int i=n-1; i>0; i--){
        if(A[i][i] == 0)
            return 0;

        for(int j=i; j<n; j++){
            if(i==j){
                A_inv[i][j] = Operator.gal251_div(A_inv[i][j], A[i][j]);
            } else {
                for(int k=i+1; k<=j; k++){
                    uint8_t temp = Operator.gal251_mult(A[i][k], A_inv[k][j]);
                    A_inv[i][j] = Operator.gal251_sub(A_inv[i][j], temp);
                }
                A_inv[i][j] = Operator.gal251_div(A_inv[i][j], A[i][i]);
            }
        }
    }

    return 1;
}

int gal251_inverse_lower_triangular_matrix(uint8_t **A, uint8_t **A_inv, int n){
    
    Gal251Op Operator;

    //init inverse matrix : unit matrix
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            A_inv[i][j] = (i == j) ? 1 : 0;
        }
    }

    for(int i=0; i<n; i++){
        if(A[i][i] == 0)
            return 0;
        
        for(int j=0; j<=i; j++){
            if(i == j){
                A_inv[i][j] = Operator.gal251_div(A_inv[i][j], A[i][i]);
            }else{
                for(int k=j; k<i; k++){
                    uint8_t temp = Operator.gal251_mult(A[i][k], A_inv[k][j]);
                    A_inv[i][j] = Operator.gal251_sub(A_inv[i][j], temp);
                }
                A_inv[i][j] = Operator.gal251_div(A_inv[i][j], A[i][i]);
            }
        }
    }
    return 1;
}
#endif //GAL251_FIELD_C