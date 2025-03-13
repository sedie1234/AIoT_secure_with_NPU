#ifndef GAL251_FIELD_C
#define GAL251_FIELD_C

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#include "gal251_field.h"
#include "matrix.h"

Gal251Op init_gal251_ops() {
    Gal251Op ops;
    ops.gal251_add = gal251_add;
    ops.gal251_sub = gal251_sub;
    ops.gal251_mult = gal251_mult;
    ops.gal251_div = gal251_div;
    return ops;
}

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
    return (num1 >= num2) ? (num1 - num2)%251 : (251 + num1 - num2)%251;
}

uint8_t gal251_sub(uint8_t num1, uint8_t num2){
    int _num1 = num1;
    int _num2 = num2;
    if(num1 < num2){
        return (251 + num1 - num2)%251;
    }else{
        return (num1 - num2)%251;
    }
}

uint8_t gal251_mult(uint8_t num1, uint8_t num2){
    int _num1 = num1;
    int _num2 = num2;
    return (_num1 * _num2) % 251;
}

uint8_t gal251_div(uint8_t num1, uint8_t num2){
    if (num2 == 0) {
        printf("[Error] Division by zero in GF(251)\n");
        exit(EXIT_FAILURE);
    }
    uint8_t num2_inv = gal251_mult_inverse(num2);
    return gal251_mult(num1, num2_inv);
}

int gal251_extended_gcd(int a, int p, int *x, int *y) {
    if (a == 0) {
        *x = 0;
        *y = 1;
        return p;
    }
    int x1, y1;
    int gcd = gal251_extended_gcd(p % a, a, &x1, &y1);
    *x = y1 - (p / a) * x1;
    *y = x1;
    return gcd;
}

uint8_t gal251_mult_inverse(uint8_t a) {
    if (a == 0) {
        printf("[Error] Zero has no multiplicative inverse in GF(251)\n");
        exit(EXIT_FAILURE);
    }
    int x, y;
    int gcd = gal251_extended_gcd(a, 251, &x, &y);
    if (gcd != 1) {
        printf("[Error] %d has no inverse in GF(251)\n", a);
        exit(EXIT_FAILURE);
    }
    return (x % 251 + 251) % 251;
}

// void gal251_matrix_multiply(uint8_t **A, uint8_t **B, uint8_t **C, int m, int n, int p){
//     int** _C = allocate_matrix_int(m, p);
//     Gal251Op field_operator = init_gal251_ops();

//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < p; j++) {
//             _C[i][j] = 0;
//             for (int k = 0; k < n; k++) {
//                 _C[i][j] += field_operator.gal251_mult(A[i][k] , B[k][j]);
//                 if(_C[i][j] > 0x0fffff00)
//                     _C[i][j] = _C[i][j] % 251;
//             }
//             C[i][j] = _C[i][j] % 251;
//         }
//     }
//     free_matrix_int(_C, m);
// }

// void gal251_matrix_add(uint8_t **A, uint8_t **B, uint8_t **C, int m, int n){
//     for(int i=0; i<m; i++){
//         for(int j=0; j<n; j++){
//             C[i][j] = gal251_add(A[i][j], B[i][j]);
//         }
//     }
// }

void gal251_vector_add(uint8_t *A, uint8_t *B, uint8_t *C, int n){
    for(int i=0; i<n; i++){
        C[i] = gal251_add(A[i], B[i]);
    }
}

// int gal251_inverse_upper_triangular_matrix(uint8_t **A, uint8_t **A_inv, int n){

//     Gal251Op Operator = init_gal251_ops();

//     //init inverse matrix : unit matrix
//     for(int i=0; i<n; i++){
//         for(int j=0; j<n; j++){
//             A_inv[i][j] = (i == j) ? 1 : 0;
//         }
//     }

//     // backward substitution
//     for(int i=n-1; i>=0; i--){

//         //error
//         if(A[i][i] == 0)
//             return 0;

//         for(int j=i; j<n; j++){
//             if(i==j){
//                 A_inv[i][j] = Operator.gal251_div(A_inv[i][j], A[i][j]);
//             } else {
//                 for(int k=i+1; k<=j; k++){
//                     uint8_t temp = Operator.gal251_mult(A[i][k], A_inv[k][j]);
//                     A_inv[i][j] = Operator.gal251_sub(A_inv[i][j], temp);
//                 }
//                 A_inv[i][j] = Operator.gal251_div(A_inv[i][j], A[i][i]);
//             }
//         }
//     }

//     return 1;
// }


// int gal251_inverse_lower_triangular_matrix(uint8_t **A, uint8_t **A_inv, int n){
    
//     Gal251Op Operator = init_gal251_ops();

//     //init inverse matrix : unit matrix
//     for(int i=0; i<n; i++){
//         for(int j=0; j<n; j++){
//             A_inv[i][j] = (i == j) ? 1 : 0;
//         }
//     }

//     for(int i=0; i<n; i++){
//         if(A[i][i] == 0)
//             return 0;
        
//         for(int j=0; j<=i; j++){
//             if(i == j){
//                 A_inv[i][j] = Operator.gal251_div(A_inv[i][j], A[i][i]);
//             }else{
//                 for(int k=j; k<i; k++){
//                     uint8_t temp = Operator.gal251_mult(A[i][k], A_inv[k][j]);
//                     A_inv[i][j] = Operator.gal251_sub(A_inv[i][j], temp);
//                 }
//                 A_inv[i][j] = Operator.gal251_div(A_inv[i][j], A[i][i]);
//             }
//         }
//     }
//     return 1;
// }

void gal251_matrix_multiply(uint8_t *A, uint8_t *B, uint8_t *C, int m, int n, int p){
    int* _C = allocate_matrix_int(m, p);
    Gal251Op field_operator = init_gal251_ops();

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            _C[i*p+j] = 0;
            for (int k = 0; k < n; k++) {
                _C[i*p+j] += field_operator.gal251_mult(A[i*n+k] , B[k*p+j]);
                if(_C[i*p+j] > 0x0fffff00)
                    _C[i*p+j] = _C[i*p+j] % 251;
            }
            C[i*p+j] = _C[i*p+j] % 251;
        }
    }
    free_matrix_int(_C, m);
}

void gal251_matrix_add(uint8_t *A, uint8_t *B, uint8_t *C, int m, int n){
    for(int i=0; i<m*n; i++){
        C[i] = gal251_add(A[i], B[i]);
    }
}

int gal251_inverse_upper_triangular_matrix(uint8_t *A, uint8_t *A_inv, int n){
    
    Gal251Op Operator = init_gal251_ops();

    //init inverse matrix : unit matrix
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            A_inv[i*n+j] = (i == j) ? 1 : 0;
        }
    }

    // backward substitution
    for(int i=n-1; i>=0; i--){

        //error
        if(A[i*n+i] == 0)
            return 0;

        for(int j=i; j<n; j++){
            if(i==j){
                A_inv[i*n+j] = Operator.gal251_div(A_inv[i*n+j], A[i*n+j]);
            } else {
                for(int k=i+1; k<=j; k++){
                    uint8_t temp = Operator.gal251_mult(A[i*n+k], A_inv[k*n+j]);
                    A_inv[i*n+j] = Operator.gal251_sub(A_inv[i*n+j], temp);
                }
                A_inv[i*n+j] = Operator.gal251_div(A_inv[i*n+j], A[i*n+i]);
            }
        }
    }

    return 1;
}


int gal251_inverse_lower_triangular_matrix(uint8_t *A, uint8_t *A_inv, int n){
    
    Gal251Op Operator = init_gal251_ops();

    //init inverse matrix : unit matrix
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            A_inv[i*n+j] = (i == j) ? 1 : 0;
        }
    }

    for(int i=0; i<n; i++){
        if(A[i*n+i] == 0)
            return 0;
        
        for(int j=0; j<=i; j++){
            if(i == j){
                A_inv[i*n+j] = Operator.gal251_div(A_inv[i*n+j], A[i*n+i]);
            }else{
                for(int k=j; k<i; k++){
                    uint8_t temp = Operator.gal251_mult(A[i*n+k], A_inv[k*n+j]);
                    A_inv[i*n+j] = Operator.gal251_sub(A_inv[i*n+j], temp);
                }
                A_inv[i*n+j] = Operator.gal251_div(A_inv[i*n+j], A[i*n+i]);
            }
        }
    }
    return 1;
}

#endif //GAL251_FIELD_C