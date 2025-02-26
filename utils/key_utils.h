#ifndef KEY_UTILS_H
#define KEY_UTILS_H

#include <stdint.h>

#define READ_BLOCK_SIZE     2048

typedef struct{
    uint8_t* key;
    int64_t size;
} KeyInfo;

int read_key(char* filename, KeyInfo* key_info);
uint8_t* return_key_array(KeyInfo key_info, int pos, int64_t size);
void key_alloc(KeyInfo* key_info);
void key_free(KeyInfo* key_info);


#endif