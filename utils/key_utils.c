#ifndef KEY_UTILS_C
#define KEY_UTILS_C

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>

#include  "key_utils.h"


int read_key(char* filename, KeyInfo* key_info){
    if(!strstr(filename,  ".ek")){
        printf("[Error] extension of key is .ek\n");
        KeyInfo errorkey;
        errorkey.size = -1;
        return -1;
    }

    int fd;

    fd = open(filename, O_RDONLY);

    if(fd == -1){
        printf("[Error] File open failed\n");
        KeyInfo errorkey;
        errorkey.size = -1;
        return -1;
    }

    uint8_t byte;
    int64_t keysize=0;

    uint8_t buffer[1024];
    ssize_t bytes_read;

    int idx=0;

    while ((bytes_read = read(fd, buffer, sizeof(buffer))) > 0) {

        key_info->size += bytes_read;
    }

    key_alloc(key_info);

    lseek(fd, 0, SEEK_SET);

    int index = 0;
    while((bytes_read = read(fd, (key_info->key + index), key_info->size))){
        index += bytes_read;
    }

    close(fd);

    return key_info;
}

uint8_t* return_key_array(KeyInfo key_info, int pos, int64_t size){
    if(key_info.size == 0){
        printf("[Error] Not Exist key info\n");
        return NULL;
    }

    if(pos + size > key_info.size){
        printf("[Error] Out of key size\n");
        return NULL;
    }

    uint8_t* key = (uint8_t*)malloc(sizeof(uint8_t)*size);

    for(int i=0; i<size; i++){
        key[i] = key_info.key[pos+i];
    }

    return key;
}

void key_alloc(KeyInfo* key_info){
    if(key_info->size > 0){
        key_info->key = (uint8_t*)malloc(sizeof(uint8_t)*(key_info->size));
        return;
    }else{
        printf("[Error] Not Exist key size info\n");
        return;
    }
}

void key_free(KeyInfo* key_info){
    if(key_info->size > 0){
        free(key_info->key);
        key_info->size = 0;
    }
    return;
}

#endif