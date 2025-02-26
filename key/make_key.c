#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

#define BUFFER_SIZE 4096  // 4KB buffer

void generate_random_data(const char *filename, size_t size);

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <size> <filename>\n", argv[0]);
        return 1;
    }

    size_t size = strtoul(argv[1], NULL, 10);
    const char *filename = argv[2];

    if (size == 0) {
        fprintf(stderr, "[Error] size = 0\n");
        return 1;
    }

    generate_random_data(filename, size);
    return 0;
}

void generate_random_data(const char *filename, size_t size) {
    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd == -1) {
        perror("[Error]File open failed\n");
        exit(1);
    }

    srand((unsigned int)time(NULL));  

    uint8_t buffer[BUFFER_SIZE];  
    size_t remaining = size;

    while (remaining > 0) {
        size_t chunk_size = (remaining < BUFFER_SIZE) ? remaining : BUFFER_SIZE;
        for (size_t i = 0; i < chunk_size; i++) {
            buffer[i] = (rand() % 250) + 1;  
        }
        write(fd, buffer, chunk_size);  
        remaining -= chunk_size;
    }

    close(fd);
    printf("%zu byte save in key file '%s'\n", size, filename);
}
