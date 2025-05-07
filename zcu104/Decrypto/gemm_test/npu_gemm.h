#ifndef _NPU_GEMM_H
#define _NPU_GEMM_H

//#define GEMM_CTRL_BASE		0x34000000
//#define GEMM_MEM_BASE		0x50000000
#define GEMM_CTRL_BASE		0xa1000000
#define GEMM_MEM_BASE		0xa2000000

#define GEMM_CTRL_SIZE		0x20000	/* 4k x 16 = 128k */
#define GEMM_MEM_BAKE_SIZE	0x20000
#define GEMM_MEM_SIZE	(GEMM_MEM_BAKE_SIZE * 9)	/* 9 bank of 128k */

#define GEMM_NPU_START		0x3C
#define GEMM_NPU_END		0x40
#define GEMM_ACT_PARAM_0	0x0C
#define GEMM_ACT_PARAM_1	0x10
#define GEMM_WEG_PARAM_0	0x1C
#define GEMM_WEG_PARAM_1	0x20
#define GEMM_PSUM_PARAM_0	0x2C

struct mem_npu {
	void *mem_base;
	volatile void *ctrl_base;
	int npu_fd;
	pthread_mutex_t mutex_lock;

	int8_t *in_a1;
	int8_t *in_a2;
	int8_t *in_a3;
	int8_t *in_b1;
	int8_t *in_b2;
	int32_t *out_c1;
	int32_t *out_c2;
	int32_t *out_c3;
	int32_t *out_c4;

	int8_t  *dma_buf;
	int32_t *dma_c1;
	int32_t *dma_c2;
	int32_t *dma_c3;
	int32_t *dma_c4;

};

void gemm_nn_npu(int M, int N, int K, uint8_t ALPHA,
		   uint8_t * A, int lda,
		   uint8_t * B, int ldb, int32_t * C, int ldc);
int init_kgemm();

#ifdef DEBUG
#define npu_printf(fmt, ...) printf((fmt), ##__VA_ARGS__)

#else
#define npu_printf(fmt, ...)

#endif

#endif /* !_NPU_GEMM_H */
