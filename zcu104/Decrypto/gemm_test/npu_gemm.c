#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <fcntl.h>
#include <ctype.h>
#include <termios.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <pthread.h>

#include "npu_gemm.h"
#include "npu_ioctl.h"

struct mem_npu *knpu;

#define TS	32
#define NPU_DEV_NAME	"/dev/keti-npu"

#define FATAL do { fprintf(stderr, "Error at line %d, file %s (%d) [%s]\n", \
  __LINE__, __FILE__, errno, strerror(errno)); exit(1); } while(0)

int init_kgemm()
{
	int fd, fd2;

	knpu = (struct mem_npu *)malloc(sizeof(struct mem_npu));
	memset(knpu, 0, sizeof(struct mem_npu));

#ifdef __x86_64__
	knpu->mem_base =
	    mmap(NULL, GEMM_MEM_SIZE, PROT_READ | PROT_WRITE,
		 MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
	knpu->ctrl_base =
	    mmap(NULL, GEMM_MEM_SIZE, PROT_READ | PROT_WRITE,
		 MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
#else
	if((fd = open("/dev/mem", O_RDWR | O_SYNC)) == -1) FATAL;
	if((fd2 = open("/dev/mem", O_RDWR | O_SYNC)) == -1) FATAL;

	knpu->mem_base = mmap(0, GEMM_MEM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, GEMM_MEM_BASE);
	knpu->ctrl_base = mmap(0, GEMM_CTRL_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd2, GEMM_CTRL_BASE);
#endif

	if (knpu->mem_base == (void *)-1)
		FATAL;
	if (knpu->ctrl_base == (void *)-1)
		FATAL;

	npu_printf("npu mem base :  phy:0x%x vir:%p\n", GEMM_MEM_BASE,
	       knpu->mem_base);
	npu_printf("npu ctrl base:  phy:0x%x vir:%p\n", GEMM_CTRL_BASE,
	       knpu->ctrl_base);

	knpu->in_a1 = knpu->mem_base;	/* bank 0 */
	knpu->in_a2 = knpu->mem_base + GEMM_MEM_BAKE_SIZE;
	knpu->in_a3 = knpu->mem_base + GEMM_MEM_BAKE_SIZE * 2;
	knpu->in_b1 = knpu->mem_base + GEMM_MEM_BAKE_SIZE * 3;
	knpu->in_b2 = knpu->mem_base + GEMM_MEM_BAKE_SIZE * 4;
	knpu->out_c1 = knpu->mem_base + GEMM_MEM_BAKE_SIZE * 5;
	knpu->out_c2 = knpu->mem_base + GEMM_MEM_BAKE_SIZE * 6;
	knpu->out_c3 = knpu->mem_base + GEMM_MEM_BAKE_SIZE * 7;
	knpu->out_c4 = knpu->mem_base + GEMM_MEM_BAKE_SIZE * 8;

	knpu->dma_c1 = valloc(GEMM_MEM_BAKE_SIZE);
	knpu->dma_c2 = valloc(GEMM_MEM_BAKE_SIZE);
	knpu->dma_c3 = valloc(GEMM_MEM_BAKE_SIZE);
	knpu->dma_c4 = valloc(GEMM_MEM_BAKE_SIZE);

	/* share buffer with c1 */
	knpu->dma_buf = (int8_t *) knpu->dma_c1;

#ifdef USE_DMA
	knpu->npu_fd = open(NPU_DEV_NAME, O_RDWR);
	if(knpu->npu_fd < 0)
		FATAL;
#endif
	pthread_mutex_init(&knpu->mutex_lock, NULL);

	return 0;
}

int npu_ctrl_write(uint32_t reg, uint32_t val)
{
	volatile uint32_t *addr;

	addr = (volatile uint32_t *) (knpu->ctrl_base + reg);
	*addr = val;

	return 0;
}

uint32_t npu_ctrl_read(uint32_t reg)
{
	volatile uint32_t *addr;

	addr = (volatile uint32_t *) (knpu->ctrl_base + reg);

	return *addr;
}

void print_uint8_array(int row, int col, uint8_t * c, int size)
{
	int i, j;

	printf("-----------------------------------------------------------\n");
	printf("print matrix: (row, col):(%d %d)\n", row, col);
	for (i = 0; i < row; i++) {
		printf("%4d: ", i);
		for (j = 0; j < col; j++) {
			if( i*col + j >= size) {
				printf("\n");
				return;
			}

			if(j%16 == 0)
				printf("\n    ");
			printf("%3u ", *(c + i*col + j));
		}
		printf("\n");
	}

}

void print_int32_array(int row, int col, int32_t * c, int size)
{
	int i, j;

	printf("-----------------------------------------------------------\n");
	printf("print matrix: (row, col):(%d %d)\n", row, col);
	for (i = 0; i < row; i++) {
		printf("%4d: ", i);
		for (j = 0; j < col; j++) {
			if( i*col + j >= size) {
				printf("\n");
				return;
			}

			if(j%8 == 0)
				printf("\n    ");
			printf("%8d ", *(c + i*col + j));
		}
		printf("\n");
	}

}


void memcpy_add(void *dest, void *src, size_t n)
{
	void *buf;
	int i;

	buf = malloc(n);
	if(buf == NULL) FATAL;

	memcpy(buf, src, n);

	//for(i=0; i<n; i++)
	//	*(char *)(dest+i) += *(char *)(buf +i);
	for(i=0; i<n/4; i++)
		*((uint32_t *)dest+i) += *((uint32_t *)buf +i);

	free(buf);

}

void gemm_nn_sub(int M, int N, int K, uint8_t * A, uint8_t * B, int32_t * C)
{
	int i;
	uint32_t status;
	int16_t mtiles, ntiles, ktiles;
	uint8_t *bufa, *bufb;

	/* int division round up */
	mtiles = (M + TS - 1) / TS;
	ntiles = (N + TS - 1) / TS;
	ktiles = (K + TS - 1) / TS;

	//printf("num of tiles:, m:%d n:%d k:%d (M:%d N:%d: K:%d)\n", mtiles, ntiles, ktiles, M, N, K);

	/* a, b matrix copy */
	//memcpy(knpu->in_a1, A, TS*TS*mtiles*ktiles);
	//memcpy(knpu->in_b1, B, TS*TS*ntiles*ktiles);
	

#ifdef ERRATA1
	/* errata, k tiles must be over 2 */
	if(ktiles < 3)
		ktiles = 3;
#endif

	bufa = malloc(TS*ktiles);
	if(bufa == NULL) FATAL;

	bufb = malloc(TS*ntiles);
	if(bufb == NULL) FATAL;

	/* 8 byte allignmend is need to access global buffer */
	for(i=0; i < TS * mtiles; i++) {
		memset(bufa, 0 , TS*ktiles);
		if( i < M ) {
			memcpy(bufa, A + K*i, K);
			memcpy(knpu->in_a1 + TS*ktiles*i, bufa, TS*ktiles);
		}
		else
			memcpy(knpu->in_a1 + TS*ktiles*i, bufa, TS*ktiles);

	}


	for(i=0; i < TS * ktiles; i++) {
		memset(bufb, 0 , TS*ntiles);
		if( i < K ) {
			memcpy(bufb, B + N*i, N);
			memcpy(knpu->in_b1 + TS*ntiles*i, bufb, TS*ntiles);
		}
		else
			memcpy(knpu->in_b1 + TS*ntiles*i, bufb, TS*ntiles);
	}

	/* numeber of tiles config */
	npu_ctrl_write(GEMM_ACT_PARAM_0, mtiles - 1);
	npu_ctrl_write(GEMM_ACT_PARAM_1, ktiles - 1);
	npu_ctrl_write(GEMM_WEG_PARAM_0, ktiles - 1);
	npu_ctrl_write(GEMM_WEG_PARAM_1, ntiles - 1);

	/* run */
	npu_ctrl_write(GEMM_NPU_START, 0x1);

	/* wait for done */
	while (1) {
		status = npu_ctrl_read(GEMM_NPU_END);

		/* complete */
		if(status & 0x1)
			break;
		/* a2 and b2 slot is ready */
		// memcpy(knpu->in_a2, a, 6)
		// memcpy(knpu->in_a2, a, 6)
	}
	npu_ctrl_write(GEMM_NPU_START, 0x0);

	free(bufa);
	free(bufb);
	//printf("print output matrix\n");
	//print_int32_array(M, N, knpu->out_c1, 256);
}

void gemm_nn_sub_dma(int M, int N, int K, uint8_t * A, uint8_t * B, int32_t * C)
{
	int i;
	uint32_t status;
	int16_t mtiles, ntiles, ktiles;
	uint32_t pos;

	/* int division round up */
	mtiles = (M + TS - 1) / TS;
	ntiles = (N + TS - 1) / TS;
	ktiles = (K + TS - 1) / TS;

#ifdef ERRATA1
	/* errata, k tiles must be over 2 */
	if(ktiles < 3)
		ktiles = 3;
#endif

	/* 8 byte allignmend is need to access global buffer */
	for(i=0; i < TS * mtiles; i++) {
		if( i < M ) {
			memcpy(knpu->dma_buf + TS*ktiles*i, A + K*i, K);
			/* memset allow size 0 */
			memset(knpu->dma_buf + TS*ktiles*i + K, 0,
					TS*ktiles - K);
		}
		else
			memset(knpu->dma_buf + TS*ktiles*i, 0, TS*ktiles);

	}

	pos = NPU_POS_A1;
	ioctl(knpu->npu_fd, NPU_IOCTL_POS, &pos);
	write(knpu->npu_fd, knpu->dma_buf, mtiles*ktiles*TS*TS);

	for(i=0; i < TS * ktiles; i++) {
		if( i < K ) {
			memcpy(knpu->dma_buf + TS*ntiles*i, B + N*i, N);
			/* memset allow size 0 */
			memset(knpu->dma_buf + TS*ntiles*i + N, 0,
					TS*ntiles - N);
		}
		else
			memset(knpu->dma_buf + TS*ntiles*i, 0, TS*ntiles);
	}

	pos = NPU_POS_B1;
	ioctl(knpu->npu_fd, NPU_IOCTL_POS, &pos);
	write(knpu->npu_fd, knpu->dma_buf, ntiles*ktiles*TS*TS);

	/* numeber of tiles config */
	npu_ctrl_write(GEMM_ACT_PARAM_0, mtiles - 1);
	npu_ctrl_write(GEMM_ACT_PARAM_1, ktiles - 1);
	npu_ctrl_write(GEMM_WEG_PARAM_0, ktiles - 1);
	npu_ctrl_write(GEMM_WEG_PARAM_1, ntiles - 1);

	/* run */
	npu_ctrl_write(GEMM_NPU_START, 0x1);

	/* wait for done */
	while (1) {
		status = npu_ctrl_read(GEMM_NPU_END);

		/* complete */
		if(status & 0x1)
			break;
		/* a2 and b2 slot is ready */
		// memcpy(knpu->in_a2, a, 6)
		// memcpy(knpu->in_a2, a, 6)
	}
	npu_ctrl_write(GEMM_NPU_START, 0x0);

}


void gemm_nn_getC(int M, int N, int32_t * C)
{
	int i;
	int16_t ntiles;
	int32_t nc;
	uint8_t *bufc;

	/* int division round up */
	//mtiles = (M + TS - 1) / TS;
	ntiles = (N + TS - 1) / TS;

	bufc = malloc(TS*ntiles*4);
	if(bufc == NULL) FATAL;

	/* get output c */
	nc = TS * 4 * ntiles;
	for(i=0; i< M/4; i++) {
		memcpy(bufc, knpu->out_c1 + nc/4*i, nc);
		memcpy(C + N*4*i     , bufc, N*4);
		memcpy(bufc, knpu->out_c2 + nc/4*i, nc);
		memcpy(C + N*4*i + N, bufc, N*4);
		memcpy(bufc, knpu->out_c3 + nc/4*i, nc);
		memcpy(C + N*4*i + 2*N, bufc, N*4);
		memcpy(bufc, knpu->out_c4 + nc/4*i, nc);
		memcpy(C + N*4*i + 3*N, bufc, N*4);

		//memcpy(C + N*4*i     , knpu->out_c1 + nc/4*i, N*4);
		//memcpy(C + N*4*i + N, knpu->out_c2 + nc/4*i, N*4);
		//memcpy(C + N*4*i + 2*N, knpu->out_c3 + nc/4*i, N*4);
		//memcpy(C + N*4*i + 3*N, knpu->out_c4 + nc/4*i, N*4);
	}

	/*last 1-3 line of c result */
	if( M%4 == 3 || M%4 == 2 || M%4 == 1) {
		memcpy(bufc, knpu->out_c1 + nc/4*i, nc);
		memcpy(C + N*4*i     , bufc, N*4);
	}
	if( M%4 == 3 || M%4 == 2) {
		memcpy(bufc, knpu->out_c2 + nc/4*i, nc);
		memcpy(C + N*4*i + N, bufc, N*4);
	}
	if( M%4 == 3) {
		memcpy(bufc, knpu->out_c3 + nc/4*i, nc);
		memcpy(C + N*4*i + 2*N, bufc, N*4);
	}

	free(bufc);
}

void gemm_nn_getC_dma(int M, int N, int32_t * C)
{
	int i;
	int16_t ntiles, mtiles;
	int32_t nc;
	int32_t pos;

	/* int division round up */
	mtiles = (M + TS - 1) / TS;
	ntiles = (N + TS - 1) / TS;

	/* get c1, c2, c3, c4 */
	pos = NPU_POS_C1;
	ioctl(knpu->npu_fd, NPU_IOCTL_POS, &pos);
	read(knpu->npu_fd, knpu->dma_c1, mtiles*ntiles*TS*TS);
	pos = NPU_POS_C2;
	ioctl(knpu->npu_fd, NPU_IOCTL_POS, &pos);
	read(knpu->npu_fd, knpu->dma_c2, mtiles*ntiles*TS*TS);
	pos = NPU_POS_C3;
	ioctl(knpu->npu_fd, NPU_IOCTL_POS, &pos);
	read(knpu->npu_fd, knpu->dma_c3, mtiles*ntiles*TS*TS);
	pos = NPU_POS_C4;
	ioctl(knpu->npu_fd, NPU_IOCTL_POS, &pos);
	read(knpu->npu_fd, knpu->dma_c4, mtiles*ntiles*TS*TS);

	/* get output c */
	nc = TS * 4 * ntiles;
	for(i=0; i< M/4; i++) {
		memcpy(C + N*4*i     , knpu->dma_c1 + nc/4*i, N*4);
		memcpy(C + N*4*i + N,  knpu->dma_c2 + nc/4*i, N*4);
		memcpy(C + N*4*i + 2*N, knpu->dma_c3 + nc/4*i, N*4);
		memcpy(C + N*4*i + 3*N, knpu->dma_c4 + nc/4*i, N*4);
	}

	/*last 1-3 line of c result */
	if( M%4 == 3 || M%4 == 2 || M%4 == 1) {
		memcpy(C + N*4*i     , knpu->dma_c1 + nc/4*i, N*4);
	}
	if( M%4 == 3 || M%4 == 2) {
		memcpy(C + N*4*i + N,  knpu->dma_c2 + nc/4*i, N*4);
	}
	if( M%4 == 3) {
		memcpy(C + N*4*i + 2*N, knpu->dma_c3 + nc/4*i, N*4);
	}

}


void gemm_nn_npu(int M, int N, int K, uint8_t ALPHA,
		   uint8_t * A, int lda,
		   uint8_t * B, int ldb, int32_t * C, int ldc)
{
	int i,j;
	int16_t mblocks, nblocks, kblocks;
	int16_t mb, nb, kb;
	int16_t bsize_ax, bsize_ay, bsize_bx, bsize_by;
	uint8_t *suba, *subb;
	int32_t *subc;
	uint32_t ax_pos, bx_pos, cx_pos;
	uint16_t ax_size, ay_size=0, bx_size=0, by_size, cx_size, cy_size;

	pthread_mutex_lock(&knpu->mutex_lock);

	npu_printf("npu gemm input(M:%d N:%d K:%d)\n", M, N, K);

	/* calc block matric using K element */
	/* 1,2,3,4,8,16 */
	/* bsize_ay x bsize_bx < 128 & bszie_bx x bsize_by < 128 */
	if(K/(16*TS)>= 1){
		bsize_ax = 16;
		bsize_ay = 8;
		bsize_bx = 8;
	}
	else if( K/(8*TS) >= 1){
		bsize_ax = 8;
		bsize_ay = 8;
		bsize_bx = 16;
	}
	else if( K/(4*TS) >= 1){
		bsize_ax = 4;
		bsize_ay = 4;
		bsize_bx = 32;
	}
	else if( K/(3*TS) >= 1){
		bsize_ax = 3;
		bsize_ay = 3;
		bsize_bx = 42;
	}
	else if( K/(2*TS) >= 1) {
		bsize_ax = 2;
		bsize_ay = 2;
		bsize_bx = 64;
	}
	else {
		bsize_ax = 1;
		bsize_ay = 1;
		bsize_bx = 128;
	}
#ifdef ERRATA1
	if(bsize_ax < 3) {
		bsize_ax = 3;
		bsize_ay = 3;
		bsize_bx = 42;
	}
#endif
	bsize_by = bsize_ax;

	/* int division round up */
	mblocks = (M + TS*bsize_ay - 1) / (TS*bsize_ay);
	nblocks = (N + TS*bsize_bx - 1) / (TS*bsize_bx);
	kblocks = (K + TS*bsize_ax - 1) / (TS*bsize_ax);

	npu_printf("large block of tiles count:(m:%d n:%d k:%d) size: (%d,%d,%d)\n", mblocks, nblocks, kblocks, bsize_ay, bsize_bx, bsize_ax);
	/* print original input(a,b) source */
	//print_uint8_array(M, K, A, 256);
	//print_uint8_array(K, N, B, 256);

	/* block_xy of 32x32 matrics */
	suba = malloc(bsize_ax*bsize_ay*TS*TS);
	if(suba == NULL) FATAL;

	subb = malloc(bsize_bx*bsize_by*TS*TS);
	if(subb == NULL) FATAL;

	subc = malloc(bsize_ay*bsize_bx*TS*TS*4);
	if(subc == NULL) FATAL;


	for(mb=0; mb < mblocks; mb++) {
	for(nb=0; nb < nblocks; nb++) {
		/* psum off at start */
		npu_ctrl_write(GEMM_PSUM_PARAM_0, 0x0);

	for(kb=0; kb < kblocks; kb++) {
		/* info sub a array */
		ax_pos = TS*bsize_ay*K*mb + TS*bsize_ax*kb;
		ax_size= (kb+1)*TS*bsize_ax <= K ? (TS*bsize_ax):(K- kb*TS*bsize_ax);
		ay_size= (mb+1)*TS*bsize_ay <= M ? (TS*bsize_ay):(M- mb*TS*bsize_ay);
		//printf("ax_pos:%d ax_size=%d ay_size=%d\n", ax_pos, ax_size, ay_size);
		/* fill in sub a block */
		for(i=0; i< ay_size; i++)
			memcpy(suba + ax_size*i, A + ax_pos + K*i, ax_size);

		/* info sub b array */
		bx_pos = TS*bsize_by*N*kb + TS*bsize_bx*nb;
		bx_size= (nb+1)*TS*bsize_bx <= N ? (TS*bsize_bx):(N- nb*TS*bsize_bx);
		by_size= (kb+1)*TS*bsize_by <= K ? (TS*bsize_by):(K- kb*TS*bsize_by);
		//printf("bx_pos:%d bx_size=%d by_size=%d\n", bx_pos, bx_size, by_size);
		/* fill in sub b block */
		for(i=0; i< by_size; i++)
			memcpy(subb + bx_size*i, B + bx_pos + N*i, bx_size);

		/* assert ax_size == by_size */
		if(ax_size != by_size)
			printf("ax_size and by_size should be same(%d, %d)\n", ax_size, by_size);

		/* calcualte sub block */
#ifdef USE_DMA
		gemm_nn_sub_dma(ay_size, bx_size, ax_size, suba, subb, subc);
#else
		gemm_nn_sub(ay_size, bx_size, ax_size, suba, subb, subc);
#endif
		/* psum on at same k blocks */
		npu_ctrl_write(GEMM_PSUM_PARAM_0, 0x1);
	}
		cx_pos = N*TS*bsize_ay*mb + TS*bsize_bx*nb;
		cx_size = bx_size;
		cy_size = ay_size;
		//printf("cx_pos:%d cx_size=%d cy_size=%d\n", cx_pos, bx_size, ay_size);
		/* get C result */
#ifdef USE_DMA
		gemm_nn_getC_dma(cy_size, cx_size, subc);
#else
		gemm_nn_getC(cy_size, cx_size, subc);
#endif

#if 0
		for(i=0; i< cy_size; i++)
			memcpy(C + cx_pos + N*i, subc + cx_size*i, cx_size * 4);
#endif
		for(i=0; i< cy_size; i++)
		for(j=0; j< cx_size; j++)
			*(C + cx_pos + N*i + j) += *(subc + cx_size*i +j);
	}
	}

	free(suba);
	free(subb);
	free(subc);

	pthread_mutex_unlock(&knpu->mutex_lock);
}

