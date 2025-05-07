#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <stdint.h>

//#include <cpuid.h>

#include "perf_counter.h"
#include "npu_gemm.h"

enum gemm_type {
    gemm_type_unknown,
    gemm_type_cgemm,
    gemm_type_sgemm,
};

static int compare_ulonglong(const void *a_ptr, const void *b_ptr)
{
    const unsigned long long a = *((unsigned long long *)a_ptr);
    const unsigned long long b = *((unsigned long long *)b_ptr);
    if (a < b) {
	return -1;
    } else if (a > b) {
	return 1;
    } else {
	return 0;
    }
}

static inline unsigned long long average(unsigned long long a,
					 unsigned long long b)
{
    return (a / 2) + (b / 2) + (a & b & 1ull);
}

unsigned long long median(unsigned long long array[], size_t length)
{
    qsort(array, length, sizeof(unsigned long long), &compare_ulonglong);
    if (length % 2 == 0) {
	const unsigned long long median_lo = array[length / 2 - 1];
	const unsigned long long median_hi = array[length / 2];
	return average(median_lo, median_hi);
    } else {
	return array[length / 2];
    }
}

int read_memory(const void *pointer, size_t bytes)
{
    int hash = 0;
    while (bytes >= 64) {
	hash ^= *((const int *)pointer);
	pointer += 64;
	bytes -= 64;
    }
    return hash;
}

void gemm_nn(int M, int N, int K, float ALPHA,
	     float *A, int lda, float *B, int ldb, float *C, int ldc)
{
    int i, j, k;
    //printf("%d %d %d %f A %d B %d C %d\n", M,N,K,ALPHA,lda,ldb,ldc);
    for (i = 0; i < M; ++i) {
	for (k = 0; k < K; ++k) {
	    register float A_PART = ALPHA * A[i * lda + k];
	    for (j = 0; j < N; ++j) {
		C[i * ldc + j] += A_PART * B[k * ldb + j];
	    }
	}
    }
}

#define R_MULT (1)		// 4 - 32
int max_abs(int src, int max_val)
{
    if (abs(src) > abs(max_val))
	src = (src > 0) ? max_val : -max_val;
    return src;
}

void gemm_nn_int8_int32(int M, int N, int K, int8_t ALPHA,
			int8_t * A, int lda,
			int8_t * B, int ldb, int32_t * C, int ldc)
{
    int32_t *c_tmp = calloc(N, sizeof(int32_t));
    int i, j, k;
    for (i = 0; i < M; ++i) {
	for (k = 0; k < K; ++k) {
	    register int16_t A_PART = ALPHA * A[i * lda + k];
	    //#pragma simd parallel for
	    for (j = 0; j < N; ++j) {
		c_tmp[j] += A_PART * B[k * ldb + j];
		//C[i*ldc + j] += max_abs(A_PART*B[k*ldb + j] / (R_MULT), (256 * 128 - 1));
	    }
	}
	for (j = 0; j < N; ++j) {
	    /* do not consider overflow */
	    C[i * ldc + j] += c_tmp[j];
	    //C[i * ldc + j] += max_abs(c_tmp[j] / (R_MULT), (1024 * 1024 * 2 - 1));
	    c_tmp[j] = 0;
	}
    }
    free(c_tmp);
}

void gemm_nn_uint8_int32(int M, int N, int K, uint8_t ALPHA,
			uint8_t * A, int lda,
			uint8_t * B, int ldb, int32_t * C, int ldc)
{
    int32_t *c_tmp = calloc(N, sizeof(int32_t));
    int i, j, k;
    for (i = 0; i < M; ++i) {
	for (k = 0; k < K; ++k) {
	    register int16_t A_PART = ALPHA * A[i * lda + k];
	    //#pragma simd parallel for
	    for (j = 0; j < N; ++j) {
		c_tmp[j] += A_PART * B[k * ldb + j];
		//C[i*ldc + j] += max_abs(A_PART*B[k*ldb + j] / (R_MULT), (256 * 128 - 1));
	    }
	}
	for (j = 0; j < N; ++j) {
	    /* do not consider overflow */
	    C[i * ldc + j] += c_tmp[j];
	    //C[i * ldc + j] += max_abs(c_tmp[j] / (R_MULT), (1024 * 1024 * 2 - 1));
	    c_tmp[j] = 0;
	}
    }
    free(c_tmp);
}

void print_32m(int8_t * M)
{
    int i, j;
    for (i = 0; i < 4; i++) {
	for (j = 0; j < 4; j++) {
	    printf("%4d ", *(M + i * 32 + j));
	}
	printf("\n");
    }
    printf("\n");
}

#define TS 32

void gemm_nn_block(int M, int N, int K, int8_t ALPHA,
		   int8_t * A, int lda,
		   int8_t * B, int ldb, int32_t * C, int ldc)
{
    int8_t Asub[TS][TS];
    int8_t Bsub[TS][TS];
    int32_t Csub[TS][TS];
    int16_t mtiles, ntiles, ktiles;
    int ax_size, ay_size, bx_size, by_size, cx_size, cy_size;
    int mk, nk, tk, i, j;

    /* int division round up */
    //mtiles = M/TS;
    mtiles = (M + TS - 1) / TS;
    ntiles = (N + TS - 1) / TS;
    ktiles = (K + TS - 1) / TS;

    //printf("%d %d %d\n", mtiles, ntiles, ktiles);
    //print_32m(A);
    //print_32m(B);
    /* select one tile of A and B */
    for (mk = 0; mk < mtiles; mk++) {
	for (nk = 0; nk < ntiles; nk++) {
	    memset(Csub, 0, TS * TS * sizeof(int32_t));
	    for (tk = 0; tk < ktiles; tk++) {
		ax_size = ((tk + 1) * TS <= K) ? TS : K % TS;
		ay_size = ((mk + 1) * TS <= M) ? TS : M % TS;
		//printf("a:%d %d \n", ax_size, ay_size);
		for (i = 0; i < TS; i++) {
		    if (i < ay_size) {
			memcpy(&Asub[i][0],
			       (A + mk * TS * K + tk * TS + i * K), ax_size);
			if (ax_size != TS)
			    memset(&Asub[i]
				   [ax_size], 0, TS - ax_size);
		    } else {
			memset(&Asub[i][0], 0, TS);
		    }

		}
		//print_32m(Asub);
		bx_size = ((nk + 1) * TS <= N) ? TS : N % TS;
		by_size = ((tk + 1) * TS <= K) ? TS : K % TS;
		//printf("b:%d %d \n", bx_size, by_size);
		for (i = 0; i < TS; i++) {
		    if (i < by_size) {
			memcpy(&Bsub[i][0],
			       (B + tk * TS * N + nk * TS + i * N), bx_size);
			if (bx_size != TS)
			    memset(&Bsub[i]
				   [bx_size], 0, TS - bx_size);
		    } else {
			memset(&Bsub[i][0], 0, TS);
		    }

		}
		//print_32m(Bsub);

		gemm_nn_int8_int32(32, 32, 32, 1,
				   (int8_t *) Asub, 32,
				   (int8_t *) Bsub, 32, (int32_t *) Csub, 32);
		//printf("%d %d\n", Csub[0][0], Csub[0][1]);
	    }
	    cx_size = ((nk + 1) * TS <= N) ? TS : N % TS;
	    cy_size = ((mk + 1) * TS <= M) ? TS : M % TS;
	    //printf("c:%d %d \n", cx_size, cy_size);
#if 1
	    for (i = 0; i < cy_size; i++) {
		for (j = 0; j < cx_size; j++) {
		    *(int32_t *) (C + mk * TS * N +
				  nk * TS + i * N + j) += Csub[i][j];
		    //printf("%d %d\n", mk*TS*N + nk*TS + i*N + j, *(C+mk*TS*N + nk*TS + i*N + j));
		    //memcpy(C+mk*TS*N + nk*TS + i*N, &Csub[i][0], cx_size);
		}
	    }
#endif
	    //memcpy(C, Csub, 32*32*2);
	}			//end of nk loop
    }				// end of mk loop

}

unsigned long long benchmark_gemm(const void *memory, size_t cache_size,
				  enum gemm_type type,
				  size_t m, size_t n, size_t k,
#ifdef NPU_INT8
				  int8_t a[], int8_t b[], int32_t c[],
#else
				  uint8_t a[], uint8_t b[], int32_t c[],
#endif
				  size_t max_iterations, int hw)
{
    unsigned long long computation_time[max_iterations];
    size_t computation_samples = 0;
    for (size_t iteration = 0; iteration < max_iterations; iteration++) {
	read_memory(memory, cache_size);

	unsigned long long start_time, end_time;
	if (!read_timer(&start_time))
	    continue;

	switch (type) {
	case gemm_type_sgemm:
	    {
		if (hw == 0) {
#ifdef NPU_INT8
		    gemm_nn_int8_int32(m, n, k, 1, a, k, b, n, c, n);
#else
		    gemm_nn_uint8_int32(m, n, k, 1, a, k, b, n, c, n);
#endif
		} else if (hw == 1) {
		    //gemm_nn_int8_int32(m, n, k, 1, a, k, b, k, c, n);
		    //gemm_nn_block(m, n, k, 1, a, k, b, n, c, n);
		    gemm_nn_npu(m, n, k, 1, (uint8_t *)a, k, (uint8_t *)b, n, c, n);
		}

		break;
	    }
	case gemm_type_cgemm:
	    {
		break;
	    }
	case gemm_type_unknown:
	    {
		break;
	    }
	}

	if (!read_timer(&end_time))
	    continue;

	computation_time[computation_samples++] = end_time - start_time;
    }

    return median(computation_time, computation_samples);
}

struct options {
    enum gemm_type type;
    size_t m;
    size_t n;
    size_t k;
    size_t iterations;
};

static void print_options_help(const char *program_name)
{
    printf("%s parameters...\n"
	   "Required parameters:\n"
	   "  -g   --gemm               The type of GEMM operation (sgemm, cgemm)\n"
	   "  -m                        The M dimension\n"
	   "  -n                        The N dimension\n"
	   "  -k                        The K dimension\n"
	   "Optional parameters:\n"
	   "  -i   --iterations         # iterations (default: 151)\n",
	   program_name);
}

static struct options parse_options(int argc, char **argv)
{
    struct options options = {
	.type = gemm_type_unknown,
	.k = 0,
	.m = 0,
	.n = 0,
	.iterations = 151
    };
    for (int argi = 1; argi < argc; argi += 1) {
	if ((strcmp(argv[argi], "-g") == 0)
	    || (strcmp(argv[argi], "--gemm") == 0)) {
	    if (argi + 1 == argc) {
		fprintf(stderr, "Error: expected k value\n");
		exit(EXIT_FAILURE);
	    }
	    if (strcmp(argv[argi + 1], "sgemm") == 0) {
		options.type = gemm_type_sgemm;
	    } else if (strcmp(argv[argi + 1], "cgemm") == 0) {
		options.type = gemm_type_cgemm;
	    } else {
		fprintf(stderr,
			"Error: invalid value %s for the gemm type: expected \"sgemm\", \"cgemm\"\n",
			argv[argi + 1]);
		exit(EXIT_FAILURE);
	    }
	    argi += 1;
	} else if (strcmp(argv[argi], "-k") == 0) {
	    if (argi + 1 == argc) {
		fprintf(stderr, "Error: expected K value\n");
		exit(EXIT_FAILURE);
	    }
	    if (sscanf(argv[argi + 1], "%zu", &options.k) != 1) {
		fprintf(stderr,
			"Error: can not parse %s as an unsigned integer\n",
			argv[argi + 1]);
		exit(EXIT_FAILURE);
	    }
	    if (options.k == 0) {
		fprintf(stderr,
			"Error: invalid value %s for the K parameter: positive value expected\n",
			argv[argi + 1]);
		exit(EXIT_FAILURE);
	    }
	    argi += 1;
	} else if (strcmp(argv[argi], "-m") == 0) {
	    if (argi + 1 == argc) {
		fprintf(stderr, "Error: expected M value\n");
		exit(EXIT_FAILURE);
	    }
	    if (sscanf(argv[argi + 1], "%zu", &options.m) != 1) {
		fprintf(stderr,
			"Error: can not parse %s as an unsigned integer\n",
			argv[argi + 1]);
		exit(EXIT_FAILURE);
	    }
	    if (options.m == 0) {
		fprintf(stderr,
			"Error: invalid value %s for the M parameter: positive value expected\n",
			argv[argi + 1]);
		exit(EXIT_FAILURE);
	    }
	    argi += 1;
	} else if (strcmp(argv[argi], "-n") == 0) {
	    if (argi + 1 == argc) {
		fprintf(stderr, "Error: expected N value\n");
		exit(EXIT_FAILURE);
	    }
	    if (sscanf(argv[argi + 1], "%zu", &options.n) != 1) {
		fprintf(stderr,
			"Error: can not parse %s as an unsigned integer\n",
			argv[argi + 1]);
		exit(EXIT_FAILURE);
	    }
	    if (options.n == 0) {
		fprintf(stderr,
			"Error: invalid value %s for the N parameter: positive value expected\n",
			argv[argi + 1]);
		exit(EXIT_FAILURE);
	    }
	    argi += 1;
	} else if ((strcmp(argv[argi], "--iterations") == 0)
		   || (strcmp(argv[argi], "-i") == 0)) {
	    if (argi + 1 == argc) {
		fprintf(stderr, "Error: expected iterations value\n");
		exit(EXIT_FAILURE);
	    }
	    if (sscanf(argv[argi + 1], "%zu", &options.iterations)
		!= 1) {
		fprintf(stderr,
			"Error: can not parse %s as an unsigned integer\n",
			argv[argi + 1]);
		exit(EXIT_FAILURE);
	    }
	    if (options.iterations == 0) {
		fprintf(stderr,
			"Error: invalid value %s for the number of iterations: positive value expected\n",
			argv[argi + 1]);
		exit(EXIT_FAILURE);
	    }
	    argi += 1;
	} else if ((strcmp(argv[argi], "--help") == 0)
		   || (strcmp(argv[argi], "-h") == 0)) {
	    print_options_help(argv[0]);
	    exit(EXIT_SUCCESS);
	} else {
	    fprintf(stderr, "Error: unknown argument '%s'\n", argv[argi]);
	    print_options_help(argv[0]);
	    exit(EXIT_FAILURE);
	}
    }
    if (options.type == gemm_type_unknown) {
	fprintf(stderr, "Error: gemm kernel type is not specified\n");
	print_options_help(argv[0]);
	exit(EXIT_FAILURE);
    }
    if (options.m == 0) {
	fprintf(stderr, "Error: M is not specified\n");
	print_options_help(argv[0]);
	exit(EXIT_FAILURE);
    }
    if (options.n == 0) {
	fprintf(stderr, "Error: N is not specified\n");
	print_options_help(argv[0]);
	exit(EXIT_FAILURE);
    }
    if (options.k == 0) {
	fprintf(stderr, "Error: K is not specified\n");
	print_options_help(argv[0]);
	exit(EXIT_FAILURE);
    }
    return options;
}

void compare_result(const int32_t a[], const int32_t b[], int num)
{
    int i;

    for (i = 0; i < num; i++) {
	if (a[i] != b[i]) {
		printf("index:%d is diff:%d %d\n", i, a[i], b[i]);
		return;
	}

    }
    npu_printf("compare result is OK\n");
}

int main(int argc, char **argv)
{
    const struct options options = parse_options(argc, argv);
    size_t components = 0;

    int32_t *c1, *c2;
#ifdef NPU_INT8
    int8_t *a, *b, *pa, *pb;
#else
    uint8_t *a, *b, *pa, *pb;
#endif

    switch (options.type) {
    case gemm_type_sgemm:
	components = 1;
	break;
    case gemm_type_cgemm:
	components = 2;
	break;
    default:
	__builtin_unreachable();
    }

    const size_t cache_size = 128 * 1024 * 1024;
    void *memory = valloc(cache_size);

    c1 = (int32_t *) valloc(options.m * options.n * components *
			   sizeof(int32_t));
    c2 = (int32_t *) valloc(options.m * options.n * components *
			   sizeof(int32_t));
    a = valloc(options.m * options.k * components * sizeof(uint8_t));
    b = valloc(options.k * options.n * components * sizeof(uint8_t));

    pa = a;
    pb = b;
    int i;
    srandom(time(NULL));
    for (i = 0; i < options.m * options.k * components; i++) {
	*pa = random() % 256;
	pa++;
    }
    for (i = 0; i < options.k * options.n * components; i++) {
	*pb = random() % 256;
	pb++;
    }

    memset(c1, 0, options.m * options.n * components * sizeof(int32_t));
    memset(c2, 0, options.m * options.n * components * sizeof(int32_t));
    //memcpy((uint8_t*)c2, a, options.m*options.k);
    //memset(c2, 0, options.m * options.n * components * sizeof(int32_t));
    //memset(a, 0, options.m * options.k * components * sizeof(float));
    //memset(b, 0, options.k * options.n * components * sizeof(float));
    //

    init_kgemm();

    {
	npu_printf("Iterations: %zu\n", options.iterations);

	unsigned long long gemm_nanoseconds = benchmark_gemm(memory, cache_size,
							     options.type,
							     options.m,
							     options.n,
							     options.k,
							     a, b, c1,
							     options.iterations,
							     0);
	double gemm_gflops =
	    2.0 * components * components * options.m * options.n * options.k /
	    ((double)gemm_nanoseconds);

	npu_printf("Time: %5.3lf ms\n", gemm_nanoseconds * 1.0e-6);
	//printf("%-10lld", gemm_nanoseconds/1000000);
	npu_printf("Performance: %5.3lf GFLOPs/s\n", gemm_gflops);

	gemm_nanoseconds = benchmark_gemm(memory, cache_size,
					  options.type,
					  options.m, options.n,
					  options.k, a, b, c2,
					  options.iterations, 1);
	gemm_gflops =
	    2.0 * components * components * options.m * options.n *
	    options.k / ((double)gemm_nanoseconds);

	npu_printf("Time: %5.3lf ms\n", gemm_nanoseconds * 1.0e-6);
	//printf(" %-10lld\n", gemm_nanoseconds/1000000);
	npu_printf("Performance: %5.3lf GFLOPs/s\n", gemm_gflops);

	compare_result(c1, c2, options.m * options.n * components);
    }
}
