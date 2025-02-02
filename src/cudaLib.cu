
#include "cudaLib.cuh"
#include "cpuLib.h"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int ii = blockIdx.x * blockDim.x + threadIdx.x;
	if(ii<size)
	y[ii] = scale * x[ii] + y[ii];

}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	float * c_a, * c_b, *c_c;

	c_a = (float *) malloc(vectorSize * sizeof(float));
	c_b = (float *) malloc(vectorSize * sizeof(float));
	c_c = (float *) malloc(vectorSize * sizeof(float));

	if (c_a == NULL || c_b == NULL) {
		printf("Unable to malloc memory ... Exiting!");
		return -1;
	}

	vectorInit(c_a, vectorSize);
	vectorInit(c_b, vectorSize);


	float * g_a, * g_b;
	float g_s = 1.5f;
	cudaMalloc(&g_a, vectorSize * sizeof(float));
	cudaMalloc(&g_b, vectorSize * sizeof(float));


	cudaMemcpy(g_a, c_a, vectorSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(g_b, c_b, vectorSize * sizeof(float), cudaMemcpyHostToDevice);

	int threadsperblock = 256;
	int n_blocks = (vectorSize+threadsperblock-1)/threadsperblock;

	saxpy_gpu<<<n_blocks, threadsperblock>>>(g_a, g_b, g_s, vectorSize);

	cudaMemcpy(c_c, g_b, vectorSize * sizeof(float), cudaMemcpyDeviceToHost);

    //for(int j=0; j<10; j++)
    //printf("Sample results: a = %f, b = %f, c = %f\n", c_a[j], c_b[j], c_c[j]);

    // Free memory
    free(c_a);
    free(c_b);
	free(c_c);
    cudaFree(g_a);
    cudaFree(g_b);

	std::cout << "Lazy, you are!\n";
	std::cout << "Write code, you must\n";

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	curandState_t rng_g;
	curand_init(clock64(), ii, 0, &rng_g);

	int count_th = 0;

	for (int jj = 0; jj < sampleSize; jj++){
		float x = curand_uniform(&rng_g);
		float y = curand_uniform(&rng_g);

		if (x*x + y*y <= 1.0f)
		count_th++;
	}

	pSums[ii] = count_th;

}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	int count_r = 0;
	for (int jj=0; jj<reduceSize; jj++){
		int kk = id * reduceSize + jj;
		count_r += pSums[kk];
	}

	totals[id] = count_r;
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//      Insert code here
	uint64_t * c_gen, *c_red;

	c_gen = (uint64_t *) malloc(generateThreadCount * sizeof(uint64_t));
	c_red = (uint64_t *) malloc(reduceThreadCount * sizeof(uint64_t));

	uint64_t * g_gen, * g_red;
	cudaMalloc(&g_gen, generateThreadCount * sizeof(uint64_t));
	cudaMalloc(&g_red, reduceThreadCount * sizeof(uint64_t));

	int n_threads_block = 64;   //assuming
	int n_blocks = (generateThreadCount+n_threads_block-1)/n_threads_block;

	generatePoints<<<n_blocks,n_threads_block>>> (g_gen, generateThreadCount, sampleSize);

	cudaDeviceSynchronize();

	n_blocks = (reduceThreadCount+n_threads_block-1)/n_threads_block;
	reduceCounts<<<n_blocks,n_threads_block>>> (g_gen, g_red, generateThreadCount, reduceSize);

	cudaMemcpy(c_gen, g_gen, generateThreadCount * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(c_red, g_red, reduceThreadCount * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Sum all counts on the CPU
    /*int total = 0;
    for (int k = 0; k < generateThreadCount; k++) {
        total += c_gen[k];
    }

    // Compute final estimate of π
    approxPi = 4.0f * total / (sampleSize*generateThreadCount);
    printf("Estimated Pi = %f\n", approxPi);*/  //only generation

	int total = 0;
    for (int k = 0; k < reduceThreadCount; k++) {
        total += c_red[k];
    }

    // Compute final estimate of π
    approxPi = 4.0f * total / (generateThreadCount * sampleSize);
    printf("Estimated Pi = %f\n", approxPi); //pi calculation with reduction 

    // Free memory
    free(c_gen);
    cudaFree(g_gen);
	cudaFree(c_red);
	cudaFree(g_red);
	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";
	return approxPi;
}
