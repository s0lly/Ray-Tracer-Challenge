
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "Vec4.h"

#define THREADS_PER_BLOCK 256

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void addVec4Kernel(Vec4 *c4, Vec4 *a4, Vec4 *b4)
{
	int i = threadIdx.x;
	c4[i] = a4[i] + b4[i];
	Equiv(c4[i].x, b4[i].y);

}

int mainCUDA(Vec4 vec4Pass)
{
	int devicesCount;
	cudaGetDeviceCount(&devicesCount);
	for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
	{
		cudaDeviceProp deviceProperties;
		cudaGetDeviceProperties(&deviceProperties, deviceIndex);
		char* check = deviceProperties.name;

		check = check;

		int test = 1;
		test++;
	}

	Vec4 vec4 = vec4Pass;
	Vec4 vec4New = Vec4{ 1.0f, 5.0f, 1.0f, 1.0f };

	printf("{%f,%f,%f,%f}\n",
		vec4.x, vec4.y, vec4New.x, vec4New.y);

	const int size = 1000;

	Vec4 vec41[size];
	Vec4 vec42[size];
	Vec4 vec43[size];

	for (int i = 0; i < size; i++)
	{
		vec41[i].x = 1.0f;
		vec41[i].y = 1.0f;

		vec42[i].x = (float)i;
		vec42[i].y = (float)i * 2.0f;
	}

	printf("{%f,%f,%f,%f,%f}\n",
		vec41[0].x, vec41[1].y, vec41[2].x, vec41[3].y, vec41[4].x);

	Vec4 *vec41c = 0;
	Vec4 *vec42c = 0;
	Vec4 *vec43c = 0;

	cudaSetDevice(0);

	cudaMalloc((void**)&vec41c, size * sizeof(Vec4));
	cudaMalloc((void**)&vec42c, size * sizeof(Vec4));
	cudaMalloc((void**)&vec43c, size * sizeof(Vec4));

	cudaMemcpy(vec41c, vec41, size * sizeof(Vec4), cudaMemcpyHostToDevice);
	cudaMemcpy(vec42c, vec42, size * sizeof(Vec4), cudaMemcpyHostToDevice);

	int blockSize = THREADS_PER_BLOCK;
	int numBlocks = (size + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK;

	addVec4Kernel <<< numBlocks, blockSize >>> (vec43c, vec41c, vec42c);
	
	cudaDeviceSynchronize();

	cudaMemcpy(vec43, vec43c, size * sizeof(Vec4), cudaMemcpyDeviceToHost);

	cudaFree(vec41c);
	cudaFree(vec42c);
	cudaFree(vec43c);

	printf("{%f,%f,%f,%f,%f}\n",
		vec43[0].x, vec43[1].y, vec43[2].x, vec43[3].y, vec43[4].x);

    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
