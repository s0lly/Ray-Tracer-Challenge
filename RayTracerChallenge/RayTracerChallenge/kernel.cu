
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "Canvas.h"
#include "Sphere.h"
#include "Material.h"
#include "PointLight.h"
#include "Ray.h"
#include "Matrix4.h"
#include "Vec4.h"
#include "Camera.h"
#include "World.h"
#include "Colors.h"


#define THREADS_PER_BLOCK 256


__global__ void DrawScene(unsigned int *colorsCuda, World *world, Camera *camera, int width, int height) //int *checkerCuda, 
{
	int val = threadIdx.x + blockIdx.x * blockDim.x;
	int i = val % width;
	int j = height - 1 - val / width;


	Ray ray = camera->RayAtPixel(i, j);
	
	Colorf color = world->ColorAt(ray);

	int r = (int)(color.r * 255.999f);
	int g = (int)(color.g * 255.999f);
	int b = (int)(color.b * 255.999f);

	r = r > 255 ? 255 : r;
	g = g > 255 ? 255 : g;
	b = b > 255 ? 255 : b;

	unsigned int dword;

	dword = (((unsigned char)r << 16u) | ((unsigned char)g << 8u) | (unsigned char)b);

	colorsCuda[val] = dword;
}

int mainCUDA(unsigned int *colors, Camera &camera)
{

	World world;


	//int *checkerCuda;
	unsigned int *colorsCuda;
	World *worldCuda;
	Camera *cameraCuda;

	//cudaMalloc((void**)&checkerCuda, camera.width * camera.height * sizeof(int));
	cudaMalloc((void**)&colorsCuda, camera.width * camera.height * sizeof(unsigned int));
	cudaMalloc((void**)&worldCuda, sizeof(World));
	cudaMalloc((void**)&cameraCuda, sizeof(Camera));

	//cudaMemcpy(checkerCuda, checker, camera.width * camera.height * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(colorsCuda, colors, camera.width * camera.height * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(worldCuda, &world, sizeof(World), cudaMemcpyHostToDevice);
	cudaMemcpy(cameraCuda, &camera, sizeof(Camera), cudaMemcpyHostToDevice);


	int blockSize = THREADS_PER_BLOCK;
	int numBlocks = (camera.width * camera.height + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK;

	DrawScene <<< numBlocks, blockSize >>> (colorsCuda, worldCuda, cameraCuda, camera.width, camera.height); //checkerCuda, 
	cudaDeviceSynchronize();

	//cudaMemcpy(checker, checkerCuda, camera.width * camera.height * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(colors, colorsCuda, camera.width * camera.height * sizeof(unsigned int), cudaMemcpyDeviceToHost);


	//cudaFree(checkerCuda);
	cudaFree(colorsCuda);
	cudaFree(worldCuda);
	cudaFree(cameraCuda);


	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}


    return 0;
}