
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

__global__ void DrawScene(int *checkerCuda, Colorf *colorsCuda, Sphere *s, PointLight *light, int width, int height)
{
	int val = threadIdx.x + blockIdx.x * blockDim.x;
	int i = val % width;
	int j = val / width;

	checkerCuda[val] = val;



	float viewBoardHalfWidth = 5.0f * (float)width / (float)height;
	float viewBoardHalfHeight = 5.0f;
	Vec4 viewBoardOrigin = Vec4::Point(0.0f, 0.0f, 10.0f);

	Vec4 rayOrigin = Vec4::Point(0.0f, 0.0f, -5.0f);


	Vec4 viewBoardLoc = viewBoardOrigin + Vec4::Point((float)(((float)i - (float)(width / 2)) / (float)(width / 2)) * viewBoardHalfWidth, -(float)(((float)j - (float)(height / 2)) / (float)(height / 2)) * viewBoardHalfHeight, 0.0f);

	Vec4 directionFromRayToViewBoard = viewBoardLoc - rayOrigin;

	Ray ray(rayOrigin, directionFromRayToViewBoard.Normalize());

	s->Intersect(ray);

	Intersection hit;

	if (ray.intersections.FindAndGetHit(hit))
	{
		Vec4 positionOnSphere = ray.Position(hit.t);
		Colorf color = s->material.Lighting(*light, ray.Position(hit.t), ray.origin - positionOnSphere, s->GetNormal(positionOnSphere), false);
		//c->SetPixel(i, j, color);
		colorsCuda[val] = color;
	}
	else
	{
		//c->SetPixel(i, j, Colorf{ 0.0f, 0.0f, 0.0f });
		colorsCuda[val] = Colorf{ 0.0f, 0.0f, 0.0f };
	}
}

int mainCUDA()
{



	//float viewBoardHalfWidth = 5.0f * (float)c.width / (float)c.height;
	//float viewBoardHalfHeight = 5.0f;
	//Vec4 viewBoardOrigin = Vec4::Point(0.0f, 0.0f, 10.0f);
	//
	//Vec4 rayOrigin = Vec4::Point(0.0f, 0.0f, -5.0f);
	
	
	
	
	
	//int countHits = 0;
	//
	//for (int i = 0; i < c.width; i++)
	//{
	//	for (int j = 0; j < c.height; j++)
	//	{
	//		Vec4 viewBoardLoc = viewBoardOrigin + Vec4::Point((float)(((float)i - (float)(c.width / 2)) / (float)(c.width / 2)) * viewBoardHalfWidth, -(float)(((float)j - (float)(c.height / 2)) / (float)(c.height / 2)) * viewBoardHalfHeight, 0.0f);
	//
	//		Vec4 directionFromRayToViewBoard = viewBoardLoc - rayOrigin;
	//
	//		Ray ray(rayOrigin, directionFromRayToViewBoard.Normalize());
	//
	//		s.Intersect(ray);
	//
	//		Intersection hit;
	//
	//		if (ray.intersections.FindAndGetHit(hit))
	//		{
	//			Vec4 positionOnSphere = ray.Position(hit.t);
	//			Colorf color = s.material.Lighting(light, ray.Position(hit.t), ray.origin - positionOnSphere, s.GetNormal(positionOnSphere));
	//			c.SetPixel(i, j, color);
	//			countHits++;
	//		}
	//		else
	//		{
	//			c.SetPixel(i, j, Colorf{ 0.0f, 0.0f, 0.0f });
	//		}
	//	}
	//}

	cudaSetDevice(0);


	
	Canvas c(1600, 900);

	Sphere s(0);
	s.material.color = Colorf{ 1.0f, 0.2f, 1.0f };
	s.AddTranformation(Matrix4::Scale(1.0f, 1.0f, 1.0f));
	s.AddTranformation(Matrix4::RotationY(PI / 2.0f));

	PointLight light(Vec4::Point(-10.0f, 10.0f, -10.0), Colorf{ 1.0f, 1.0f, 1.0f });


	int *checker = new int[c.width * c.height];
	for (int i = 0; i < c.width * c.height; i++)
	{
		checker[i] = 0;
	}

	Colorf *colors = new Colorf[c.width * c.height];
	for (int i = 0; i < c.width * c.height; i++)
	{
		colors[i] = Colorf{ 0.0f, 0.0f, 0.0f };
	}


	int *checkerCuda;
	Colorf *colorsCuda;
	Sphere *sCuda;
	PointLight *lightCuda;

	cudaMalloc((void**)&checkerCuda, c.width * c.height * sizeof(int));
	cudaMalloc((void**)&colorsCuda, c.width * c.height * sizeof(Colorf));
	cudaMalloc((void**)&sCuda, sizeof(Sphere));
	cudaMalloc((void**)&lightCuda, sizeof(PointLight));

	cudaMemcpy(checkerCuda, checker, c.width * c.height * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(colorsCuda, colors, c.width * c.height * sizeof(Colorf), cudaMemcpyHostToDevice);
	cudaMemcpy(sCuda, &s, sizeof(Sphere), cudaMemcpyHostToDevice);
	cudaMemcpy(lightCuda, &light, sizeof(PointLight), cudaMemcpyHostToDevice);


	int blockSize = THREADS_PER_BLOCK;
	int numBlocks = (c.width * c.height + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK;

	DrawScene <<< numBlocks, blockSize >>> (checkerCuda, colorsCuda, sCuda, lightCuda, c.width, c.height);
	cudaDeviceSynchronize();

	cudaMemcpy(checker, checkerCuda, c.width * c.height * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(colors, colorsCuda, c.width * c.height * sizeof(Colorf), cudaMemcpyDeviceToHost);


	cudaFree(checkerCuda);

	cudaDeviceReset();

	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	memcpy(c.pixels, colors, c.width * c.height * sizeof(Colorf));



	c.CreatePPM("chapter6.ppm");












	//Vec4 vec4Pass{ 1.0f, 1.0f, 1.0f, 1.0f };
	//
	//int devicesCount;
	//cudaGetDeviceCount(&devicesCount);
	//for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
	//{
	//	cudaDeviceProp deviceProperties;
	//	cudaGetDeviceProperties(&deviceProperties, deviceIndex);
	//	char* check = deviceProperties.name;
	//
	//	check = check;
	//
	//	int test = 1;
	//	test++;
	//}
	//
	//Vec4 vec4 = vec4Pass;
	//Vec4 vec4New = Vec4{ 1.0f, 5.0f, 1.0f, 1.0f };
	//
	//printf("{%f,%f,%f,%f}\n",
	//	vec4.x, vec4.y, vec4New.x, vec4New.y);
	//
	//const int size = 1000;
	//
	//Vec4 vec41[size];
	//Vec4 vec42[size];
	//Vec4 vec43[size];
	//
	//for (int i = 0; i < size; i++)
	//{
	//	vec41[i].x = 1.0f;
	//	vec41[i].y = 1.0f;
	//
	//	vec42[i].x = (float)i;
	//	vec42[i].y = (float)i * 2.0f;
	//}
	//
	//printf("{%f,%f,%f,%f,%f}\n",
	//	vec41[0].x, vec41[1].y, vec41[2].x, vec41[3].y, vec41[4].x);
	//
	//Vec4 *vec41c = 0;
	//Vec4 *vec42c = 0;
	//Vec4 *vec43c = 0;
	//
	//cudaSetDevice(0);
	//
	//cudaMalloc((void**)&vec41c, size * sizeof(Vec4));
	//cudaMalloc((void**)&vec42c, size * sizeof(Vec4));
	//cudaMalloc((void**)&vec43c, size * sizeof(Vec4));
	//
	//cudaMemcpy(vec41c, vec41, size * sizeof(Vec4), cudaMemcpyHostToDevice);
	//cudaMemcpy(vec42c, vec42, size * sizeof(Vec4), cudaMemcpyHostToDevice);
	//
	//int blockSize = THREADS_PER_BLOCK;
	//int numBlocks = (size + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK;
	//
	//addVec4Kernel <<< numBlocks, blockSize >>> (vec43c, vec41c, vec42c);
	//
	//cudaDeviceSynchronize();
	//
	//cudaMemcpy(vec43, vec43c, size * sizeof(Vec4), cudaMemcpyDeviceToHost);
	//
	//cudaFree(vec41c);
	//cudaFree(vec42c);
	//cudaFree(vec43c);
	//
	//printf("{%f,%f,%f,%f,%f}\n",
	//	vec43[0].x, vec43[1].y, vec43[2].x, vec43[3].y, vec43[4].x);
	//
    //const int arraySize = 5;
    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    //int c[arraySize] = { 0 };
	//
    //// Add vectors in parallel.
    //cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return 1;
    //}
	//
    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    //    c[0], c[1], c[2], c[3], c[4]);
	//
    //// cudaDeviceReset must be called before exiting in order for profiling and
    //// tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceReset failed!");
    //    return 1;
    //}
	//
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
