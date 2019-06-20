#pragma once

#include "Vec4.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

struct PointLight
{
	// Data
	
	Vec4 position;
	Colorf color;


	// Functions
	CUDA_CALLABLE_MEMBER PointLight() {}

	CUDA_CALLABLE_MEMBER PointLight(Vec4 pos, Colorf inten)
	{
		position = pos;
		color = inten;
	}

};