#pragma once

#include "Vec4.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif



struct Intersection
{
	// Data
	float t;
	int objID;


	// Functions
	CUDA_CALLABLE_MEMBER Intersection()
	{
		t = 0.0f;
		objID = -1.0f;
	}

	CUDA_CALLABLE_MEMBER Intersection(float in_t, int in_objID)
	{
		t = in_t;
		objID = in_objID;
	}
};


struct IntersectionList
{
	// Data
	Intersection list[100];
	int currentNum;


	// Functions
	CUDA_CALLABLE_MEMBER IntersectionList()
	{
		// TODO: make list size modifiable?
		//list = new Intersection[100];
		currentNum = 0;
	}

	//~IntersectionList()
	//{
	//	delete[] list;
	//}


	CUDA_CALLABLE_MEMBER void AddIntersection(float in_t, int in_objID)
	{
		list[currentNum] = Intersection(in_t, in_objID);
		currentNum++;
	}

	CUDA_CALLABLE_MEMBER bool FindAndGetHit(Intersection &intersection)
	{
		bool foundHit = false;
		float smallestPosT = 999999999999999.0f;
		int foundInList = -1;

		for (int i = 0; i < currentNum; i++)
		{
			if (list[i].t < smallestPosT && list[i].t > 0.0f)
			{
				smallestPosT = list[i].t;
				foundHit = true;
				foundInList = i;
			}
		}

		if (foundHit)
		{
			intersection = list[foundInList];
		}

		return foundHit;
	}
};



struct Ray
{
	// Data
	Vec4 origin;
	Vec4 direction;
	IntersectionList intersections;

	// Functions

	CUDA_CALLABLE_MEMBER Ray() {}

	CUDA_CALLABLE_MEMBER Ray(Vec4 orig, Vec4 dir)
	{
		origin = orig;
		direction = dir;
	}

	CUDA_CALLABLE_MEMBER Vec4 Position(float t)
	{
		return origin + direction * t;
	}

	CUDA_CALLABLE_MEMBER void Transform(Matrix4 transformationMat)
	{
		origin = transformationMat.MMult(origin);
		direction = transformationMat.MMult(direction);
	}

};
