#pragma once

#include "Vec4.h"




struct Intersection
{
	// Data
	float t;
	int objID;


	// Functions
	Intersection()
	{
		t = 0.0f;
		objID = -1.0f;
	}

	Intersection(float in_t, int in_objID)
	{
		t = in_t;
		objID = in_objID;
	}
};


struct IntersectionList
{
	// Data
	Intersection *list;
	int currentNum;


	// Functions
	IntersectionList()
	{
		// TODO: make list size modifiable?
		list = new Intersection[100];
		currentNum = 0;
	}

	~IntersectionList()
	{
		delete[] list;
	}


	void AddIntersection(float in_t, int in_objID)
	{
		list[currentNum] = Intersection(in_t, in_objID);
		currentNum++;
	}

	bool FindAndGetHit(Intersection &intersection)
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

	Ray() {}

	Ray(Vec4 orig, Vec4 dir)
	{
		origin = orig;
		direction = dir;
	}

	Vec4 Position(float t)
	{
		return origin + direction * t;
	}

	void Transform(Matrix4 transformationMat)
	{
		origin = transformationMat.MMult(origin);
		direction = transformationMat.MMult(direction);
	}

};
