#pragma once

#include "Ray.h"

struct Shape;


struct ComputeInfo
{
	// Data

	float t;
	Shape *obj;
	Vec4 point;
	Vec4 eyeVec;
	Vec4 normalVec;
	Vec4 reflectedRayDirection;
	bool isInside;



	// Functions

	CUDA_CALLABLE_MEMBER void Prepare(Intersection &i, Ray &r)
	{
		t = i.t;
		obj = i.obj;
		point = r.Position(t);
		eyeVec = -r.direction;
		normalVec = obj->GetNormal(point);
		reflectedRayDirection = r.direction.GetReflectionOff(normalVec);
		if (normalVec.Dot(eyeVec) < 0.0f)
		{
			normalVec = -normalVec;
			isInside = true;
		}
		else
		{
			isInside = false;
		}
	}

};