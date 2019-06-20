#pragma once

#include "Ray.h"

struct Sphere;


struct ComputeInfo
{
	// Data

	float t;
	Sphere *obj;
	Vec4 point;
	Vec4 eyeVec;
	Vec4 normalVec;
	bool isInside;



	// Functions

	void Prepare(Intersection &i, Ray &r)
	{
		t = i.t;
		obj = i.obj;
		point = r.Position(t);
		eyeVec = -r.direction;
		normalVec = obj->GetNormal(point);
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