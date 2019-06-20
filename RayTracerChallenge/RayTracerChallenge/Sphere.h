#pragma once

#include "Vec4.h"
#include "Matrix4.h"
#include "Ray.h"
#include "Material.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif


struct Sphere
{
	// Data
	Vec4 origin;
	float radius;
	int id;
	Matrix4 tranformationToWorldSpace;
	Material material;


	// Functions
	CUDA_CALLABLE_MEMBER Sphere(int in_id)
	{
		origin = Vec4::Point(0.0f, 0.0f, 0.0f);
		radius = 1.0f;
		id = in_id;
		tranformationToWorldSpace = Matrix4::Identity();
	}

	CUDA_CALLABLE_MEMBER Sphere(Vec4 in_origin, float in_radius, int in_id)
	{
		origin = in_origin;
		radius = in_radius;
		id = in_id;
		tranformationToWorldSpace = Matrix4::Identity();
	}

	CUDA_CALLABLE_MEMBER void Intersect(Ray &ray)
	{
		if (tranformationToWorldSpace.IsInvertible())
		{
			Ray rayTransformed;
			rayTransformed.origin = ray.origin;
			rayTransformed.direction = ray.direction;
			Matrix4 placeholder = tranformationToWorldSpace.Inverse();
			rayTransformed.Transform(placeholder);
			

			Vec4 sphereToRay = rayTransformed.origin - origin;
			float a = rayTransformed.direction.Dot(rayTransformed.direction);
			float b = 2.0f * rayTransformed.direction.Dot(sphereToRay);
			float c = sphereToRay.Dot(sphereToRay) - radius * radius;

			float discriminant = b * b - 4.0f * a * c;

			if (discriminant >= 0.0f)
			{
				float t = (-b - sqrt(discriminant)) / (2.0f * a);
				ray.intersections.AddIntersection(t, id);
				t = (-b + sqrt(discriminant)) / (2.0f * a);
				ray.intersections.AddIntersection(t, id);
			}
		}
	}

	CUDA_CALLABLE_MEMBER void AddTranformation(Matrix4 &rhs)
	{
		tranformationToWorldSpace = rhs.MMult(tranformationToWorldSpace);
	}

	CUDA_CALLABLE_MEMBER Vec4 GetNormal(Vec4 p)
	{
		Matrix4 inverseTransformToWS = tranformationToWorldSpace.Inverse();
		Vec4 objectSpacePoint = inverseTransformToWS.MMult(p);
		Vec4 objectSpaceNormal = objectSpacePoint - origin;
		Vec4 worldSpaceNormal = inverseTransformToWS.Transpose().MMult(objectSpaceNormal);
		worldSpaceNormal.w = 0.0f;
		return worldSpaceNormal.Normalize();
	}

};