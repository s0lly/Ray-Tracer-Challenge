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
	Matrix4 inverseTranformationToWorldSpace;
	Material material;


	// Functions
	CUDA_CALLABLE_MEMBER Sphere()
	{
		origin = Vec4::Point(0.0f, 0.0f, 0.0f);
		radius = 1.0f;
		id = -1;
		tranformationToWorldSpace = Matrix4::Identity();
		inverseTranformationToWorldSpace = Matrix4::Identity();
	}

	CUDA_CALLABLE_MEMBER Sphere(int in_id)
	{
		origin = Vec4::Point(0.0f, 0.0f, 0.0f);
		radius = 1.0f;
		id = in_id;
		tranformationToWorldSpace = Matrix4::Identity();
		inverseTranformationToWorldSpace = Matrix4::Identity();
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
				ray.intersections.AddIntersection(t, this);
				t = (-b + sqrt(discriminant)) / (2.0f * a);
				ray.intersections.AddIntersection(t, this);
			}
		}
	}

	CUDA_CALLABLE_MEMBER void AddTranformation(Matrix4 &rhs)
	{
		tranformationToWorldSpace = rhs.MMult(tranformationToWorldSpace);
		if (tranformationToWorldSpace.IsInvertible())
		{
			inverseTranformationToWorldSpace = tranformationToWorldSpace.Inverse();
		}
		
	}

	CUDA_CALLABLE_MEMBER Vec4 GetNormal(Vec4 p)
	{
		Vec4 objectSpacePoint = inverseTranformationToWorldSpace.MMult(p);
		Vec4 objectSpaceNormal = objectSpacePoint - origin;
		Vec4 worldSpaceNormal = inverseTranformationToWorldSpace.Transpose().MMult(objectSpaceNormal);
		worldSpaceNormal.w = 0.0f;
		return worldSpaceNormal.Normalize();
	}

};