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

enum SHAPE_TYPE
{
	SHAPE_SPHERE,
	SHAPE_PLANE
};


struct Shape
{
	// Data
	Vec4 origin;
	float radius;
	int id;
	Matrix4 tranformationToWorldSpace;
	Matrix4 inverseTranformationToWorldSpace;
	Material material;
	SHAPE_TYPE type = SHAPE_TYPE::SHAPE_SPHERE;


	// Functions
	CUDA_CALLABLE_MEMBER Shape(SHAPE_TYPE type = SHAPE_TYPE::SHAPE_SPHERE)
	{
		origin = Vec4::Point(0.0f, 0.0f, 0.0f);
		radius = 1.0f;
		id = -1;
		tranformationToWorldSpace = Matrix4::Identity();
		inverseTranformationToWorldSpace = Matrix4::Identity();
	}

	CUDA_CALLABLE_MEMBER Shape(int in_id, SHAPE_TYPE type = SHAPE_TYPE::SHAPE_SPHERE)
	{
		origin = Vec4::Point(0.0f, 0.0f, 0.0f);
		radius = 1.0f;
		id = in_id;
		tranformationToWorldSpace = Matrix4::Identity();
		inverseTranformationToWorldSpace = Matrix4::Identity();
	}

	CUDA_CALLABLE_MEMBER Shape(Vec4 in_origin, float in_radius, int in_id)
	{
		origin = in_origin;
		radius = in_radius;
		id = in_id;
		tranformationToWorldSpace = Matrix4::Identity();
	}

	CUDA_CALLABLE_MEMBER void Intersect(Ray &ray)
	{
		switch (type)
		{
		case SHAPE_TYPE::SHAPE_SPHERE:
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
		}break;
		case SHAPE_TYPE::SHAPE_PLANE:
		{
			if (tranformationToWorldSpace.IsInvertible())
			{
				Ray rayTransformed;
				rayTransformed.origin = ray.origin;
				rayTransformed.direction = ray.direction;
				Matrix4 placeholder = tranformationToWorldSpace.Inverse();
				rayTransformed.Transform(placeholder);
				
				if (!Equiv(rayTransformed.direction.y, 0.0f))
				{
					float t = -rayTransformed.origin.y / rayTransformed.direction.y;
					ray.intersections.AddIntersection(t, this);
				}
			}
		}break;

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
		switch (type)
		{
		case SHAPE_TYPE::SHAPE_SPHERE:
		{
			Vec4 objectSpacePoint = inverseTranformationToWorldSpace.MMult(p);
			Vec4 objectSpaceNormal = objectSpacePoint - origin;
			Vec4 worldSpaceNormal = inverseTranformationToWorldSpace.Transpose().MMult(objectSpaceNormal);
			worldSpaceNormal.w = 0.0f;
			return worldSpaceNormal.Normalize();
		}break;
		case SHAPE_TYPE::SHAPE_PLANE:
		{
			Vec4 worldSpaceNormal = inverseTranformationToWorldSpace.Transpose().MMult(Vec4::Vec(0.0f, 1.0f, 0.0f));
			worldSpaceNormal.w = 0.0f;
			return worldSpaceNormal.Normalize();
		}break;
		
		return Vec4::Vec(0.0f, 0.0f, 0.0f);
		}
		
	}

};