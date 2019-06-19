#pragma once

#include "Vec4.h"
#include "Ray.h"

struct Sphere
{
	// Data
	Vec4 origin;
	float radius;
	int id;
	Matrix4 tranformation;

	// Functions
	Sphere(int in_id)
	{
		origin = Vec4::Point(0.0f, 0.0f, 0.0f);
		radius = 1.0f;
		id = in_id;
		tranformation = Matrix4::Identity();
	}

	Sphere(Vec4 in_origin, float in_radius, int in_id)
	{
		origin = in_origin;
		radius = in_radius;
		id = in_id;
		tranformation = Matrix4::Identity();
	}

	void Intersect(Ray &ray)
	{
		if (tranformation.IsInvertible())
		{
			Ray rayTransformed;
			rayTransformed.origin = ray.origin;
			rayTransformed.direction = ray.direction;
			Matrix4 placeholder = tranformation.Inverse();
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

	void AddTranformation(Matrix4 &rhs)
	{
		tranformation = rhs.MMult(tranformation);
	}
};