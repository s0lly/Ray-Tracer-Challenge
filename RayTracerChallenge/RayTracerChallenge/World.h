#pragma once

#include "Sphere.h"
#include "Material.h"
#include "PointLight.h"
#include "ComputeInfo.h"

struct World
{
	Shape spheres[6];
	int numSpheres;
	PointLight pointLights[1];
	int numLights;

	CUDA_CALLABLE_MEMBER World()
	{
		numSpheres = 6;

		//spheres = Sphere[numSpheres];

		spheres[0].type = SHAPE_TYPE::SHAPE_PLANE;
		//spheres[0].material.color = Colorf{ 1.0f, 0.9f, 0.9f };
		spheres[0].material.specular = 0.0f;
		spheres[0].material.reflective = 0.5f;

		spheres[1].type = SHAPE_TYPE::SHAPE_PLANE;
		//spheres[1].AddTranformation(Matrix4::Scale(10.0f, 0.01f, 10.0f));
		spheres[1].AddTranformation(Matrix4::RotationX(PI / 2.0f));
		//spheres[1].AddTranformation(Matrix4::RotationY(-PI / 4.0f));
		spheres[1].AddTranformation(Matrix4::Translation(0.0f, 0.0f, 5.0f));
		spheres[1].material = spheres[0].material;

		spheres[2].type = SHAPE_TYPE::SHAPE_PLANE;
		//spheres[2].AddTranformation(Matrix4::Scale(10.0f, 0.01f, 10.0f));
		//spheres[2].AddTranformation(Matrix4::RotationX(PI / 2.0f));
		//spheres[2].AddTranformation(Matrix4::RotationY(PI / 4.0f));
		spheres[2].AddTranformation(Matrix4::Translation(0.0f, 0.0f, 5.0f));
		spheres[2].material = spheres[0].material;

		spheres[3].AddTranformation(Matrix4::Translation(-0.5f, 1.0f, 0.5f));
		spheres[3].material.color = Colorf{ 0.1f, 1.0f, 0.5f };
		spheres[3].material.diffuse = 0.7f;
		spheres[3].material.specular = 0.3f;
		spheres[3].material.reflective = 0.5f;

		spheres[4].AddTranformation(Matrix4::Scale(0.5f, 0.5f, 0.5f));
		spheres[4].AddTranformation(Matrix4::Translation(1.5f, 0.5f, -0.5f));
		spheres[4].material.color = Colorf{ 0.5f, 1.0f, 0.1f };
		spheres[4].material.diffuse = 0.7f;
		spheres[4].material.specular = 0.3f;
		spheres[4].material.reflective = 0.5f;

		spheres[5].AddTranformation(Matrix4::Scale(0.33f, 0.33f, 0.33f));
		spheres[5].AddTranformation(Matrix4::Translation(-1.5f, 0.33f, -0.75f));
		spheres[5].material.color = Colorf{ 1.0f, 0.8f, 0.1f };
		spheres[5].material.diffuse = 0.7f;
		spheres[5].material.specular = 0.3f;
		spheres[5].material.reflective = 0.5f;

		numLights = 1;

		//pointLights = new PointLight[numLights];

		pointLights[0] = PointLight(Vec4::Point(-10.0f, 10.0f, -10.0f), Colorf{ 1.0f, 0.8f, 0.6f });
		//pointLights[1] = PointLight(Vec4::Point(10.0f, 10.0f, -10.0f), Colorf{ 0.2f, 0.2f, 1.0f });
	}

	CUDA_CALLABLE_MEMBER void Intersect(Ray &ray)
	{
		for (int i = 0; i < numSpheres; i++)
		{
			spheres[i].Intersect(ray);
		}

		ray.intersections.Sort();
	}

	CUDA_CALLABLE_MEMBER Colorf ShadeHit(ComputeInfo &compInfo, int remaining = 10)
	{
		Colorf color;
		for (int i = 0; i < numLights; i++)
		{
			bool isInShadow = IsInShadow(compInfo, pointLights[i]);
			color = color + compInfo.obj->material.Lighting(pointLights[i], compInfo.point, compInfo.eyeVec, compInfo.normalVec, isInShadow);
			color = color + ReflectedColor(compInfo, remaining);
		}
		return color;
	}

	CUDA_CALLABLE_MEMBER bool IsInShadow(ComputeInfo &compInfo, PointLight &light)
	{
		bool isInShadow = false;

		Vec4 adjustedPoint = compInfo.point + compInfo.normalVec * FLOAT_EPSILON * 100.0f;
		Vec4 pointToLight = light.position - adjustedPoint;
		float distance = pointToLight.Magnitude();
		Vec4 direction = pointToLight.Normalize();

		Ray ray(adjustedPoint, direction);
		Intersect(ray);
		Intersection hit;
		if (ray.intersections.FindAndGetHit(hit) && hit.t < distance)
		{
			isInShadow = true;
		}
		return isInShadow;
	}

	CUDA_CALLABLE_MEMBER Colorf ColorAt(Ray &ray, int remaining = 10)
	{
		Colorf color;
		
		Intersect(ray);
		Intersection hit;
		bool isHit = ray.intersections.FindAndGetHit(hit);

		if (isHit)
		{
			ComputeInfo compInfo;
			compInfo.Prepare(hit, ray);

			color = ShadeHit(compInfo, remaining);
		}
		
		return color;
	}

	CUDA_CALLABLE_MEMBER Colorf ReflectedColor(ComputeInfo &compInfo, int remaining)
	{
		if (remaining > 0 && compInfo.obj->material.reflective > 0.0f)
		{
			Ray reflectedRay = Ray(compInfo.point + compInfo.normalVec * FLOAT_EPSILON * 100.0f, compInfo.reflectedRayDirection);
			Colorf reflectedColor = ColorAt(reflectedRay, remaining - 1);
			return reflectedColor * compInfo.obj->material.reflective;
		}
		else
		{
			return Colorf{ 0.0f, 0.0f, 0.0f };
		}
	}
};