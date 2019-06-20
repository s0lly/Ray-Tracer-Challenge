#pragma once

#include "Sphere.h"
#include "Material.h"
#include "PointLight.h"
#include "ComputeInfo.h"

struct World
{
	Sphere *spheres;
	int numSpheres;
	PointLight *pointLights;
	int numLights;

	CUDA_CALLABLE_MEMBER World()
	{
		numSpheres = 2;

		spheres = new Sphere[numSpheres];

		spheres[0].id = 0;
		spheres[0].material.color = Colorf{ 0.8f, 1.0f, 0.6f };
		spheres[0].material.diffuse = 0.7f;
		spheres[0].material.specular = 0.2f;

		spheres[1].id = 1;
		spheres[1].AddTranformation(Matrix4::Scale(0.5f, 0.5f, 0.5f));



		numLights = 1;

		pointLights = new PointLight[numLights];

		pointLights[0] = PointLight(Vec4::Point(-10.0f, 10.0f, -10.0f), Colorf{ 1.0f, 1.0f, 1.0f });
	}

	CUDA_CALLABLE_MEMBER void Intersect(Ray &ray)
	{
		for (int i = 0; i < numSpheres; i++)
		{
			spheres[i].Intersect(ray);
		}

		ray.intersections.Sort();
	}

	CUDA_CALLABLE_MEMBER Colorf ShadeHit(ComputeInfo &compInfo)
	{
		Colorf color;
		for (int i = 0; i < numLights; i++)
		{
			color = color + compInfo.obj->material.Lighting(pointLights[i], compInfo.point, compInfo.eyeVec, compInfo.normalVec);
		}
		return color;
	}

	CUDA_CALLABLE_MEMBER Colorf ColorAt(Ray &ray)
	{
		Colorf color;
		
		Intersect(ray);
		Intersection hit;
		bool isHit = ray.intersections.FindAndGetHit(hit);

		if (isHit)
		{
			ComputeInfo compInfo;
			compInfo.Prepare(hit, ray);

			color = ShadeHit(compInfo);
		}
		
		return color;
	}
};