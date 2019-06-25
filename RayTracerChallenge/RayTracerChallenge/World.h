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
		numSpheres = 6;

		spheres = new Sphere[numSpheres];

		spheres[0].AddTranformation(Matrix4::Scale(10.0f, 0.01f, 10.0f));
		spheres[0].material.color = Colorf{ 1.0f, 0.9f, 0.9f };
		spheres[0].material.specular = 0.0f;

		spheres[1].AddTranformation(Matrix4::Scale(10.0f, 0.01f, 10.0f));
		spheres[1].AddTranformation(Matrix4::RotationX(PI / 2.0f));
		spheres[1].AddTranformation(Matrix4::RotationY(-PI / 4.0f));
		spheres[1].AddTranformation(Matrix4::Translation(0.0f, 0.0f, 5.0f));
		spheres[1].material = spheres[0].material;

		spheres[2].AddTranformation(Matrix4::Scale(10.0f, 0.01f, 10.0f));
		spheres[2].AddTranformation(Matrix4::RotationX(PI / 2.0f));
		spheres[2].AddTranformation(Matrix4::RotationY(PI / 4.0f));
		spheres[2].AddTranformation(Matrix4::Translation(0.0f, 0.0f, 5.0f));
		spheres[2].material = spheres[0].material;

		spheres[3].AddTranformation(Matrix4::Translation(-0.5f, 1.0f, 0.5f));
		spheres[3].material.color = Colorf{ 0.1f, 1.0f, 0.5f };
		spheres[3].material.diffuse = 0.7f;
		spheres[3].material.specular = 0.3f;

		spheres[4].AddTranformation(Matrix4::Scale(0.5f, 0.5f, 0.5f));
		spheres[4].AddTranformation(Matrix4::Translation(1.5f, 0.5f, -0.5f));
		spheres[4].material.color = Colorf{ 0.5f, 1.0f, 0.1f };
		spheres[4].material.diffuse = 0.7f;
		spheres[4].material.specular = 0.3f;

		spheres[5].AddTranformation(Matrix4::Scale(0.33f, 0.33f, 0.33f));
		spheres[5].AddTranformation(Matrix4::Translation(-1.5f, 0.33f, -0.75f));
		spheres[5].material.color = Colorf{ 1.0f, 0.8f, 0.1f };
		spheres[5].material.diffuse = 0.7f;
		spheres[5].material.specular = 0.3f;

		numLights = 1;

		pointLights = new PointLight[numLights];

		pointLights[0] = PointLight(Vec4::Point(0.0f, 10.0f, -10.0f), Colorf{ 1.0f, 1.0f, 1.0f });
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