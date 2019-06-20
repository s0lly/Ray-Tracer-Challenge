#include "Canvas.h"
#include "Matrix4.h"
#include "Vec4.h"
#include "Ray.h"
#include "Sphere.h"
#include "World.h"
#include "ComputeInfo.h""

#include <cuda_runtime.h>

int mainCUDA();


struct Projectile
{
	Vec4 pos;
	Vec4 vel;
};

struct Environment
{
	Vec4 gravity;
	Vec4 wind;
};


static Projectile Tick(Environment &env, Projectile &proj)
{
	Vec4 pos = proj.pos + proj.vel;
	Vec4 vel = proj.vel + env.gravity + env.wind;
	return Projectile{ pos, vel };
}


int main()
{
	//float *test;
	//cudaMallocManaged(&test, 100 * sizeof(float));



	
	//Matrix4 mat = {3.0f, -9.0f, 7.0f, 3.0f, 3.0f, -8.0f, 2.0f, -9.0f, -4.0f, 4.0f, 4.0f, 1.0f, -6.0f, 5.0f, -1.0f, 1.0f };
	//Matrix4 mat2 = { 8.0f, 2.0f, 2.0f, 2.0f, 3.0f, -1.0f, 7.0f, 0.0f, 7.0f, 0.0f, 5.0f, 4.0f, 6.0f, -2.0f, 0.0f, 5.0f };
	//Matrix4 mat3 = Matrix4::Scale(2.0f, 3.0f, 4.0f);
	//float det = mat.Determinant();
	//mat3.e[2][2] = 5.0f;

	//Vec4 point = Vec4::Point(1.0f, 0.0f, 1.0f);
	//Matrix4 transforms[3];
	//transforms[0] = Matrix4::RotationX(PI / 2.0f);
	//transforms[1] = Matrix4::Scale(5.0f, 5.0f, 5.0f);
	//transforms[2] = Matrix4::Translation(10.0f, 5.0f, 7.0f);
	//Matrix4 transformer = Matrix4::Transformer(transforms, 3);
	//
	//Vec4 matCheck = transformer.MMult(point);

	
	//Projectile p = Projectile{ Vec4::Point(0.0f, 1.0f, 0.0f), Vec4::Vec(1.0f, 1.5f, 0.0f).Normalize() * 10.0f };
	//Environment e = Environment{ Vec4::Vec(0.0f, -0.1f, 0.0f), Vec4::Vec(-0.01f, 0.0f, 0.0f) };
	//
	
	//
	//int tickNum = 0;
	//
	//while(p.pos.y > 0)
	//{
	//	p = Tick(e, p);
	//
	//	if (p.pos.x >= 0 && p.pos.x < c.width &&
	//		p.pos.y >= 0 && p.pos.y < c.height)
	//	{
	//		c.SetPixel((int)p.pos.x, (int)((float)c.height - p.pos.y), Colorf{ 1.0f, 0.0f, 0.0f });
	//	}
	//	
	//	tickNum++;
	//}


	

	//for (int i = 0; i < 12; i++)
	//{
	//	Vec4 point = Vec4::Point(0.0f, 200.0f, 0.0f);
	//	point = Matrix4::RotationZ(i * 2.0f * PI / 12.0f).MMult(point);
	//	point = Matrix4::Translation((float)c.width / 2.0f, (float)c.height / 2.0f, 0.0f).MMult(point);
	//	c.SetPixel((int)point.x, (int)((float)c.height - point.y), Colorf{ 1.0f, 0.0f, 0.0f });
	//}
	//
	//
	//c.CreatePPM("chapter4.ppm");

	
	//Ray ray(Vec4::Point(0.0f, 0.0f, 5.0f), Vec4::Vec(0.0f, 0.0f, 1.0f));
	//Sphere sphere(0);
	//
	//sphere.origin = Vec4::Point(0.0f, 0.0f, 0.0f);
	//sphere.radius = 1.0f;
	//
	//Matrix4 mat2 = Matrix4::Identity();
	//
	//Matrix4 mat3 = mat2.Inverse();
	//
	//ray.Transform(mat3);
	//
	//sphere.Intersect(ray);
	//
	//Intersection hit;
	//
	//if (ray.intersections.FindAndGetHit(hit))
	//{
	//	// do stuff?
	//}

	//Ray ray(Vec4::Point(1.0f, 2.0f, 3.0f), Vec4::Vec(0.0f, 1.0f, 0.0f));
	//Matrix4 translation = Matrix4::Translation(3.0f, 4.0f, 5.0f);
	//Matrix4 scale = Matrix4::Scale(2.0f, 3.0f, 4.0f);
	//Ray rayTransformed = ray.Transform(scale);


	//Sphere s(0);
	//
	//s.AddTranformation(Matrix4::RotationZ(PI / sqrt(5.0f)));
	//s.AddTranformation(Matrix4::Scale(1.0f, 0.5f, 1.0f));
	//
	//Vec4 normalTest = s.GetNormal(Vec4::Point(0.0f, -sqrt(2.0f) / 2.0f, sqrt(2.0f) / 2.0f));


	//World world;
	//Ray ray(Vec4::Point(0.0f, 0.0f, 0.75f), Vec4::Vec(0.0f, 0.0f, -1.0f));
	//world.spheres[0].material.ambient = 1.0f;
	//world.spheres[1].material.ambient = 1.0f;
	//
	//Colorf color = world.ColorAt(ray);

	//world.pointLights[0] = PointLight(Vec4::Point(0.0f, 0.25f, 0.0f), Colorf{ 1.0f, 1.0f, 1.0f });
	//
	//Intersection i(0.5f, &world.spheres[1]);
	//
	//ComputeInfo compInfo;
	//compInfo.Prepare(i, ray);
	//
	//Colorf color = world.ShadeHit(compInfo);

	//world.Intersect(ray);








	Canvas c(1600, 900);
	
	float viewBoardHalfWidth = 5.0f * (float)c.width / (float)c.height;
	float viewBoardHalfHeight = 5.0f;
	Vec4 viewBoardOrigin = Vec4::Point(0.0f, 0.0f, 10.0f);
	
	Vec4 rayOrigin = Vec4::Point(0.0f, 0.0f, -5.0f);
	
	World world;

	world.spheres[1].AddTranformation(Matrix4::Translation(-0.5f, 0.0f, -0.5f));
	world.spheres[1].material.color.g = 0.2f;
	//Sphere s(0);
	//s.material.color = Colorf{ 1.0f, 0.2f, 1.0f };
	//
	//PointLight light(Vec4::Point(-10.0f, 10.0f, -10.0), Colorf{ 1.0f, 1.0f, 1.0f });
	//
	//s.AddTranformation(Matrix4::Scale(0.5f, 1.0f, 1.0f));
	//s.AddTranformation(Matrix4::RotationY(PI / 2.0f));
	
	
	int countHits = 0;
	
	for (int i = 0; i < c.width; i++)
	{
		for (int j = 0; j < c.height; j++)
		{
			Vec4 viewBoardLoc = viewBoardOrigin + Vec4::Point((float)(((float)i - (float)(c.width / 2)) / (float)(c.width / 2)) * viewBoardHalfWidth, -(float)(((float)j - (float)(c.height / 2)) / (float)(c.height / 2)) * viewBoardHalfHeight, 0.0f);
	
			Vec4 directionFromRayToViewBoard = viewBoardLoc - rayOrigin;
	
			Ray ray(rayOrigin, directionFromRayToViewBoard.Normalize());
	
			Colorf color = world.ColorAt(ray);
	
			c.SetPixel(i, j, color);
		}
	}
	
	
	c.CreatePPM("chapter7test.ppm");
	

	int testcheck = 0;

	
	//mainCUDA();

	
}