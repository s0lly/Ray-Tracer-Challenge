#include "Canvas.h"
#include "Matrix4.h"
#include "Vec4.h"

int mainCUDA(Vec4 vec4Pass);


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
	Canvas c(900, 550);
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


	

	for (int i = 0; i < 12; i++)
	{
		Vec4 point = Vec4::Point(0.0f, 200.0f, 0.0f);
		point = Matrix4::RotationZ(i * 2.0f * PI / 12.0f).MMult(point);
		point = Matrix4::Translation((float)c.width / 2.0f, (float)c.height / 2.0f, 0.0f).MMult(point);
		c.SetPixel((int)point.x, (int)((float)c.height - point.y), Colorf{ 1.0f, 0.0f, 0.0f });
	}
	
	

	c.CreatePPM("chapter4.ppm");

	Vec4 vec4Pass{ 1.0f, 1.0f };
	mainCUDA(vec4Pass);
}