#pragma once

#include <cmath>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#define FLOAT_EPSILON 0.00001f
#define PI 3.1415926f

CUDA_CALLABLE_MEMBER static bool Equiv(float a, float b)
{
	return (abs(a - b) < FLOAT_EPSILON);
}

struct Vec4
{
	// Data
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
	float w = 0.0f;
	

	// Functions

	CUDA_CALLABLE_MEMBER static Vec4 Point(float a, float b, float c)
	{
		return Vec4{ a, b, c, 1.0f };
	}

	CUDA_CALLABLE_MEMBER static Vec4 Vec(float a, float b, float c)
	{
		return Vec4{ a, b, c, 0.0f };
	}


	float& operator [](int j)
	{
		switch (j)
		{
		case 0: return x;
		case 1: return y;
		case 2: return z;
		case 3: return w;
		};
		return x;
	}


	CUDA_CALLABLE_MEMBER Vec4 operator +(Vec4 &rhs)
	{
		return Vec4{ x + rhs.x, y + rhs.y, z + rhs.z, w + rhs.w };
	}

	CUDA_CALLABLE_MEMBER Vec4 operator -(Vec4 &rhs)
	{
		return Vec4{ x - rhs.x, y - rhs.y, z - rhs.z, w - rhs.w };
	}

	CUDA_CALLABLE_MEMBER Vec4 operator *(Vec4 &rhs)
	{
		return Vec4{ x * rhs.x, y * rhs.y, z * rhs.z, w * rhs.w };
	}

	CUDA_CALLABLE_MEMBER Vec4 operator *(float rhs)
	{
		return Vec4{ x * rhs, y * rhs, z * rhs, w * rhs };
	}

	CUDA_CALLABLE_MEMBER Vec4 operator /(float rhs)
	{
		return Vec4{ x / rhs, y / rhs, z / rhs, w / rhs };
	}

	CUDA_CALLABLE_MEMBER Vec4 operator -()
	{
		return Vec4{ -x, -y, -z, -w };
	}

	CUDA_CALLABLE_MEMBER bool operator ==(Vec4 &rhs)
	{
		return Equiv(x, rhs.x) && Equiv(y, rhs.y) && Equiv(z, rhs.z) && Equiv(w, rhs.w);
	}

	CUDA_CALLABLE_MEMBER float MagnitudeSqrd()
	{
		return (x * x + y * y + z * z + w * w);
	}

	CUDA_CALLABLE_MEMBER float Magnitude()
	{
		return sqrt(MagnitudeSqrd());
	}

	CUDA_CALLABLE_MEMBER Vec4 Normalize()
	{
		return ((*this) / Magnitude());
	}

	CUDA_CALLABLE_MEMBER float Dot(Vec4 &rhs)
	{
		return (x * rhs.x + y * rhs.y + z * rhs.z + w * rhs.w);
	}

	CUDA_CALLABLE_MEMBER Vec4 Cross(Vec4 &rhs)
	{
		return Vec(
			y * rhs.z - z * rhs.y,
			z * rhs.x - x * rhs.z,
			x * rhs.y - y * rhs.x);
	}
};




struct Vec3
{
	// Data
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;


	// Functions

	float& operator [](int j)
	{
		switch (j)
		{
		case 0: return x;
		case 1: return y;
		case 2: return z;
		};
		return x;
	}


	CUDA_CALLABLE_MEMBER Vec3 operator +(Vec3 &rhs)
	{
		return Vec3{ x + rhs.x, y + rhs.y, z + rhs.z };
	}

	CUDA_CALLABLE_MEMBER Vec3 operator -(Vec3 &rhs)
	{
		return Vec3{ x - rhs.x, y - rhs.y, z - rhs.z };
	}

	CUDA_CALLABLE_MEMBER Vec3 operator *(Vec3 &rhs)
	{
		return Vec3{ x * rhs.x, y * rhs.y, z * rhs.z };
	}

	CUDA_CALLABLE_MEMBER Vec3 operator *(float rhs)
	{
		return Vec3{ x * rhs, y * rhs, z * rhs };
	}

	CUDA_CALLABLE_MEMBER Vec3 operator /(float rhs)
	{
		return Vec3{ x / rhs, y / rhs, z / rhs };
	}

	CUDA_CALLABLE_MEMBER Vec3 operator -()
	{
		return Vec3{ -x, -y, -z };
	}

	CUDA_CALLABLE_MEMBER bool operator ==(Vec3 &rhs)
	{
		return Equiv(x, rhs.x) && Equiv(y, rhs.y) && Equiv(z, rhs.z);
	}

	CUDA_CALLABLE_MEMBER float MagnitudeSqrd()
	{
		return (x * x + y * y + z * z);
	}

	CUDA_CALLABLE_MEMBER float Magnitude()
	{
		return sqrt(MagnitudeSqrd());
	}

	CUDA_CALLABLE_MEMBER Vec3 Normalize()
	{
		return ((*this) / Magnitude());
	}

	CUDA_CALLABLE_MEMBER float Dot(Vec3 &rhs)
	{
		return (x * rhs.x + y * rhs.y + z * rhs.z);
	}

	CUDA_CALLABLE_MEMBER Vec3 Cross(Vec3 &rhs)
	{
		return Vec3{
			y * rhs.z - z * rhs.y,
			z * rhs.x - x * rhs.z,
			x * rhs.y - y * rhs.x };
	}
};





struct Vec2
{
	// Data
	float x = 0.0f;
	float y = 0.0f;


	// Functions

	float& operator [](int j)
	{
		switch (j)
		{
		case 0: return x;
		case 1: return y;
		};
		return x;
	}


	CUDA_CALLABLE_MEMBER Vec2 operator +(Vec2 &rhs)
	{
		return Vec2{ x + rhs.x, y + rhs.y };
	}

	CUDA_CALLABLE_MEMBER Vec2 operator -(Vec2 &rhs)
	{
		return Vec2{ x - rhs.x, y - rhs.y };
	}

	CUDA_CALLABLE_MEMBER Vec2 operator *(Vec2 &rhs)
	{
		return Vec2{ x * rhs.x, y * rhs.y };
	}

	CUDA_CALLABLE_MEMBER Vec2 operator *(float rhs)
	{
		return Vec2{ x * rhs, y * rhs };
	}

	CUDA_CALLABLE_MEMBER Vec2 operator /(float rhs)
	{
		return Vec2{ x / rhs, y / rhs };
	}

	CUDA_CALLABLE_MEMBER Vec2 operator -()
	{
		return Vec2{ -x, -y };
	}

	CUDA_CALLABLE_MEMBER bool operator ==(Vec2 &rhs)
	{
		return Equiv(x, rhs.x) && Equiv(y, rhs.y);
	}

	CUDA_CALLABLE_MEMBER float MagnitudeSqrd()
	{
		return (x * x + y * y);
	}

	CUDA_CALLABLE_MEMBER float Magnitude()
	{
		return sqrt(MagnitudeSqrd());
	}

	CUDA_CALLABLE_MEMBER Vec2 Normalize()
	{
		return ((*this) / Magnitude());
	}

	CUDA_CALLABLE_MEMBER float Dot(Vec2 &rhs)
	{
		return (x * rhs.x + y * rhs.y);
	}
};




struct Colorf
{
	// Data
	float r = 0.0f;
	float g = 0.0f;
	float b = 0.0f;


	CUDA_CALLABLE_MEMBER Colorf operator +(Colorf &rhs)
	{
		return Colorf{ r + rhs.r, g + rhs.g, b + rhs.b };
	}

	CUDA_CALLABLE_MEMBER Colorf operator -(Colorf &rhs)
	{
		return Colorf{ r - rhs.r, g - rhs.g, b - rhs.b };
	}

	CUDA_CALLABLE_MEMBER Colorf operator *(Colorf &rhs)
	{
		return Colorf{ r * rhs.r, g * rhs.g, b * rhs.b };
	}

	CUDA_CALLABLE_MEMBER Colorf operator *(float rhs)
	{
		return Colorf{ r * rhs, g * rhs, b * rhs };
	}

};