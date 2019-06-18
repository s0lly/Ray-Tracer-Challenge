#pragma once

#include <cmath>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#define FLOAT_EPSILON 0.00001f

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