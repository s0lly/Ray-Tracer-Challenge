#pragma once

#include "PointLight.h"
#include "Vec4.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

struct Material
{
	// Data

	float ambient;
	float diffuse;
	float specular;
	float shininess;
	Colorf color;

	// Functions

	CUDA_CALLABLE_MEMBER Material()
	{
		color = Colorf{ 1.0f, 1.0, 1.0f };
		ambient = 0.1f;
		diffuse = 0.9f;
		specular = 0.9f;
		shininess = 200.0f;
	}


	CUDA_CALLABLE_MEMBER Colorf Lighting(PointLight &p, Vec4 &pos, Vec4 &eyeVec, Vec4 &normalVec, bool inShadow)
	{
		Colorf effectiveColor = color * p.color;

		Colorf ambientColor = Colorf{ 0.0f, 0.0f, 0.0f };
		Colorf diffuseColor = Colorf{ 0.0f, 0.0f, 0.0f };
		Colorf specularColor = Colorf{ 0.0f, 0.0f, 0.0f };

		Vec4 lightVec = (p.position - pos).Normalize();

		ambientColor = effectiveColor * ambient;

		float lightToNomal = normalVec.Dot(lightVec);
		
		if (!inShadow)
		{
			if (lightToNomal > 0.0f)
			{
				diffuseColor = effectiveColor * diffuse * lightToNomal;

				float lightReflectEye = ((-lightVec).GetReflectionOff(normalVec)).Dot(eyeVec.Normalize());
				if (lightReflectEye >= 0.0f)
				{
					float specularAmount = pow(lightReflectEye, shininess);
					specularColor = p.color * specular * specularAmount;
				}
			}
		}

		return (ambientColor + diffuseColor + specularColor);
	}
};