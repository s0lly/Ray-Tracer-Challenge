#pragma once

#include "PointLight.h"
#include "Vec4.h"

struct Material
{
	// Data

	float ambient;
	float diffuse;
	float specular;
	float shininess;
	Colorf color;

	// Functions

	Material()
	{
		ambient = 0.1f;
		diffuse = 0.9f;
		specular = 0.9f;
		shininess = 200.0f;
	}


	Colorf Lighting(PointLight &p, Vec4 &pos, Vec4 &eyeVec, Vec4 &normalVec)
	{
		Colorf effectiveColor = color * p.color;
		effectiveColor.r *= (pos.Normalize().x + 1.0f) / 2.0f;
		effectiveColor.g *= (pos.Normalize().y + 1.0f) / 2.0f;
		effectiveColor.b *= (pos.Normalize().z + 1.0f) / 2.0f;
		Colorf ambientColor = Colorf{ 0.0f, 0.0f, 0.0f };
		Colorf diffuseColor = Colorf{ 0.0f, 0.0f, 0.0f };
		Colorf specularColor = Colorf{ 0.0f, 0.0f, 0.0f };

		Vec4 lightVec = (p.position - pos).Normalize();

		ambientColor = effectiveColor * ambient;

		float lightToNomal = normalVec.Dot(lightVec);
		
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

		return (ambientColor + diffuseColor + specularColor);
	}
};