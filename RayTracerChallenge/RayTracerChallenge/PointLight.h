#pragma once

#include "Vec4.h"

struct PointLight
{
	// Data
	
	Vec4 position;
	Colorf color;


	// Functions

	PointLight(Vec4 pos, Colorf inten)
	{
		position = pos;
		color = inten;
	}

};