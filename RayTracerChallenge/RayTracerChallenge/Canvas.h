#pragma once

#include "Vec4.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>


struct Canvas
{
	// Data
	int width;
	int height;
	Colorf* pixels;


	// Functions

	Canvas(int w, int h)
	{
		width = w;
		height = h;
		pixels = new Colorf[w * h];
		std::fill_n(pixels, w * h, Colorf{ 0.0f, 0.0f, 0.0f });
	}

	~Canvas()
	{
		delete pixels;
	}

	void SetPixel(int x, int y, Colorf c)
	{
		pixels[y * width + x] = c;
	}

	Colorf GetPixel(int x, int y)
	{
		return pixels[y * width + x];
	}

	void CreatePPM(std::string name)
	{
		std::ofstream outfile;
		outfile.open(name);

		outfile << "P3\n";
		outfile << std::to_string(width) + " " + std::to_string(height) + "\n";
		outfile << std::to_string(255) + "\n";

		for (int j = 0; j <width * height; j++)
		{
			int r = (int)(pixels[j].r * 255.999f);
			int g = (int)(pixels[j].g * 255.999f);
			int b = (int)(pixels[j].b * 255.999f);

			r = r > 255 ? 255 : r;
			g = g > 255 ? 255 : g;
			b = b > 255 ? 255 : b;

			outfile <<	std::to_string(r) + " " +
						std::to_string(g) + " " +
						std::to_string(b) + "\n";
		}
		
	}

};