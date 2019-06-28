/****************************************************************************************** 
 *	Chili DirectX Framework Version 16.07.20											  *	
 *	Game.cpp																			  *
 *	Copyright 2016 PlanetChili.net <http://www.planetchili.net>							  *
 *																						  *
 *	This file is part of The Chili DirectX Framework.									  *
 *																						  *
 *	The Chili DirectX Framework is free software: you can redistribute it and/or modify	  *
 *	it under the terms of the GNU General Public License as published by				  *
 *	the Free Software Foundation, either version 3 of the License, or					  *
 *	(at your option) any later version.													  *
 *																						  *
 *	The Chili DirectX Framework is distributed in the hope that it will be useful,		  *
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of						  *
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the						  *
 *	GNU General Public License for more details.										  *
 *																						  *
 *	You should have received a copy of the GNU General Public License					  *
 *	along with The Chili DirectX Framework.  If not, see <http://www.gnu.org/licenses/>.  *
 ******************************************************************************************/
#include "MainWindow.h"
#include "Game.h"

#include <cuda_runtime.h>



Game::Game( MainWindow& wnd )
	:
	wnd( wnd ),
	gfx( wnd )
{
	camera = Camera(gfx.ScreenWidth, gfx.ScreenHeight, PI / 3.0f);

	camera.SetViewTransform(Vec4::Point(0.0f, 1.5f, -5.0f), Vec4::Point(0.0f, 1.0f, 0.0f), Vec4::Vec(0.0f, 1.0f, 0.0f));

	colors = new Colorf[gfx.ScreenWidth * gfx.ScreenHeight];

	for (int i = 0; i < gfx.ScreenWidth * gfx.ScreenHeight; i++)
	{
		colors[i] = Colorf{ 0.0f, 0.0f, 0.0f };
	}
}

void Game::Go()
{
	gfx.BeginFrame();	
	UpdateModel();
	ComposeFrame();
	gfx.EndFrame();
}

void Game::UpdateModel()
{
	if (wnd.kbd.KeyIsPressed('W'))
	{
		camera.viewTransform = Matrix4::Translation(0.0f, 0.0f, 0.5f).MMult(camera.viewTransform.Inverse()).Inverse();
	}
	if (wnd.kbd.KeyIsPressed('S'))
	{
		camera.viewTransform = Matrix4::Translation(0.0f, 0.0f, -0.5f).MMult(camera.viewTransform.Inverse()).Inverse();
	}
	if (wnd.kbd.KeyIsPressed('A'))
	{
		camera.viewTransform = Matrix4::Translation(-0.5f, 0.0f, 0.0f).MMult(camera.viewTransform.Inverse()).Inverse();
	}
	if (wnd.kbd.KeyIsPressed('D'))
	{
		camera.viewTransform = Matrix4::Translation(0.5f, 0.0f, 0.0f).MMult(camera.viewTransform.Inverse()).Inverse();
	}
	if (wnd.kbd.KeyIsPressed('R'))
	{
		camera.viewTransform = Matrix4::Translation(0.0f, -0.5f, 0.0f).MMult(camera.viewTransform.Inverse()).Inverse();
	}
	if (wnd.kbd.KeyIsPressed('F'))
	{
		camera.viewTransform = Matrix4::Translation(0.0f, 0.5f, 0.0f).MMult(camera.viewTransform.Inverse()).Inverse();
	}

	if (wnd.kbd.KeyIsPressed('J'))
	{
		camera.viewTransform = Matrix4::RotationY(0.1f).MMult(camera.viewTransform.Inverse()).Inverse();
	}
	if (wnd.kbd.KeyIsPressed('L'))
	{
		camera.viewTransform = Matrix4::RotationY(-0.1f).MMult(camera.viewTransform.Inverse()).Inverse();
	}
	if (wnd.kbd.KeyIsPressed('I'))
	{
		camera.viewTransform = Matrix4::RotationX(-0.1f).MMult(camera.viewTransform.Inverse()).Inverse();
	}
	if (wnd.kbd.KeyIsPressed('K'))
	{
		camera.viewTransform = Matrix4::RotationX(0.1f).MMult(camera.viewTransform.Inverse()).Inverse();
	}
	if (wnd.kbd.KeyIsPressed('U'))
	{
		camera.viewTransform = Matrix4::RotationZ(0.1f).MMult(camera.viewTransform.Inverse()).Inverse();
	}
	if (wnd.kbd.KeyIsPressed('O'))
	{
		camera.viewTransform = Matrix4::RotationZ(-0.1f).MMult(camera.viewTransform.Inverse()).Inverse();
	}
}

void Game::ComposeFrame()
{
	//Canvas c(1800, 900);
	//
	//Camera camera(1800, 900, PI / 3.0f);
	//camera.SetViewTransform(Vec4::Point(0.0f, 1.5f, -5.0f), Vec4::Point(0.0f, 1.0f, 0.0f), Vec4::Vec(0.0f, 1.0f, 0.0f));
	//
	//World world;
	//
	//
	//for (int i = 0; i < c.width; i++)
	//{
	//	for (int j = 0; j < c.height; j++)
	//	{
	//		Ray ray = camera.RayAtPixel(i, j);
	//
	//		Colorf color = world.ColorAt(ray);
	//
	//		c.SetPixel(i, c.height - 1 - j , color);
	//	}
	//}
	//
	//p[
	//c.CreatePPM("chapter8.ppm");

	
	



	int testcheck = 0;


	mainCUDA((unsigned int*)gfx.pSysBuffer, camera);

	//for (int j = 0; j < gfx.ScreenHeight; j++)
	//{
	//	for (int i = 0; i < gfx.ScreenWidth; i++)
	//	{
	//		Color color;
	//
	//		
	//	}
	//}
}
