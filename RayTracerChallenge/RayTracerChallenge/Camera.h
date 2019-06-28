#pragma once

#include "Matrix4.h"
#include "Vec4.h"

struct Camera
{
	float width;
	float height;
	float fovAdjustment;
	Matrix4 viewTransform;

	CUDA_CALLABLE_MEMBER Camera() {}

	CUDA_CALLABLE_MEMBER Camera(int w, int h, float in_fov)
	{
		width = w;
		height = h;
		fovAdjustment = tan(in_fov / 2.0f) / (float)height;
		viewTransform = Matrix4::Identity();
	}

	CUDA_CALLABLE_MEMBER void SetViewTransform(Vec4 &from, Vec4 &to, Vec4 &up)
	{
		Vec4 forwardVec = (to - from).Normalize();
		Vec4 leftVec = forwardVec.Cross(up.Normalize());
		Vec4 trueUpVec = leftVec.Cross(forwardVec);

		viewTransform = Matrix4::Identity();
		viewTransform.rows[0] = leftVec;
		viewTransform.rows[1] = trueUpVec;
		viewTransform.rows[2] = -forwardVec;

		viewTransform = Matrix4::Translation(from.x, from.y, from.z).MMult(viewTransform);
	}

	CUDA_CALLABLE_MEMBER Ray RayAtPixel(int i, int j)
	{
		float worldX = (((float)i + 0.5f) - (float)width / 2.0f) * fovAdjustment;
		float worldY = (((float)j + 0.5f) - (float)height / 2.0f) * fovAdjustment;


		Vec4 rayOrigin = viewTransform.Col(3);
		rayOrigin.w = 1.0f;

		Vec4 viewBoardLoc = viewTransform.MMult(Vec4::Point(-worldX, worldY, -1.0f));
		viewBoardLoc.w = 1.0f;

		Vec4 directionFromRayToViewBoard = (viewBoardLoc - rayOrigin).Normalize();
		directionFromRayToViewBoard.w = 0.0f;

		return Ray(rayOrigin, directionFromRayToViewBoard);
	}
};