#pragma once

#include "Vec4.h"

struct Matrix2
{
	// Data
	union
	{
		float e[2][2];
		float c[4];
		Vec2 rows[2];
	};



	// Functions

	Matrix2()
	{
		for (int i = 0; i < 4; i++)
		{
			c[i] = 0.0f;
		}
	}

	Matrix2(float e00, float e01,
		float e10, float e11)
	{
		e[0][0] = e00;
		e[0][1] = e01;

		e[1][0] = e10;
		e[1][1] = e11;
	}

	bool operator ==(Matrix2 &rhs)
	{
		bool check = true;
		for (int i = 0; i < 4; i++)
		{
			check = check && Equiv(c[i], rhs.c[i]);
		}
		return check;
	}

	float Determinant()
	{
		return e[0][0] * e[1][1] - e[0][1] * e[1][0];
	}

	bool IsInvertible()
	{
		return (Determinant() != 0);
	}
};


struct Matrix3
{
	// Data
	union
	{
		float e[3][3];
		float c[9];
		Vec3 rows[3];
	};



	// Functions

	Matrix3()
	{
		for (int i = 0; i < 9; i++)
		{
			c[i] = 0.0f;
		}
	}

	Matrix3(float e00, float e01, float e02,
		float e10, float e11, float e12,
		float e20, float e21, float e22)
	{
		e[0][0] = e00;
		e[0][1] = e01;
		e[0][2] = e02;

		e[1][0] = e10;
		e[1][1] = e11;
		e[1][2] = e12;

		e[2][0] = e20;
		e[2][1] = e21;
		e[2][2] = e22;
	}

	bool operator ==(Matrix3 &rhs)
	{
		bool check = true;
		for (int i = 0; i < 9; i++)
		{
			check = check && Equiv(c[i], rhs.c[i]);
		}
		return check;
	}

	Matrix2 Submatrix(int iSkip, int jSkip)
	{
		Matrix2 placeholder;

		int iDest = 0;

		for (int i = 0; i < 3; i++, iDest++)
		{
			if (i == iSkip)
			{
				i++;
			}

			int jDest = 0;

			for (int j = 0; j < 3; j++, jDest++)
			{
				if (j == jSkip)
				{
					j++;
				}
				placeholder.e[iDest][jDest] = e[i][j];
			}
		}

		return placeholder;
	}


	float Minor(int i, int j)
	{
		return Submatrix(i, j).Determinant();
	}

	float Cofactor(int i, int j)
	{
		float signOfMinor = (((i + j) % 2 == 0) ? 1.0f : -1.0f);
		return signOfMinor * Minor(i, j);
	}

	float Determinant()
	{
		float cof0 = Cofactor(0, 0);
		float cof1 = Cofactor(0, 1);
		float cof2 = Cofactor(0, 2);

		return rows[0].Dot(Vec3{ cof0, cof1, cof2 });
	}

	bool IsInvertible()
	{
		return (Determinant() != 0);
	}
};


struct Matrix4
{
	// Data
	union
	{
		float e[4][4];
		float c[16];
		Vec4 rows[4];
	};
	


	// Functions

	Matrix4()
	{
		for (int i = 0; i < 16; i++)
		{
			c[i] = 0.0f;
		}
	}

	Matrix4(float e00, float e01, float e02, float e03,
		float e10, float e11, float e12, float e13,
		float e20, float e21, float e22, float e23,
		float e30, float e31, float e32, float e33)
	{
		e[0][0] = e00;
		e[0][1] = e01;
		e[0][2] = e02;
		e[0][3] = e03;

		e[1][0] = e10;
		e[1][1] = e11;
		e[1][2] = e12;
		e[1][3] = e13;

		e[2][0] = e20;
		e[2][1] = e21;
		e[2][2] = e22;
		e[2][3] = e23;

		e[3][0] = e30;
		e[3][1] = e31;
		e[3][2] = e32;
		e[3][3] = e33;
	}

	static Matrix4 Identity()
	{
		Matrix4 placeholder;

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				if (i == j)
				{
					placeholder.e[i][j] = 1.0f;
				}
				else
				{
					placeholder.e[i][j] = 0.0f;
				}
			}
		}

		return placeholder;
	}

	bool operator ==(Matrix4 &rhs)
	{
		bool check = true;
		for (int i = 0; i < 16; i++)
		{
			check = check && Equiv(c[i], rhs.c[i]);
		}
		return check;
	}

	Vec4 Col(int i)
	{
		return Vec4{e[0][i],
					e[1][i],
					e[2][i],
					e[3][i]};
	}

	Matrix4 MMult(Matrix4 &rhs)
	{
		Matrix4 placeholder;

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				placeholder.e[i][j] = rows[i].Dot(rhs.Col(j));
			}
		}

		return placeholder;
	}

	Vec4 MMult(Vec4 &rhs)
	{
		Vec4 placeholder;

		for (int i = 0; i < 4; i++)
		{
			placeholder[i] = rows[i].Dot(rhs);
		}

		return placeholder;
	}

	Matrix4 Transpose()
	{
		Matrix4 placeholder;

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				placeholder.e[j][i] = e[i][j];
			}
		}

		return placeholder;
	}

	Matrix3 Submatrix(int iSkip, int jSkip)
	{
		Matrix3 placeholder;

		int iDest = 0;

		for (int i = 0; i < 4; i++, iDest++)
		{
			if (i == iSkip)
			{
				i++;
			}

			int jDest = 0;
			
			for (int j = 0; j < 4; j++, jDest++)
			{
				if (j == jSkip)
				{
					j++;
				}
				placeholder.e[iDest][jDest] = e[i][j];
			}
		}

		return placeholder;
	}

	float Minor(int i, int j)
	{
		return Submatrix(i, j).Determinant();
	}

	float Cofactor(int i, int j)
	{
		float signOfMinor = (((i + j) % 2 == 0) ? 1.0f : -1.0f);
		return signOfMinor * Minor(i, j);
	}

	float Determinant()
	{
		float cof0 = Cofactor(0, 0);
		float cof1 = Cofactor(0, 1);
		float cof2 = Cofactor(0, 2);
		float cof3 = Cofactor(0, 3);

		return rows[0].Dot(Vec4{ cof0, cof1, cof2, cof3 });
	}

	bool IsInvertible()
	{
		return (Determinant() != 0);
	}

	Matrix4 Inverse()
	{
		Matrix4 placeholder;

		float det = Determinant();

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				placeholder.e[i][j] = Cofactor(j, i) / det;
			}
		}

		return placeholder;
	}
};






