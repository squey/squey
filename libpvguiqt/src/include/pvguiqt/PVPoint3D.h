/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef POINT3D_H
#define POINT3D_H

#include "math.h"

#include <qglobal.h>

namespace PVGuiQt
{

struct PVPoint3D {
	float x, y, z;

	PVPoint3D() : x(0), y(0), z(0) {}

	PVPoint3D(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

	PVPoint3D operator+(const PVPoint3D& p) const { return PVPoint3D(*this) += p; }

	PVPoint3D operator-(const PVPoint3D& p) const { return PVPoint3D(*this) -= p; }

	PVPoint3D operator*(float f) const { return PVPoint3D(*this) *= f; }

	PVPoint3D& operator+=(const PVPoint3D& p)
	{
		x += p.x;
		y += p.y;
		z += p.z;
		return *this;
	}

	PVPoint3D& operator-=(const PVPoint3D& p)
	{
		x -= p.x;
		y -= p.y;
		z -= p.z;
		return *this;
	}

	PVPoint3D& operator*=(float f)
	{
		x *= f;
		y *= f;
		z *= f;
		return *this;
	}

	PVPoint3D normalize() const
	{
		float r = 1. / sqrt(x * x + y * y + z * z);
		return PVPoint3D(x * r, y * r, z * r);
	}
	float& operator[](unsigned int index)
	{
		Q_ASSERT(index < 3);
		return (&x)[index];
	}

	const float& operator[](unsigned int index) const
	{
		Q_ASSERT(index < 3);
		return (&x)[index];
	}
};

inline float dot(const PVPoint3D& a, const PVPoint3D& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline PVPoint3D cross(const PVPoint3D& a, const PVPoint3D& b)
{
	return PVPoint3D(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
} // namespace PVGuiQt

#endif
