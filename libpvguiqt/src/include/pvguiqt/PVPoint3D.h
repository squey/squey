#ifndef POINT3D_H
#define POINT3D_H

#include "math.h"

#include <qglobal.h>

namespace PVGuiQt
{

struct PVPoint3D
{
    float x, y, z;

    PVPoint3D()
        : x(0)
        , y(0)
        , z(0)
    {
    }

    PVPoint3D(float x_, float y_, float z_)
        : x(x_)
        , y(y_)
        , z(z_)
    {
    }

    PVPoint3D operator+(const PVPoint3D &p) const
    {
        return PVPoint3D(*this) += p;
    }

    PVPoint3D operator-(const PVPoint3D &p) const
    {
        return PVPoint3D(*this) -= p;
    }

    PVPoint3D operator*(float f) const
    {
        return PVPoint3D(*this) *= f;
    }


    PVPoint3D &operator+=(const PVPoint3D &p)
    {
        x += p.x;
        y += p.y;
        z += p.z;
        return *this;
    }

    PVPoint3D &operator-=(const PVPoint3D &p)
    {
        x -= p.x;
        y -= p.y;
        z -= p.z;
        return *this;
    }

    PVPoint3D &operator*=(float f)
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
    float &operator[](unsigned int index) {
        Q_ASSERT(index < 3);
        return (&x)[index];
    }

    const float &operator[](unsigned int index) const {
        Q_ASSERT(index < 3);
        return (&x)[index];
    }
};

inline float dot(const PVPoint3D &a, const PVPoint3D &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline PVPoint3D cross(const PVPoint3D &a, const PVPoint3D &b)
{
    return PVPoint3D(a.y * b.z - a.z * b.y,
                   a.z * b.x - a.x * b.z,
                   a.x * b.y - a.y * b.x);
}

}

#endif
