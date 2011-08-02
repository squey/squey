//! \file types.h
//! $Id: types.h 2387 2011-04-17 17:24:35Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVBASE_TYPES_H
#define PVBASE_TYPES_H

#include <QtGlobal>

typedef int pv_column;
typedef int pv_row;

typedef int PVCol;
typedef quint64 PVRow;

typedef int pvcol;
typedef quint64 pvrow;

#define _U_ __attribute__((unused))

#ifdef WIN32
#ifndef uint32_t
	typedef unsigned int uint32_t;
#endif
#ifndef uint16_t
	typedef unsigned short uint16_t;
#endif
#ifndef uint8_t
	typedef unsigned char uint8_t;
#endif
#ifndef int16_t
	typedef short int16_t;
#endif
#ifndef ssize_t
        typedef signed long ssize_t;
	/* typedef int ssize_t; */
#endif
#endif

typedef signed char pv_int8_t;
typedef unsigned char pv_uint8_t;
typedef signed short pv_int16_6;
typedef unsigned short pv_uint16_t;

typedef quint64 chunk_index;

#ifdef __cplusplus		/* FIXME: We need this since we still have C code somewhere. That should all be C++ */


/**
 *
 */
struct vec2
{
	float x; //!< The x coordinate of the vector.
	float y; //!< The y coordinate of the vector.
	vec2(float x_=0, float y_=0):x(x_),y(y_){}
	vec2&operator+=(const vec2&v){x+=v.x;y+=v.y;return *this;}
	vec2 operator-(const vec2&v)const{return vec2(x-v.x,y-v.y);}
};

/**
 *
 */
struct vec3
{
	float x; //!< The x coordinate of the vector.
	float y; //!< The y coordinate of the vector.
	float z; //!< The z coordinate of the vector.
	vec3(float x_, float y_, float z_):x(x_),y(y_),z(z_){}
};

/**
 *
 */
struct vec4
{
	float x; //!< The x coordinate of the vector.
	float y; //!< The y coordinate of the vector.
	float z; //!< The z coordinate of the vector.
	float w; //!< The w coordinate of the vector.
	vec4(float x_, float y_, float z_, float w_):x(x_),y(y_),z(z_),w(w_){}
};

/**
 *
 */
struct ubvec4
{
	unsigned char x; //!< The x coordinate of the vector.
	unsigned char y; //!< The y coordinate of the vector.
	unsigned char z; //!< The z coordinate of the vector.
	unsigned char w; //!< The w coordinate of the vector.
	ubvec4(unsigned char x_ = 0, unsigned char y_ = 0, unsigned char z_ = 0, unsigned char w_ = 255):x(x_),y(y_),z(z_),w(w_){}
	ubvec4(const unsigned char *v):x(v[0]),y(v[1]),z(v[2]),w(v[3]){}
};

#endif // __cplusplus

#endif	/* PVBASE_TYPES_H */
