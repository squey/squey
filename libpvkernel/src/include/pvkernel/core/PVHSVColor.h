/**
 * \file PVHSVColor.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_HSVCOLOR_H
#define PVCORE_HSVCOLOR_H

//#include <pvkernel/core/general.h>
//#include <pvkernel/core/stdint.h>
#include <pvbase/types.h>
#include <stdint.h>
#include <pvkernel/cuda/constexpr.h>

#include <QColor>

//#define HSV_COLOR_NBITS_ZONE 6
//#define HSV_COLOR_MASK_ZONE 63

#define HSV_COLOR_NBITS_ZONE 5
#define HSV_COLOR_MASK_ZONE 31
#define HSV_COLOR_COUNT 192 // = (2**5)*6, without black & white

// Special colors
#define HSV_COLOR_WHITE 255
#define HSV_COLOR_BLACK 254

// Some colors that can be useful
#define HSV_COLOR_BLUE  10
#define HSV_COLOR_GREEN 59
#define HSV_COLOR_RED   126

namespace PVCore {

class PVHSVColor
{
	typedef uint8_t T;
public:
	typedef T h_type;
	static CUDA_CONSTEXPR uint8_t color_max = (1<<HSV_COLOR_NBITS_ZONE)*6;

public:
	// Unitialized, and this is wanted !
	PVHSVColor() { }
	PVHSVColor(T h_) { _h = h_; }

public:
	inline T& h() { return _h; };
	inline T  h() const { return _h; };
	static PVHSVColor* init_colors(PVRow nb_colors);
	bool is_valid() const;

public:
	void to_rgb(T& r, T& g, T& b) const;
	void to_rgb(T* rgb) const;

	void toQColor(QColor& qc) const;
	QColor toQColor() const;

private:
	T _h;
};

}

#endif
