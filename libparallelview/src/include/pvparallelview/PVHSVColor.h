/**
 * \file PVHSVColor.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_HSVCOLOR_H
#define PVPARALLELVIEW_HSVCOLOR_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/stdint.h>

//#define HSV_COLOR_NBITS_ZONE 6
//#define HSV_COLOR_MASK_ZONE 63

#define HSV_COLOR_NBITS_ZONE 5
#define HSV_COLOR_MASK_ZONE 31

namespace PVParallelView {

class PVHSVColor
{
	typedef uint8_t T;
public:
	// Unitialized, and this is wanted !
	PVHSVColor() { }
	PVHSVColor(T h_) { _h = h_; }

public:
	inline T& h() { return _h; };
	inline T  h() const { return _h; };
	static PVHSVColor* init_colors(PVRow nb_colors);

public:
	void to_rgb(T& r, T& g, T& b) const;
	void to_rgb(T* rgb) const;

private:
	T _h;
};

}

#endif
