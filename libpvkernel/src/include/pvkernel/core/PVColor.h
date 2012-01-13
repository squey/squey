//! \file PVColor.h
//! $Id: PVColor.h 2590 2011-05-07 15:43:12Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVCORE_PVCOLOR_H
#define PVCORE_PVCOLOR_H

#include <QColor>

#include <pvbase/general.h>
#include <pvbase/types.h>

namespace PVCore {

/**
 * \struct PVColor
 * All the user-defined constructors have been deleted so that this type
 * is considered a POD in C++03.
 */
struct LibKernelDecl PVColor: ubvec4 {
	
	QColor toQColor() const;
	void fromQColor(QColor qcolor);
	static PVColor fromRgba(unsigned char r, unsigned char g, unsigned char b, unsigned char a = 0);

	inline unsigned char &r() { return x; };
	inline unsigned char &g() { return y; };
	inline unsigned char &b() { return z; };
	inline unsigned char &a() { return w; };

	inline unsigned char r() const { return x; };
	inline unsigned char g() const { return y; };
	inline unsigned char b() const { return z; };
	inline unsigned char a() const { return w; };
};
}

#endif /* PICVIZ_PVCOLOR_H */
