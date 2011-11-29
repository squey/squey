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
 */
struct LibKernelDecl PVColor: ubvec4 {

	/**
	 * Constructor
	 */
	PVColor();
	
	/**
	 * Constructor
	 */
	PVColor(unsigned char r, unsigned char g, unsigned char b);

	/**
	 * Constructor
	 */
	PVColor(unsigned char r, unsigned char g, unsigned char b, unsigned char a);
	
	QColor toQColor() const;
	void fromQColor(QColor qcolor);

	unsigned char &r();
	unsigned char &g();
	unsigned char &b();
	unsigned char &a();
};
}

#endif /* PICVIZ_PVCOLOR_H */
