//! \file PVColor.h
//! $Id: PVColor.h 2590 2011-05-07 15:43:12Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVCOLOR_H
#define PICVIZ_PVCOLOR_H

#include <QColor>

#include <pvkernel/core/general.h>

namespace Picviz {

/**
 * \struct PVColor:ubvec4
 */
struct LibPicvizDecl PVColor : ubvec4 {

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
	
	QColor toQColor();
	void fromQColor(QColor qcolor);

	unsigned char &r();
	unsigned char &g();
	unsigned char &b();
	unsigned char &a();
};
}

#endif /* PICVIZ_PVCOLOR_H */
