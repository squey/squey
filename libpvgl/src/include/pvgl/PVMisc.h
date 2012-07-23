/**
 * \file PVMisc.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef LIBPVGL_MISC_H
#define LIBPVGL_MISC_H

#include <string>

#include <pvgl/PVWidget.h>

namespace PVGL {

/**
 * \class PVMisc
 *
 * \brief 
 */
class LibGLDecl PVMisc : public PVWidget {
protected:
	int padding_x; //!<
	int padding_y; //!<
	float align_x; //!<
	float align_y; //!<
	
	/**
	 * Constructor.
	 *
	 * @param widget_manager A pointer to our widget manager.
	 */
	PVMisc(PVWidgetManager *widget_manager);
public:
	/**
	 * @param align_x
	 * @param align_y
	 */
	void set_alignment(float align_x, float align_y);

	/**
	 * @param padding_x
	 * @param padding_y
	 */
	void set_padding(int padding_x, int padding_y);
};
}
#endif
