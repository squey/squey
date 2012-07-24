/**
 * \file PVContainer.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef LIBPVGL_CONTAINER_H
#define LIBPVGL_CONTAINER_H

#include <pvgl/PVWidget.h>

namespace PVGL {
/**
 *
 */
class LibGLDecl PVContainer : public PVWidget {
protected:
	int border_width; //!<
	/**
	 * Constructor.
	 *
	 * @param widget_manager A pointer to our widget manager.
	 */
	PVContainer(PVWidgetManager *widget_manager);
public:
	/**
	 *
	 * @param new_border_width The new border width for this container.
	 */
	void set_border_width(int new_border_width);

	/**
	 *
	 */
	virtual void size_adjust() = 0;

	/**
	 *
	 * @param child The child to be added.
	 */
	virtual void add(PVWidget *child) = 0;

};
}
#endif
