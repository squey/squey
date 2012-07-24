/**
 * \file PVHBox.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef LIBPVGL_HBOX_H
#define LIBPVGL_HBOX_H

#include <pvkernel/core/general.h>

#include <pvgl/PVBox.h>

namespace PVGL {
/**
 *
 */
class LibGLDecl PVHBox : public PVBox {
protected:
public:
	/**
	 * Constructor.
	 *
	 * @param widget_manager A pointer to our widget manager.
	 */
	PVHBox(PVWidgetManager *widget_manager);

	/**
	 * @param new_allocation
	 */
	virtual void allocate_size(const PVAllocation &new_allocation);

	/**
	 *
	 * @param x
	 * @param y
	 */
	virtual void move(int x, int y);

	/**
	 *
	 */
	virtual void size_adjust();

	/**
	 *
	 * @param child
	 * @param expand
	 */
	virtual void pack_start(PVWidget *child, bool expand = true);

	/**
	 *
	 * @param child
	 * @param expand
	 */
	virtual void pack_end(PVWidget *child, bool expand = true);
};
}
#endif
