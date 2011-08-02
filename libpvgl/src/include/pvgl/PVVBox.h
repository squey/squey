//! \file PVVBox.h
//! $Id: PVVBox.h 2439 2011-04-22 04:15:09Z dindinx $
//! Copyright (C) Sébastien Tricaud 2009, 2010
//! Copyright (C) Philippe Saade 2009, 2010
//! Copyright (C) Picviz Labs 2011

#ifndef LIBPVGL_VBOX_H
#define LIBPVGL_VBOX_H

#include <pvkernel/core/general.h>

#include <pvgl/PVBox.h>

namespace PVGL {
/**
 *
 */
class LibGLDecl PVVBox : public PVBox {
protected:
public:
	/**
	 * Constructor.
	 *
	 * @param widget_manager A pointer to our widget manager.
	 */
	PVVBox(PVWidgetManager *widget_manager);
	virtual void allocate_size(const PVAllocation &new_allocation);
	virtual void move(int x, int y);
	virtual void size_adjust();
	virtual void pack_start(PVWidget *child, bool expand = true);
	virtual void pack_end(PVWidget *child, bool expand = true);
};
}
#endif
