/**
 * \file PVLayout.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef LIBPVGL_LAYOUT_H
#define LIBPVGL_LAYOUT_H

#include <list>

#include <pvgl/PVContainer.h>

namespace PVGL {
/**
 * \class PVLayout
 */
class LibGLDecl PVLayout : public PVContainer {
	/**
	 *
	 */
	struct LayoutChild {
		PVWidget *widget; //!<
		int         x1;   //!<
		int         y1;   //!<
		int         x2;   //!<
		int         y2;   //!<
	};
	
protected:
	std::list<LayoutChild> children; //!<
	
public:
	/**
	 * Constructor.
	 *
	 * @param widget_manager A pointer to our widget manager.
	 */
	PVLayout(PVWidgetManager *widget_manager);
	
	/**
	 * Display the Layout.
	 */
	virtual void draw();

	/**
	 * Add a widget into the layout.
	 *
	 * @param child  The new child to be added to the layout.
	 * @param x1     The x1 coordinate the child want to be within the layout.
	 * @param y1     The x1 coordinate the child want to be within the layout.
	 * @param x2     The y2 coordinate the child want to be within the layout.
	 * @param y2     The y2 coordinate the child want to be within the layout.
	 */
	void add(PVWidget *child, int x1, int y1, int x2, int y2);

	/**
	 * Adds a PVWidget to this layout
	 *
	 * @param child The new child PVWidget to be added to the layout.
	 */
	void add(PVWidget *child);

	/**
	 * Sets the size of this PVLayout
	 * @param width
	 * @param height
	 */
	void set_size(int width, int height);

	/**
	 *
	 * @param new_allocation
	 */
	virtual void allocate_size(const PVAllocation &new_allocation);

	/**
	 *
	 */
	virtual void size_adjust();
};
}
#endif
