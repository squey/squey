/**
 * \file PVWidget.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef LIBPVGL_WIDGET_H
#define LIBPVGL_WIDGET_H

namespace PVGL {
class PVWidgetManager;
class PVContainer;

struct LibGLDecl PVAllocation {
	int x;
	int y;
	int width;
	int height;
};

struct LibGLDecl PVRequisition {
	int width;
	int height;
};

/**
 * \class PVWidget
 */
class LibGLDecl PVWidget {
protected:
	PVWidgetManager *widget_manager;       //!<
	PVContainer     *parent;            //!<
	PVAllocation     allocation;        //!<
	PVRequisition    requisition;       //!<
	bool visible;                     //!<

	/**
	 * Constructor.
	 *
	 * @param widget_manager A pointer to our widget manager.
	 */
	PVWidget(PVWidgetManager *widget_manager);
public:
	PVRequisition  min_size;          //!<
	virtual ~PVWidget();
	const PVRequisition &get_requisition()const {return requisition;}
	/**
	 * Display the widget.
	 */
	virtual void draw() = 0;

	virtual void allocate_size(const PVAllocation &new_allocation);
	/**
	 * Move the widget.
	 *
	 * @param x The new x position on screen.
	 * @param y The new y position on screen.
	 */
	void move(int x, int y);

	/**
	 * Make the widget visible.
	 */
	void show();

	/**
	 * Make the widget invisible.
	 */
	void hide();
	bool is_visible()const{return visible;}
	void set_parent(PVContainer *parent_){parent = parent_;}
	bool in_allocation(int x, int y);
};
}
#endif
