//! \file PVBox.h
//! $Id: PVBox.h 2492 2011-04-25 05:41:54Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef LIBPVGL_BOX_H
#define LIBPVGL_BOX_H

#include <list>

#include <pvgl/PVContainer.h>

namespace PVGL {
/**
 * \class PVBox.
 *
 * \brief An abstract widget class, used to stack widgets.
 *
 * Parent of #PVHBox and #PVVBox.
 */
class LibGLDecl PVBox : public PVContainer {
protected:
	/**
	* \class PVBoxChild
	*
	* \brief A child-class, hodling the needed data for handling one child with the box.
	*/
	struct PVBoxChild {
		/**
		* Constructor.
		*
		* @param child_   The child itself. Can be any #PVWidget.
		* @param expand_  A flag telling if the child should be given all the room, or only what it requested.
		*/
		PVBoxChild(PVWidget *child_, bool expand_):child(child_),expand(expand_){}
		PVWidget *child;   //!< The child itself
		bool      expand;  //!< true if the child wants to have as much room as possible, false otherwise.
	};
	std::list<PVBoxChild> end_list;   //!< A list holding all the children added to the end of the box (currently unused).
	std::list<PVBoxChild> start_list; //!< A list holding all the children added to the start of the box.
	/**
	 * Constructor.
	 *
	 * @param widget_manager A pointer to our widget manager.
	 */
	PVBox(PVWidgetManager *widget_manager);
public:
	/**
	 * Destructor.
	 *
	 * Call the desctructor of all the children.
	 */
	virtual ~PVBox();

	/**
	 * Add a new widget into the box, at the end of the start_list.
	 *
	 * Since we do not handle the end list for now, calling this function really add a child to the end of the box.
	 *
	 * @param child  The child to be added.
	 * @param expand A flag telling if the child should be expanded. See #PVBoxChild.
	 */
	virtual void pack_start(PVWidget *child, bool expand = true) = 0;

	/**
	 * Add a new widget into the box, at the end of the end_list.
	 *
	 * @note This function is currently unimplemented by either #PVHBox nor #PVVBox.
	 *
	 * @param child  The child to be added.
	 * @param expand A flag telling if the child should be expanded. See #PVBoxChild.
	 */
	virtual void pack_end(PVWidget *child, bool expand = true) = 0;

	/**
	 * Draw the Box widget and its children.
	 */
	void draw();

	/**
	 * Add a new widget into the box.
	 *
	 * @note This is a call to pack_start(child, true).
	 *
	 * @param child  The child to be added.
	 */
	void add(PVWidget* child);
};
}
#endif
