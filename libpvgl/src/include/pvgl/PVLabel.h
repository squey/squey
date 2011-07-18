//! \file PVLabel.h
//! $Id: PVLabel.h 2488 2011-04-24 17:40:43Z psaade $
//! Copyright (C) Sébastien Tricaud 2009, 2010
//! Copyright (C) Philippe Saade 2009, 2010
//! Copyright (C) Picviz Labs 2011

#ifndef LIBPVGL_LABEL_H
#define LIBPVGL_LABEL_H

#include <string>

#include <pvgl/PVMisc.h>

namespace PVGL {
	
/**
 * \class PVLabel
 */
class LibGLDecl PVLabel : public PVMisc {

	std::string text;      //!<
	bool        shadow;    //!<
	ubvec4      color;     //!<
	int         font_size; //!<
	std::string font_name; //!<
	int         ascent;    //!<
public:
	/**
	 * Constructor.
	 *
	 * @param widget_manager A pointer to our widget manager.
	 * @param text           The initial text displayed by the label.
	 */
	PVLabel(PVWidgetManager *widget_manager, const std::string &text);

	/**
	 * Destructor.
	 */
	virtual ~PVLabel();

	/**
	 * @param color
	 */
	void set_color(const ubvec4 &color);

	/**
	 *
	 * @param do_shadow
	 */
	void set_shadow(bool do_shadow);

	/**
	 *
	 * @param text
	 */
	void set_text(const std::string &text);

	// drawing
	/**
	 * Display the label.
	 */
	void draw(void);

	/**
	 *
	 * @param new_allocation
	 */
	virtual void allocate_size(const PVAllocation &new_allocation);
};
}
#endif
