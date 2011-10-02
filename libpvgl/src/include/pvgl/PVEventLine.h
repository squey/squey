//! \file PVEventLine.h
//! $Id: PVEventLine.h 2875 2011-05-19 04:18:05Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef LIBPVGL_EVENT_LINE_H
#define LIBPVGL_EVENT_LINE_H

#include <vector>

#include <picviz/PVView_types.h>

#include <pvgl/PVUtils.h>
#include <pvgl/PVWidget.h>

namespace PVGL {
class PVWidgetManager;
class PVView;

/**
 *
 */
class LibGLDecl PVEventLine : public PVWidget {
	Picviz::PVView_p picviz_view;      //!< A pointer to the Picviz::PVView this eventline should represent.
	PVGL::PVView    *view;             //!<
	PVSDK::PVMessenger *pv_message;

	int last_mouse_press_position_x;  //!<
	int last_mouse_press_position_y;  //!<

	float sliders_positions[3];       //!<
	bool  prelight[3];                //!<

	bool grabbing;                     //!<
	int  grabbed_slider;               //!<
	int  mouse_diff;                  //!<

	int max_lines_interactivity;

public:
	/**
	 * Constructor.
	 *
	 * @param widget_manager
	 * @param message
	 * @param pvgl_view
	 */
	PVEventLine(PVWidgetManager *widget_manager, PVView *pvgl_view, PVSDK::PVMessenger *message);

	/**
	 * @param picviz_view
	 */
	void set_view(Picviz::PVView_p picviz_view);

	/**
	 * Change the requested size of an event-line widget.
	 *
	 * @param width  The new requested width.
	 * @param height The new requested height.
	 */
	void set_size(int width, int height);

	// drawing
	/**
	 * Display the event line.
	 */
	void draw(void);

	/**
	 * Toggle the visibility of the event line.
	 */

	/**
	 * @param button     A bitfield indicating which mouse buttons are pressed.
	 * @param x          The x coordinate of the mouse pointer, in window space.
	 * @param y          The y coordinate of the mouse pointer, in window space.
	 * @param modifiers
	 *
	 * @return true if the event has been handle by the event-line widget, false otherwise.
	 */
	bool mouse_down(int button, int x, int y, int modifiers);

	/**
	 * @param x          The x coordinate of the mouse pointer, in window space
	 * @param y          The y coordinate of the mouse pointer, in window space
	 * @param modifiers
	 *
	 * @return true if the event has been handle by the event-line widget, false otherwise.
	 */
	bool mouse_move(int x, int y, int modifiers);

	/**
	 * @param button     A bitfield indicating which mouse buttons are pressed.
	 * @param x          The x coordinate of the mouse pointer, in window space
	 * @param y          The y coordinate of the mouse pointer, in window space
	 * @param modifiers
	 *
	 * @return true if the event has been handle by the event-line widget, false otherwise.
	 */
	bool mouse_up(int button, int x, int y, int modifiers);

	/**
	 * @param x          The x coordinate of the mouse pointer, in window space
	 * @param y          The y coordinate of the mouse pointer, in window space
	 * @param modifiers
	 *
	 * @return true if the event has been handle by the event-line widget, false otherwise.
	 */
	bool passive_motion(int x, int y, int modifiers);
};
}
#endif
