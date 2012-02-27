//! \file PVView.h
//! $Id: PVView.h 2982 2011-05-26 06:58:33Z dindinx $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef LIBPVGL_VIEW_H
#define LIBPVGL_VIEW_H

#include <QtCore>
#include <vector>

#include <picviz/PVView.h>

namespace PVGL {
class PVView;
}

#include <pvsdk/PVMessenger.h>

#include <pvgl/PVFonts.h>
#include <pvgl/PVUtils.h>
#include <pvgl/PVAxes.h>
#include <pvgl/PVSelectionSquare.h>
#include <pvgl/PVLines.h>
// disabled map for now #include <pvgl/PVMap.h>
#include <pvgl/PVWidgetManager.h>
#include <pvgl/PVEventLine.h>
#include <pvgl/PVLabel.h>
#include <pvgl/PVLayout.h>

#include <pvgl/PVDrawable.h>

namespace PVGL {
/**
 *
 */
class LibGLDecl PVView : public PVDrawable {
	PVSelectionSquare  selection_square;    //!<

	PVWidgetManager  widget_manager;        //!<
	PVLines lines;                          //!<
	//PVMap   map;
	PVAxes             axes;                //!<

	bool selection_dirty;          //!< A boolean to ask for a redraw of the selection
	bool update_line_dirty;
	bool size_dirty;               //!< A boolean telling if a size request has been queued.
	bool show_axes;

	PVLabel  *label_nb_lines;
	PVLabel  *label_axis_mode;
	PVLabel  *label_lpr;
	PVLabel  *label_fps;
	PVLayout *top_bar;
	PVEventLine *event_line;
	/**
	 *
	 */
	//void toggle_map();
	
public: //FIXME!

	float xmin;                               //!<
	float ymin;                               //!<
	float xmax;                               //!<
	float ymax;                               //!<

	int old_width;                            //!<
	int old_height;
	vec2 translation;                         //!<

	int zoom_level_x;                         //!<
	int zoom_level_y;                         //!<

	int last_mouse_press_position_x;          //!<
	int last_mouse_press_position_y;          //!<

public:
	/**
	 * Constructor.
	 *
	 * @param window_id
	 * @param message
	 */
	PVView(int window_id, PVSDK::PVMessenger *message);

	virtual ~PVView();

	/**
	 *
	 */
	void init(Picviz::PVView_p view);

  //! \name getters
	
	/**
	 *
	 * @return
	 */
	vec2 get_center_position();
	
	
	//! \{
	/**
	 *
	 * @return
	 */
	PVWidgetManager &get_widget_manager(){return widget_manager;}

	/**
	 *
	 * @return
	 */
	PVCol get_leftmost_visible_axis();

	/**
	 *
	 * @return
	 */
	PVLines &get_lines(){return lines;}

	/**
	 *
	 * @return
	 */
	//PVMap &get_map(){return map;}

	/**
	 *
	 * @return
	 */
	PVCol get_most_centered_visible_axis();

	/**
	 *
	 * @return true if there is an update_line event pending, false otherwise
	 */
	bool is_update_line_dirty()const{return update_line_dirty;}

	/**
	 * Check if a size request should be handled.
	 *
	 * @return true if there is a size request pending, false otherwise
	 */
	bool is_set_size_dirty()const;

	//! \}

	/**
	 *
	 */
	void set_update_line_dirty(){update_line_dirty = true;}

	//! \name updating
	//! \{
	/**
	 *
	 */
	void update_all();

	/**
	 *
	 */
	void update_axes();

	/**
	 * Update lines that are selected
	 */
	void update_lines() { lines.update_arrays_selection(); update_line_dirty = false; }

	/**
	 *
	 */
	void update_listing();

	/**
	 *
	 */
	void update_selection_except_listing() { selection_square.update_arrays(); }

	/**
	 *
	 */
	void update_selection();

	/**
	 *
	 */
	void update_set_size();
	//! \}

	void update_colors();
	void update_z();
	void update_positions();
	void update_zombies();
	void update_selections();
	void update_current_layer();


	void set_show_axes(bool show) { show_axes = show; }

	//! \name drawing
	//! \{
	/**
	 *
	 */
	void draw();

	/**
	 *
	 */
	void draw_axes(){axes.draw_bg();axes.draw(false);}

	/**
	 *
	 */
	void draw_selection_square(void){selection_square.draw();}
  //! \}

	/**
	 *
	 */
	void set_dirty(){lines.set_main_fbo_dirty(); lines.set_zombie_fbo_dirty();}

	/**
	 *
	 * @param width
	 * @param height
	 */
	void set_size(int width, int height);

	/**
	 *
	 */
	void reset_to_home(void);

	/**
	 * @param screen
	 *
	 * @return
	 */
	vec2 screen_to_plotted(vec2 screen);

	/**
	 * @param screen
	 *
	 * @return
	 */
	void update_displacement(vec2 screen);

	/**
	 * @param key
	 * @param x
	 * @param y
	 */
	void keyboard(unsigned char key, int x, int y);

	/**
	 * @param key
	 * @param x
	 * @param y
	 */
	void special_keys(int key, int x, int y);

	/**
	 *
	 * @param delta_zoom_level
	 * @param x
	 * @param y
	 */
	void mouse_wheel(int delta_zoom_level, int x, int y);

	/**
	 * @param button     A bitfield telling which of the mouse buttons are pressed (1 = left, 2 = middle, 3 = right, 4 and 5 = wheel).
	 * @param x          The x coordinate of the mouse pointer, in window space.
	 * @param y          The y coordinate of the mouse pointer, in window space.
	 * @param modifiers
	 *
	 */
	void mouse_down(int button, int x, int y, int modifiers);

	/**
	 * @param x          The x coordinate of the mouse pointer, in window space.
	 * @param y          The y coordinate of the mouse pointer, in window space.
	 * @param modifiers
	 *
	 * @return true if the event has been handle by the gui, false otherwise.
	 */
	bool mouse_move(int x, int y, int modifiers);

	/**
	 * @param button     A bitfield telling which of the mouse buttons are pressed (1 = left, 2 = middle, 3 = right, 4 and 5 = wheel).
	 * @param x          The x coordinate of the mouse pointer, in window space.
	 * @param y          The y coordinate of the mouse pointer, in window space.
	 * @param modifiers
	 *
	 * @return true if the event has been handle by the gui, false otherwise.
	 */
	bool mouse_up(int button, int x, int y, int modifiers);

	/**
	 * @param x          The x coordinate of the mouse pointer, in window space.
	 * @param y          The y coordinate of the mouse pointer, in window space.
	 * @param modifiers
	 *
	 * @return true if the event has been handle by the gui, false otherwise.
	 */
	bool passive_motion(int x, int y, int modifiers);
	//! \name Selection handling
	//! \{
	/**
	 *
	 */
	void set_selection_dirty(){selection_dirty = true;}

	/**
	 *
	 */
	void set_selection_clean(){selection_dirty = false;}

	/**
	 *
	 */
	bool is_selection_dirty(){return selection_dirty;}
	//! \}

	/**
	 *
	 */
	void change_axes_count();


	void update_axes_combination();
	void update_label_lines_selected_eventline();
	void update_label_lpr();
	void reinit_picviz_view();
	void axes_toggle_show_limits() { axes.toggle_show_limits(); }
};


}

#endif
