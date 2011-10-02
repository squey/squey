//! \file PVScatter.h
//! $Id: PVScatter.h 2875 2011-05-19 04:18:05Z aguinet $
//! Copyright (C) SÃ©bastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef LIBPVGL_SCATTER_H
#define LIBPVGL_SCATTER_H

#include <vector>

#include <picviz/PVView.h>

#include <pvgl/PVUtils.h>
#include <pvgl/PVFonts.h>
#include <pvgl/views/PVScatterPlotSelectionSquare.h>
#include <pvgl/PVWidgetManager.h>
#include <pvgl/PVLayout.h>

#include <pvgl/PVDrawable.h>

namespace PVGL {
/**
 *
 */
class LibGLDecl PVScatter : public PVDrawable {
	PVWidgetManager widget_manager; //!<
	GLuint main_vao;                //!<
	GLuint tbo_selection;           //!<
	GLuint tbo_selection_texture;   //!<
	GLuint tbo_zombie;              //!<
	GLuint tbo_zombie_texture;      //!<
	GLuint vbo_color;               //!<
	GLuint vbo_zla;                 //!<
	GLuint vbo_position;            //!<
	GLuint main_program;            //!<
	int    old_width;               //!<
	int    old_height;              //!<
	bool   draw_unselected;         //!<
	bool   draw_zombie;             //!<
	PVCol  first_axis;              //!<
	PVCol  second_axis;
	PVLayout *top_bar;
	PVScatterPlotSelectionSquare selection_square;

public: //FIXME!
	float xmin;                               //!<
	float ymin;                               //!<
	float xmax;                               //!<
	float ymax;                               //!<

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
	PVScatter(int window_id, PVSDK::PVMessenger *message);
	/**
	 *
	 */
	void init(Picviz::PVView_p view);

	//! \name drawing
	//! \{
	/**
	 *
	 */
	void draw(void);

	/**
	 * @param screen
	 *
	 * @return
	 */
	vec2 screen_to_plotted(vec2 screen);

	/**
	 *
	 * @param width
	 * @param height
	 */
	void set_size(int width, int height);

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
	 *
	 * @param button
	 * @param x
	 * @param y
	 * @param modifiers
	 */
	void mouse_down(int button, int x, int y, int modifiers);

	/**
	 * @param x
	 * @param y
	 * @param modifiers
	 */
	bool mouse_move(int x, int y, int modifiers);

	/**
	 *
	 * @param button
	 * @param x
	 * @param y
	 * @param modifiers
	 */
	bool mouse_up(int button, int x, int y, int modifiers);

	/**
	 *
	 * @param x
	 * @param y
	 * @param modifiers
	 */
	bool passive_motion(int x, int y, int modifiers);

	/**
	 *
	 */
	void update_arrays_z(void);

	/**
	 *
	 */
	void update_arrays_colors(void);

	/**
	 *
	 */
	void update_arrays_selection(void);

	/**
	 *
	 */
	void update_arrays_zombies(void);

	/**
	 *
	 */
	void update_arrays_positions(void);

  void reinit_picviz_view();
};
}
#endif
