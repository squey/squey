//! \file PVMap.h
//! $Id: PVMap.h 2875 2011-05-19 04:18:05Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009, 2010
//! Copyright (C) Philippe Saade 2009, 2010
//! Copyright (C) Picviz Labs 2011

#ifndef LIBPVGL_MAP_H
#define LIBPVGL_MAP_H

#include <vector>

#define GLEW_STATIC 1
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <picviz/PVView.h>

#include <pvgl/PVWidget.h>
#include <pvgl/PVUtils.h>

// Forward declarations.
namespace PVGL {
class PVView;
class PVWidgetManager;
class PVLines;

/**
 * \class PVMap
 *
 */
class LibExport PVMap {
	Picviz::PVView_p  picviz_view;    //!< The Picviz::PVView this PVMap is attached to
	PVView           *view;           //!< The PVView this PVMap is representing
	PVWidgetManager  *widget_manager; //!< The PVWidgetManager that manages this PVMap
	PVLines          *lines;          //!< The PVLines represented in this PVMap

	bool   main_fbo_dirty;    //!< The "dirtyness" of the main FBO
	GLuint main_fbo;          //!< The main FBO
	GLuint main_fbo_tex;      //!< The texture of the main FBO
	GLuint main_fbo_program;  //!< The program associated with the main FBO
	GLuint main_fbo_vao;      //!< The VAO associated with the main FBO
	GLuint main_fbo_vbo;      //!< The VBO associated with the main FBO
	
	/**
	 *
	 */
	void init_fbo();

	// Selected lines FBO
	bool   lines_fbo_dirty; //!<
	GLuint lines_fbo;       //!<
	GLuint lines_fbo_tex;   //!<
	int    drawn_lines;     //!<
	/**
	 *
	 */
	void init_lines_fbo();

	/**
	 * @param modelview
	 */
	void draw_selected_lines(GLfloat modelview[16]);

	// An fbo for the zombie lines
	bool   zombie_fbo_dirty;   //!<
	GLuint zombie_fbo;         //!<
	GLuint zombie_fbo_tex;     //!<
	int    drawn_zombie_lines; //!<
	
	/**
	 *
	 */
	void init_zombie_fbo();
	
	/**
	 * @param modelview
	 */
	void draw_zombie_lines(GLfloat modelview[16]);

	GLuint mask_program;       //!<
	GLuint mask_vao;           //!<
	GLuint mask_vbo;           //!<

	/**
	 *
	 */
	void init_mask();


	bool         visible;         //!<
	PVAllocation allocation;
	bool         grabbing;         //!<
	bool         dragging;        //!<
	int          old_mouse_x;     //!<
	int          old_mouse_y;     //!<
	bool         move_view_mode;  //!<
public:

	/**
	 * Constructor.
	 *
	 * @param view
	 * @param widget_manager A pointer to our widget manager.
	 * @param lines
	 * @param width
	 * @param height
	 */
	PVMap(PVView *view, PVWidgetManager *widget_manager, PVLines *lines, int width, int height);

	bool is_panning()const{return move_view_mode;}
	/**
	 *
	 * @param picviz_view
	 */
	void init(Picviz::PVView_p picviz_view);

	/**
	 *
	 */
	void set_main_fbo_dirty();

	/**
	 *
	 */
	void set_lines_fbo_dirty();

	/**
	 *
	 */
	void set_zombie_fbo_dirty();
	/**
	 *
	 */
	void update_arrays_z();

	/**
	 *
	 */
	void update_arrays_colors();

	/**
	 *
	 */
	void update_arrays_selection();

	/**
	 *
	 */
	void update_arrays_zombies();

	/**
	 *
	 */
	void update_arrays_positions();


	/**
	 * @param width
	 * @param height
	 */
	void set_size(int width, int height);

	/**
	 *
	 */
	void draw();

	/**
	 *
	 */
	void toggle_map();

	/**
	 *
	 * @param x
	 * @param y
	 *
	 * @return
	 */
	bool mouse_down(int x, int y);

	/**
	 *
	 * @param x
	 * @param y
	 *
	 * @return
	 */
	bool mouse_move(int x, int y);

	/**
	 *
	 * @param x
	 * @param y
	 *
	 * @return
	 */
	bool mouse_up(int x, int y);
};
}
#endif
