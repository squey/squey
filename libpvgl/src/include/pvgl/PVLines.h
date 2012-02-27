//! \file PVLines.h
//! $Id: PVLines.h 2936 2011-05-23 05:59:33Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef LIBPVGL_LINES_H
#define LIBPVGL_LINES_H

#include <vector>

#define GLEW_STATIC 1
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <pvkernel/core/general.h>

#include <picviz/PVView.h>

#include <pvgl/PVMap.h>



namespace PVGL {
class PVView;

/**
 * \class PVLines
 */
class LibGLDecl PVLines {
	/**
	 *
	 */
	struct Batch {
		GLuint vao;                         //!< The Vertex Array Object for this batch.
		GLuint vbo_position;                //!< The vbo containing all the y coordinates for this batch.
		GLuint program;                     //!< The shaders (V,G,F) program used for drawing the selected lines for this batch.
		GLuint zombie_program;              //!< The shaders (V,G,F) program used for drawing the zombie lines for this batch.
		GLuint vbo_pos_alloc_size; // hack
	};
	Picviz::PVView_p picviz_view;          //!< A pointer to the Picviz::PVView related to the lines.
	PVView          *view;                 //!<
	GLuint           vbo_color;            //!<
	GLuint           vbo_zla;              //!<
	GLuint           tbo_selection;        //!< The Texture buffer object containing the current selection (the one from the post-filter layer).
	GLuint           tbo_selection_texture; //!< The texture object attached to the #tbo_selection.

	friend class PVMap;

	std::vector<Batch>  batches;        //!<
	unsigned            nb_batches;     //!<
	int                 drawn_lines;    //!<

	// FBO stuff
	bool   main_fbo_dirty;    //!< A flag indicating if we should redraw the fbo content.
	GLuint main_fbo;          //!<
	GLuint main_fbo_tex;      //!<
	GLuint main_fbo_program;  //!<
	GLuint main_fbo_axis_mode_program;  //!< The shader program used to draw the fbo texture on the main screen, in axis mode.
	GLuint fbo_vao;      //!<
	int    fbo_width;    //!< The width of the PVGL view fbo (the width of the window plus a percentage), in pixel.
	int	   fbo_height;   //!< The height of the PVGL view fbo (the height of the window plus a percentage), in pixel.

	vec2   offset; //!<
	/**
	 *
	 */
	void   init_main_fbo();

	// Selected lines FBO
	bool   lines_fbo_dirty; //!<
	GLuint lines_fbo;       //!<
	GLuint lines_fbo_tex;   //!<
	/**
	 *
	 */
	void   init_lines_fbo();

	// FBO for zombie/non-zombie lines
	bool   zombie_fbo_dirty;   //!< A flag indicating if we should redraw the zombie fbo content.
	GLuint zombie_fbo;         //!<
	GLuint zombie_fbo_tex;     //!<
	GLuint tbo_zombie;         //!< The Texture buffer object containing the currently zombie/unzombie lines selection (the one from the "layer stack output" layer).
	GLuint tbo_zombie_texture; //!< The texture object attached to the #tbo_zombie.
	GLuint zombie_fbo_program; //!<
	int    drawn_zombie_lines; //!<
	/**
	 *
	 */
	void init_zombie_fbo();

	/**
	 *
	 * @param modelview
	 */
	void draw_zombie_lines(GLfloat modelview[16]);

	/**
	 *
	 * @param modelview
	 */
	void draw_selected_lines(GLfloat modelview[16]);

	void free_buffers();

public:
	/**
	 * Constructor.
	 *
	 * @param view_
	 */
	PVLines(PVView *view_);

	~PVLines();

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
	 *
	 */
	void draw();
	/**
	 *
	 * @param pv_view_
	 */
	void init(Picviz::PVView_p pv_view_);

	/**
	 *
	 */
	void set_main_fbo_dirty();

	/**
	 *
	 */
	//void set_unselected_fbo_dirty();

	/**
	 *
	 */
	void set_zombie_fbo_dirty();

	/**
	 *
	 * @param width
	 * @param height
	 */
	void set_size(int width, int height);

	/**
	 *
	 */
	void change_axes_count();

	/**
	 *
	 */
	void reset_offset();

	/**
	 *
	 * @param delta
	 */
	void move_offset(const vec2 &delta);

  void reinit_picviz_view();

  void update_lpr();
  void create_batches();
  void fill_vbo_colors_and_zla(GLint start, GLsizei count);
  void fill_vbo_positions(unsigned int batch_index, GLuint start, GLsizei count);
};
}
#endif
