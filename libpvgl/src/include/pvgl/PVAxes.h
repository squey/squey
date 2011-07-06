//! \file PVAxes.h
//! $Id: PVAxes.h 2875 2011-05-19 04:18:05Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009, 2010
//! Copyright (C) Philippe Saade 2009, 2010
//! Copyright (C) Picviz Labs 2011

#ifndef LIBPVGL_AXES_H
#define LIBPVGL_AXES_H

#include <vector>

#define GLEW_STATIC 1
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <pvcore/general.h>
#include <picviz/PVView.h>

namespace PVGL {
class PVView;

/**
 * \class PVAxes
 *
 * \brief Handle the drawing of the axes, unselected and selected (when in axis mode)
 */
class LibExport PVAxes {
	Picviz::PVView_p pv_view;             //!< A pointer to the Picviz::PVView we are handling.
	PVView          *view;                //!< A pointer to the PVGL::PVView we are drawing to.

	GLuint              vao;             //!< The VertexArrayObject for the axes.
	GLuint              vbo_position;    //!< The VertexBufferObject for the axes.
	std::vector<vec3>   position_array;  //!< The positions of the vertices defining the axes.
	GLuint              vbo_color;       //!< The VertexBufferObject for the colors of the axes.
	std::vector<ubvec4> color_array;     //!< The color of the vertices defining the axes.
	GLuint              program;         //!< The shader program handle used when drawing the axes.

	GLuint            vao_bg;            //!< The VertexArrayObject for the background of the selected axis.
	GLuint            vbo_bg_position;   //!< The VertexBufferObject for the background of the selected axis.
	std::vector<vec3> bg_position_array; //!< The positions of the vertices of the background of the selected axis.
	GLuint            program_bg;        //!< The shader program handle used to draw the background of the selected axis.
	bool              show_limits;

public:

	/**
	 * Constructor.
	 *
	 * @param view The PVGL::PVView we are attached to.
	 */
	PVAxes(PVView *view);

	/**
	 * Initialize everything we need to setup the rendering of the axes.
	 *
	 * @param view The Picviz::PVView we have to represent.
	 */
	void init(Picviz::PVView_p view);

	/**
	 * Update the axes positions.
	 */
	void update_arrays(void);

	/**
	 * Draw the axes.
	 *
	 * @param axes_mode A boolean telling if we should draw using the nifty axes mode style.
	 */
	void draw(bool axes_mode);

	/**
	 * Update the position of the selected axis.
	 */
	void update_arrays_bg(void);

	/**
	 * Draw the selected axis (when in axis mode) with a fancy background.
	 */
	void draw_bg(void);

	/**
	 * Draw the axes titles.
	 */
	void draw_names();

	void toggle_show_limits();
};
}
#endif
