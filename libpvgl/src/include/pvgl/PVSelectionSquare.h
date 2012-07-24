/**
 * \file PVSelectionSquare.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef LIBPVGL_SELECTION_SQUARE_H
#define LIBPVGL_SELECTION_SQUARE_H

#include <vector>

#define GLEW_STATIC 1
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <pvkernel/core/general.h>

#include <picviz/PVView_types.h>

namespace PVGL {
class PVView;

/**
 * \class PVSelectionSquare
 */
class LibGLDecl PVSelectionSquare {
	Picviz::PVView_p pv_view;                //!<
	PVView      *view;

	GLuint                vao;             //!<
	GLuint                vbo_position;    //!<
	GLuint                vbo_indices;     //!<
	std::vector<vec3>     position_array;  //!<
	std::vector<GLushort> indices_array;   //!<
	GLuint                program;         //!<

public:
	/**
	 * Constructor.
	 *
	 */
	PVSelectionSquare(PVView *view_) : view(view_) {}

	/**
	 *
	 * @param view
	 */
	void init(Picviz::PVView_p view);

	/**
	 *
	 */
	void update_arrays();

	/**
	 *
	 */
	void draw();
};
}
#endif
