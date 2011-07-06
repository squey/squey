//! \file PVSelectionSquare.h
//! $Id: PVSelectionSquare.h 2875 2011-05-19 04:18:05Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009, 2010
//! Copyright (C) Philippe Saade 2009, 2010
//! Copyright (C) Picviz Labs 2011

#ifndef LIBPVGL_SELECTION_SQUARE_H
#define LIBPVGL_SELECTION_SQUARE_H

#include <vector>

#define GLEW_STATIC 1
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <pvcore/general.h>

#include <picviz/PVView.h>

namespace PVGL {
class PVView;

/**
 * \class PVSelectionSquare
 */
class LibExport PVSelectionSquare {
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
