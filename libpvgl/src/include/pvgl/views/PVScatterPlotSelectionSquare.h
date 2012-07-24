/**
 * \file PVScatterPlotSelectionSquare.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef LIBPVGL_PVSCATTER_PLOT_SELECTION_SQUARE_H
#define LIBPVGL_PVSCATTER_PLOT_SELECTION_SQUARE_H

#include <vector>

#define GLEW_STATIC 1
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <pvkernel/core/general.h>

#include <picviz/PVView.h>

namespace PVGL {
class PVScatter;

/**
 * \class PVScatterPlotSelectionSquare
 */
class LibGLDecl PVScatterPlotSelectionSquare {
	Picviz::PVView_p pv_view;                //!<
	PVScatter      *scatter;

	std::vector<GLushort> indices_array;   //!<
	std::vector<vec3>     position_array;  //!<
	GLuint                program;         //!<
	GLuint                vao;             //!<
	GLuint                vbo_indices;     //!<
	GLuint                vbo_position;    //!<

	vec2 bottom_left_point;
	vec2 end_point;
	vec2 start_point;
	vec2 top_right_point;

public:
	/**
	 * Constructor.
	 *
	 */
	PVScatterPlotSelectionSquare(PVScatter *scatter_) : scatter(scatter_) {}

	/**
	 *
	 * @param view
	 */
	void init(Picviz::PVView_p view);

	/**
	 *
	 */
	void draw();

	/**
	 * Gets the bottom_left_point.
	 *
	 * @ return The Bottom_Left point
	 */
	 vec2 get_bottom_left_point() const { return bottom_left_point;}
	 
	/**
	 * Gets the end_point.
	 *
	 * @ return The end point
	 */
	 vec2 get_end_point() const { return end_point;}

	/**
	 * Gets the start_point.
	 *
	 * @ return The Start point
	 */
	 vec2 get_start_point() const { return start_point;}

	/**
	 * Gets the top_right_point.
	 *
	 * @ return The Top_Right point
	 */
	 vec2 get_top_right_point() const { return top_right_point;}

	/**
	 * Sets the start_point.
	 *
	 * WARNING : this will reset end_point, bottom_left_point and top_right_point
	 *           to start_point.
	 */
	 void set_start_point(vec2 point);

	/**
	 * Sets the end_point.
	 *
	 * WARNING : this will recompute bottom_left_point and top_right_point.
	 */
	 void set_end_point(vec2 point);

	/**
	 *
	 */
	void update_arrays();

};
}
#endif
