/**
 * \file PVScatterPlotSelectionSquare.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#define GLEW_STATIC 1
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <picviz/PVView.h>

#include <pvgl/PVUtils.h>
#include <pvgl/views/PVParallel.h>
#include <pvgl/views/PVScatterPlotSelectionSquare.h>

/******************************************************************************
 *
 * PVGL::PVScatterPlotSelectionSquare::init
 *
 *****************************************************************************/
void PVGL::PVScatterPlotSelectionSquare::init(Picviz::PVView_sp view)
{
	PVLOG_DEBUG("PVGL::PVScatterPlotSelectionSquare::%s\n", __FUNCTION__);

	pv_view = view;
	start_point = vec2(-10.0f, -10.0f);
	end_point = start_point;

	std::vector<std::string> attributes;
	glGenVertexArrays(1, &vao); PRINT_OPENGL_ERROR();
	glBindVertexArray(vao); PRINT_OPENGL_ERROR();
	glGenBuffers(1, &vbo_position); PRINT_OPENGL_ERROR();
	glEnableVertexAttribArray(0); PRINT_OPENGL_ERROR();
	attributes.push_back("position");
	position_array.push_back(vec3(-0,0, 310.0f));
	position_array.push_back(vec3(-0,1, 310.0f));
	position_array.push_back(vec3(-1,1, 310.0f));
	position_array.push_back(vec3(-1,0, 310.0f));
	glBindBuffer(GL_ARRAY_BUFFER, vbo_position); PRINT_OPENGL_ERROR();
	glBufferData(GL_ARRAY_BUFFER, position_array.size() * sizeof(vec3), &position_array[0], GL_STATIC_DRAW); PRINT_OPENGL_ERROR();
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); PRINT_OPENGL_ERROR();

	glGenBuffers(1, &vbo_indices); PRINT_OPENGL_ERROR();
	indices_array.push_back(0);
	indices_array.push_back(1);
	indices_array.push_back(2);
	indices_array.push_back(3);
	indices_array.push_back(0);
	glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, vbo_indices); PRINT_OPENGL_ERROR();
	glBufferData (GL_ELEMENT_ARRAY_BUFFER, indices_array.size() * sizeof (GLushort), &indices_array[0], GL_STATIC_DRAW); PRINT_OPENGL_ERROR();

	glBindVertexArray(0); PRINT_OPENGL_ERROR();
	program = read_shader("scatter/selection.vert", "", "scatter/selection.frag", "", "", "", attributes);
	glUseProgram(0); PRINT_OPENGL_ERROR();
}

/******************************************************************************
 *
 * PVGL::PVScatterPlotSelectionSquare::draw
 *
 *****************************************************************************/
void PVGL::PVScatterPlotSelectionSquare::draw(void)
{
	GLfloat m[16];
	Picviz::PVStateMachine *state_machine = pv_view->state_machine;

	PVLOG_DEBUG("PVGL::PVScatterPlotSelectionSquare::%s\n", __FUNCTION__);

	if (!pv_view->is_consistent()) {
		return;
	}
	update_arrays();

	/* We check the ANTIALIASING mode */
	if (state_machine->is_antialiased()) {
		/* We activate ANTIALISASING */
		glEnable(GL_BLEND); PRINT_OPENGL_ERROR();
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); PRINT_OPENGL_ERROR();
		glEnable(GL_LINE_SMOOTH); PRINT_OPENGL_ERROR();
		glHint(GL_LINE_SMOOTH_HINT, GL_FASTEST); PRINT_OPENGL_ERROR();
	} else {
		glDisable(GL_BLEND); PRINT_OPENGL_ERROR();
		glDisable(GL_LINE_SMOOTH); PRINT_OPENGL_ERROR();
	}

	/* We only draw the square area if the SQUARE_AREA_MODE requires it */
// 	if (state_machine->get_square_area_mode () != Picviz::StateMachine::AREA_MODE_OFF) {
		// Set the LineWidth.
		glLineWidth(1.5f); PRINT_OPENGL_ERROR();

		// Rendering
		glUseProgram(program); PRINT_OPENGL_ERROR();
// 		if (view->is_update_line_dirty()) {
			glUniform4f(get_uni_loc(program, "color"), 1.0, 0.0, 0.0, 1.0); PRINT_OPENGL_ERROR();
// 		} else {
// 			glUniform4f(get_uni_loc(program, "color"), 255.0/255.0, 100.0/255.0, 30.0/255.0, 255.0/255.0);
// 		}
		glGetFloatv(GL_MODELVIEW_MATRIX, m); PRINT_OPENGL_ERROR();
		glUniformMatrix4fv(get_uni_loc(program, "modelview"), 1, GL_FALSE, m); PRINT_OPENGL_ERROR();
		glBindVertexArray(vao); PRINT_OPENGL_ERROR();
		glDrawElements(GL_LINE_STRIP, indices_array.size(), GL_UNSIGNED_SHORT, 0); PRINT_OPENGL_ERROR();
		glBindVertexArray(0); PRINT_OPENGL_ERROR();
		glUseProgram(0); PRINT_OPENGL_ERROR();

// 	}
}

/******************************************************************************
 *
 * PVGL::PVScatterPlotSelectionSquare::set_start_point
 *
 *****************************************************************************/
void PVGL::PVScatterPlotSelectionSquare::set_end_point(vec2 point)
{
	end_point = point;

	// We check if the end point is at the right of the start point
	if (start_point.x < end_point.x) {
		bottom_left_point.x = start_point.x;
		top_right_point.x = end_point.x;
	} else {
		bottom_left_point.x = end_point.x;
		top_right_point.x = start_point.x;
	}

	// We check if the start_point is below the end_point
	if (start_point.y < end_point.y) {
		bottom_left_point.y = start_point.y;
		top_right_point.y = end_point.y;
	} else {
		bottom_left_point.y = end_point.y;
		top_right_point.y = start_point.y;
	}
}

/******************************************************************************
 *
 * PVGL::PVScatterPlotSelectionSquare::set_start_point
 *
 *****************************************************************************/
void PVGL::PVScatterPlotSelectionSquare::set_start_point(vec2 point)
{
	start_point = point;
	end_point = point;
	bottom_left_point = point;
	top_right_point = point;
}


/******************************************************************************
 *
 * PVGL::PVScatterPlotSelectionSquare::update_arrays
 *
 *****************************************************************************/
void PVGL::PVScatterPlotSelectionSquare::update_arrays (void)
{
	PVLOG_DEBUG("PVGL::PVScatterPlotSelectionSquare::%s\n", __FUNCTION__);

	if (!pv_view->is_consistent()) {
		return;
	}
//	if ((picviz_state_machine_get_grab_mode(pv_view->state_machine) == PICVIZ_SM_GRAB_MODE_OFF)
//	    && (picviz_state_machine_get_square_area_mode(pv_view->state_machine) != PICVIZ_SM_SQUARE_AREA_MODE_OFF)) {
		GLfloat sx, sy, ex, ey;
		/* the square area mode is active */

// 		sx = pv_view->square_area.get_start_x();
// 		sy = pv_view->square_area.get_start_y();
// 		ex = pv_view->square_area.get_end_x();
// 		ey = pv_view->square_area.get_end_y();

		sx = start_point.x;
		sy = start_point.y;
		ex = end_point.x;
		ey = end_point.y;

		position_array.clear();
		position_array.push_back(vec3(sx, sy, 310.0f));
		position_array.push_back(vec3(sx, ey, 310.0f));
		position_array.push_back(vec3(ex, ey, 310.0f));
		position_array.push_back(vec3(ex, sy, 310.0f));
		glBindVertexArray(vao); PRINT_OPENGL_ERROR();
		glBindBuffer(GL_ARRAY_BUFFER, vbo_position); PRINT_OPENGL_ERROR();
		glBufferData(GL_ARRAY_BUFFER, position_array.size() * sizeof(vec3), &position_array[0], GL_STATIC_DRAW); PRINT_OPENGL_ERROR();
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); PRINT_OPENGL_ERROR();
		glBindVertexArray(0); PRINT_OPENGL_ERROR();

//	}
}

