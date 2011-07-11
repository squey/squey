//! \file PVSelectionSquare.cpp
//! $Id: PVSelectionSquare.cpp 2875 2011-05-19 04:18:05Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009, 2010
//! Copyright (C) Philippe Saade 2009,2010
//! Copyright (C) Picviz Labs 2011

#define GLEW_STATIC 1
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <picviz/PVView.h>

#include <pvgl/PVUtils.h>
#include <pvgl/PVView.h>
#include <pvgl/PVSelectionSquare.h>

/******************************************************************************
 *
 * PVGL::PVSelectionSquare::init
 *
 *****************************************************************************/
void PVGL::PVSelectionSquare::init(Picviz::PVView_p view)
{
	PVLOG_DEBUG("PVGL::PVSelectionSquare::%s\n", __FUNCTION__);

	pv_view = view;
	std::vector<std::string> attributes;
	glGenVertexArrays(1, &vao); PRINT_OPENGL_ERROR();
	glBindVertexArray(vao); PRINT_OPENGL_ERROR();
	glGenBuffers(1, &vbo_position); PRINT_OPENGL_ERROR();
	glEnableVertexAttribArray(0); PRINT_OPENGL_ERROR();
	attributes.push_back("position");
	position_array.push_back(vec3(-10,0, 310.0f));
	position_array.push_back(vec3(-10,1, 310.0f));
	position_array.push_back(vec3(-11,1, 310.0f));
	position_array.push_back(vec3(-11,0, 310.0f));
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
	program = read_shader("parallel/selection.vert", "", "parallel/selection.frag", "", "", "", attributes);
	glUseProgram(0); PRINT_OPENGL_ERROR(); PRINT_OPENGL_ERROR();
}

/******************************************************************************
 *
 * PVGL::PVSelectionSquare::draw
 *
 *****************************************************************************/
void PVGL::PVSelectionSquare::draw(void)
{
	GLfloat m[16];
	Picviz::PVStateMachine *state_machine = pv_view->state_machine;

	PVLOG_DEBUG("PVGL::PVSelectionSquare::%s\n", __FUNCTION__);

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
	if (state_machine->get_square_area_mode () != Picviz::PVStateMachine::AREA_MODE_OFF) {
		// Set the LineWidth.
		glLineWidth(1.5f); PRINT_OPENGL_ERROR();

		// Rendering
		glUseProgram(program); PRINT_OPENGL_ERROR();
		if (view->is_update_line_dirty()) {
			glUniform4f(get_uni_loc(program, "color"), 1.0, 0.0, 0.0, 1.0); PRINT_OPENGL_ERROR();
		} else {
			glUniform4f(get_uni_loc(program, "color"), 255.0/255.0, 100.0/255.0, 30.0/255.0, 255.0/255.0); PRINT_OPENGL_ERROR();
		}
		glGetFloatv(GL_MODELVIEW_MATRIX, m); PRINT_OPENGL_ERROR();
		glUniformMatrix4fv(get_uni_loc(program, "modelview"), 1, GL_FALSE, m); PRINT_OPENGL_ERROR();
		glBindVertexArray(vao); PRINT_OPENGL_ERROR();
		glDrawElements(GL_LINE_STRIP, indices_array.size(), GL_UNSIGNED_SHORT, 0); PRINT_OPENGL_ERROR();
		glBindVertexArray(0); PRINT_OPENGL_ERROR();
		glUseProgram(0); PRINT_OPENGL_ERROR();
	}
}

/******************************************************************************
 *
 * PVGL::PVSelectionSquare::update_arrays
 *
 *****************************************************************************/
void PVGL::PVSelectionSquare::update_arrays (void)
{
	PVLOG_DEBUG("PVGL::PVSelectionSquare::%s\n", __FUNCTION__);

	if (!pv_view->is_consistent()) {
		return;
	}
//	if ((picviz_state_machine_get_grab_mode(pv_view->state_machine) == PICVIZ_SM_GRAB_MODE_OFF)
//	    && (picviz_state_machine_get_square_area_mode(pv_view->state_machine) != PICVIZ_SM_SQUARE_AREA_MODE_OFF)) {
		GLfloat sx, sy, ex, ey;
		/* the square area mode is active */

		sx = pv_view->square_area.get_start_x();
		sy = pv_view->square_area.get_start_y();
		ex = pv_view->square_area.get_end_x();
		ey = pv_view->square_area.get_end_y();

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

