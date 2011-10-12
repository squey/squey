//! \file PVScatter.cpp
//! $Id: PVScatter.cpp 3081 2011-06-08 07:38:24Z aguinet $
//! Copyright (C) SÃ©bastien Tricaud 2009, 2010
//! Copyright (C) Philippe Saade 2009,2010
//! Copyright (C) Picviz Labs 2011

#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>

#define GLEW_STATIC 1
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <picviz/PVLinesProperties.h>
#include <picviz/PVView.h>

// Filters
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVArgument.h>
#include <picviz/PVSelectionFilterScatterPlotSelectionSquare.h>

#include <pvgl/PVConfig.h>
#include <pvgl/PVUtils.h>
//#include <pvgl/PVCom.h>
#include <pvgl/PVLabel.h>
#include <pvgl/PVWTK.h>

#include <pvgl/views/PVScatter.h>

/******************************************************************************
 *
 * PVGL::PVScatter::PVScatter
 *
 *****************************************************************************/
PVGL::PVScatter::PVScatter(int win_id, PVSDK::PVMessenger *message) : PVGL::PVDrawable(win_id, message),
		selection_square(this)
{
	PVLabel *title;

	PVLOG_DEBUG("PVGL::PVScatter::%s\n", __FUNCTION__);

	// Register selection filters
	REGISTER_CLASS(QString("Scatter Plot Selection Square"), Picviz::PVSelectionFilterScatterPlotSelectionSquare);

	translation = vec2(0.0f, 0.0f);
	zoom_level_x = 0;
	zoom_level_y = 0;

	first_axis = 0;
	second_axis = 1;

	draw_unselected = draw_zombie = true;
	top_bar = new PVLayout(&widget_manager);
	top_bar->move(0, 0);
	top_bar->set_size(width, 60);
	title = new PVLabel(&widget_manager, "Scatter view");
	top_bar->add(title, 0, 0, -1, -1);
}

/******************************************************************************
 *
 * PVGL::PVScatter::init
 *
 *****************************************************************************/
void PVGL::PVScatter::init(Picviz::PVView_p view)
{
	size_t temp_row_count = view->get_row_count();
	size_t max_number_of_lines_in_view = temp_row_count;//picviz_min(temp_row_count, size_t(PICVIZ_EVENTLINE_LINES_MAX));
	std::vector<std::string> attributes;

	PVLOG_DEBUG("PVGL::PVScatter::%s\n", __FUNCTION__);
	picviz_view = view;

	// We fix the initial mathematical extent of the PVGL::PVScatter
	xmin = -0.2f;
	ymin = -0.2f;
	xmax = 1.2f;
	ymax = 1.2f;

	// We initialize the 1 fundamental object of the PVView
	selection_square.init(view);

	glGenVertexArrays(1, &main_vao); PRINT_OPENGL_ERROR();
	glBindVertexArray(main_vao); PRINT_OPENGL_ERROR();

	// A TBO for the selection
	glGenBuffers(1, &tbo_selection); PRINT_OPENGL_ERROR();
	glBindBuffer(GL_TEXTURE_BUFFER, tbo_selection); PRINT_OPENGL_ERROR();
	glGenTextures(1, &tbo_selection_texture); PRINT_OPENGL_ERROR();
	glActiveTexture(GL_TEXTURE1); PRINT_OPENGL_ERROR();
	glBindTexture(GL_TEXTURE_BUFFER, tbo_selection_texture); PRINT_OPENGL_ERROR();
	glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, tbo_selection); PRINT_OPENGL_ERROR();

	// A TBO for the zombie/not-zombie
	glGenBuffers(1, &tbo_zombie); PRINT_OPENGL_ERROR();
	glBindBuffer(GL_TEXTURE_BUFFER, tbo_zombie); PRINT_OPENGL_ERROR();
	glGenTextures(1, &tbo_zombie_texture); PRINT_OPENGL_ERROR();
	glActiveTexture(GL_TEXTURE2); PRINT_OPENGL_ERROR();
	glBindTexture(GL_TEXTURE_BUFFER, tbo_zombie_texture); PRINT_OPENGL_ERROR();
	glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, tbo_zombie); PRINT_OPENGL_ERROR();
	glActiveTexture(GL_TEXTURE0); PRINT_OPENGL_ERROR();

	// The VBO holding the dot colors.
	glGenBuffers(1, &vbo_color); PRINT_OPENGL_ERROR();
	glBindBuffer(GL_ARRAY_BUFFER, vbo_color); PRINT_OPENGL_ERROR();
	glEnableVertexAttribArray(0); PRINT_OPENGL_ERROR();
	glVertexAttribPointer(0, 4, GL_UNSIGNED_BYTE, GL_TRUE, 0, BUFFER_OFFSET(0));
	attributes.push_back("color_v"); PRINT_OPENGL_ERROR();

	// The VBO holding the dot depth.
	glGenBuffers(1, &vbo_zla); PRINT_OPENGL_ERROR();
	glBindBuffer(GL_ARRAY_BUFFER, vbo_zla); PRINT_OPENGL_ERROR();
	glEnableVertexAttribArray(1); PRINT_OPENGL_ERROR();
	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	attributes.push_back("zla_v"); PRINT_OPENGL_ERROR();

	// The VBO holding the dot positions.
	glGenBuffers(1, &vbo_position); PRINT_OPENGL_ERROR();
	glBindBuffer(GL_ARRAY_BUFFER, vbo_position); PRINT_OPENGL_ERROR();
	glEnableVertexAttribArray(2); PRINT_OPENGL_ERROR();
	glBufferData(GL_ARRAY_BUFFER, max_number_of_lines_in_view * sizeof(vec2), 0, GL_DYNAMIC_DRAW); PRINT_OPENGL_ERROR();
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	attributes.push_back("position"); PRINT_OPENGL_ERROR();

	// The shader for drawing the dots.
	main_program = read_shader("scatter/dot.vert", "", "scatter/dot.frag", "", "", "", attributes);
	glUniform1i(get_uni_loc(main_program, "selection_sampler"), 1); PRINT_OPENGL_ERROR();
	glUniform1i(get_uni_loc(main_program, "zombie_sampler"), 2); PRINT_OPENGL_ERROR();

	// Restore a sane state.
	glBindVertexArray(0); PRINT_OPENGL_ERROR();
	glUseProgram(0); PRINT_OPENGL_ERROR();
	update_arrays_z();
	update_arrays_colors();
	update_arrays_selection();
	update_arrays_zombies();
	update_arrays_positions();

	selection_square.update_arrays();
}

/******************************************************************************
 *
 * PVGL::PVScatter::draw
 *
 *****************************************************************************/
void PVGL::PVScatter::draw(void)
{
	GLfloat modelview[16];

	PVLOG_DEBUG("PVGL::PVScatter::%s\n", __FUNCTION__);

	if (!picviz_view->is_consistent()) {
		return;
	}
	glMatrixMode(GL_MODELVIEW); PRINT_OPENGL_ERROR();
	glLoadIdentity(); PRINT_OPENGL_ERROR();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); PRINT_OPENGL_ERROR();
	glOrtho(-0.2, 1.2, -0.2, 1.2, -350, 350); PRINT_OPENGL_ERROR();

	// Draw the axes
	glColor3f(1.0f, 1.0f, 1.0f); PRINT_OPENGL_ERROR();
	glBegin(GL_LINES);
	glVertex2f(-0.1f,  0.0f);
	glVertex2f( 1.1f,  0.0f);
	glVertex2f( 0.0f, -0.1f);
	glVertex2f( 0.0f,  1.1f);
	glEnd(); PRINT_OPENGL_ERROR();

	// Draw the dots
	glGetFloatv(GL_MODELVIEW_MATRIX, modelview); PRINT_OPENGL_ERROR();
	glBindVertexArray(main_vao); PRINT_OPENGL_ERROR();
	glUseProgram(main_program); PRINT_OPENGL_ERROR();
	glUniformMatrix4fv(get_uni_loc(main_program, "view"), 1, GL_FALSE, modelview); PRINT_OPENGL_ERROR ();
	glUniform1f(get_uni_loc(main_program, "draw_unselected"), draw_unselected); PRINT_OPENGL_ERROR();
	glUniform1f(get_uni_loc(main_program, "draw_zombie"), draw_zombie); PRINT_OPENGL_ERROR();
	glUniform1i(get_uni_loc(main_program, "eventline_first"), picviz_view->eventline.get_first_index()); PRINT_OPENGL_ERROR();
	glUniform1i(get_uni_loc(main_program, "eventline_current"), picviz_view->eventline.get_current_index()); PRINT_OPENGL_ERROR();
	glActiveTexture(GL_TEXTURE2); PRINT_OPENGL_ERROR();
	glBindTexture(GL_TEXTURE_BUFFER, tbo_zombie_texture); PRINT_OPENGL_ERROR();
	glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, tbo_zombie); PRINT_OPENGL_ERROR();
	glActiveTexture(GL_TEXTURE1); PRINT_OPENGL_ERROR();
	glBindTexture(GL_TEXTURE_BUFFER, tbo_selection_texture); PRINT_OPENGL_ERROR();
	glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, tbo_selection); PRINT_OPENGL_ERROR();

	glDrawArrays(GL_POINTS, 0, picviz_view->get_row_count());

	// Restore a sane state.
	glActiveTexture(GL_TEXTURE0); PRINT_OPENGL_ERROR();
	glBindVertexArray(0); PRINT_OPENGL_ERROR();
	glUseProgram(0); PRINT_OPENGL_ERROR();

	selection_square.draw();

	// Draw the axis names
	glLoadIdentity(); PRINT_OPENGL_ERROR();
	glOrtho(0, width, height, 0, -1,1); PRINT_OPENGL_ERROR();
	int x = 0.3 / 1.4 * width;
	int y = (1.0 - 0.1 / 1.4) * height;
	widget_manager.draw_text(x, y, qPrintable(picviz_view->get_axis_name(first_axis)), 14);

	x = 0.1 / 1.4 * width;
	y = (1.0 - 0.3 / 1.4) * height;
	glPushMatrix(); PRINT_OPENGL_ERROR();
	glTranslatef (x, y, 0); PRINT_OPENGL_ERROR();
	glRotatef (-90, 0, 0, 1); PRINT_OPENGL_ERROR();
	glTranslatef (-x, -y, 0); PRINT_OPENGL_ERROR();
	widget_manager.draw_text(x, y, qPrintable(picviz_view->get_axis_name(second_axis)), 14);
	glPopMatrix(); PRINT_OPENGL_ERROR();

	top_bar->draw();
}

/******************************************************************************
 *
 * PVGL::PVScatter::keyboard
 *
 *****************************************************************************/
void PVGL::PVScatter::keyboard(unsigned char key, int, int)
{
	Picviz::PVStateMachine *state_machine;
	PVSDK::PVMessage       message;

	if (!picviz_view) { // The view isn't finished to be read and parsed
		return;
	}
	if (!picviz_view->is_consistent()) {
		return;
	}
	state_machine = picviz_view->state_machine;

	switch (key) {
		case ' ':
				update_arrays_z();
				update_arrays_colors();
				update_arrays_selection();
				update_arrays_zombies();
				update_arrays_positions();
				break;
		case 'a': case 'A': // Select all
				if (glutGetModifiers() & GLUT_ACTIVE_SHIFT) {
					picviz_view->floating_selection.select_all();
				} else {
					picviz_view->volatile_selection = picviz_view->layer_stack_output_layer.get_selection();
				}
				/* We deactivate the square area */
				state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_OFF);
				/* We process the view from the selection */
				picviz_view->process_from_selection();
				/* We refresh the listing */
				message.function = PVSDK_MESSENGER_FUNCTION_REFRESH_LISTING;
				message.pv_view = picviz_view;
				pv_message->post_message_to_qt(message);
				break;
		case 'f': case 'F': // Toggle fullscreen
				if (is_fullscreen()) {
					glutReshapeWindow(old_width, old_height);
					toggle_fullscreen(false);
				} else {
					old_width = get_width();
					old_height= get_height();
					glutFullScreen();
					toggle_fullscreen(true);
				}
				break;
		case 'g':
				if (top_bar->is_visible()) {
					top_bar->hide();
				} else {
					top_bar->show();
				}
				break;
		case 'u': case 'U': // Toggle unselected
				draw_unselected = !draw_unselected;
				break;
		case 'z': case 'Z': // Toggle unselected
				draw_zombie = !draw_zombie;
				break;

	}
}

/******************************************************************************
 *
 * PVGL::PVScatter::mouse_down
 *
 *****************************************************************************/
void PVGL::PVScatter::mouse_down(int /*button*/, int x, int y, int /*modifiers*/)
{
	Picviz::PVStateMachine *state_machine;
	vec2 plotted_mouse;

	PVLOG_DEBUG("PVGL::PVScatter::%s\n", __FUNCTION__);

	if (!picviz_view) { // Sanity check
		return;
	}
	if (!picviz_view->is_consistent()) {
		return;
	}
	state_machine = picviz_view->state_machine;

	// Compute the x and y position of the mouse click in the plotted coordinates.
	plotted_mouse = screen_to_plotted(vec2(x, y));

	// Store the position of this initial mouse press in the plotted coordinates system.
	selection_square.set_start_point(plotted_mouse);

	state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE);
	picviz_view->volatile_selection.select_none();
	picviz_view->floating_selection.select_none();

}

/******************************************************************************
 *
 * PVGL::PVScatter::mouse_move
 *
 *****************************************************************************/
bool PVGL::PVScatter::mouse_move(int x, int y, int /*modifiers*/)
{
	Picviz::PVStateMachine   *state_machine;
//	PVGL::PVMessage         message;
	vec2                    plotted_mouse;

	if (!picviz_view) { // The view isn't finished to be read and parsed
		return false;
	}
	if (!picviz_view->is_consistent()) {
		return false;
	}
	state_machine = picviz_view->state_machine;

	// Store the position of this last square_area position in the plotted coordinates system.
	plotted_mouse = screen_to_plotted(vec2(x, y));
	selection_square.set_end_point(plotted_mouse);

#if 0
	Picviz::PVSelectionFilter::p_type filter_org = LIB_CLASS(Picviz::PVSelectionFilter)::get().get_filter_by_name("Scatter Plot Selection Square");
	Picviz::PVSelectionFilter::p_type fclone = filter_org->clone();
	Picviz::PVSelectionFilter* filter = (Picviz::PVSelectionFilter*) fclone.get();

	PVCore::PVArgumentList arguments = filter->get_args();
	arguments["x1_min"] = selection_square.x1_min;
	arguments["x1_max"] = selection_square.x1_max;
	arguments["x2_min"] = selection_square.x2_min;
	arguments["x2_max"] = selection_square.x2_max;
	arguments["x1_axis_index"] = first_axis;
	arguments["x2_axis_index"] = second_axis;

	filter->set_args(arguments);
	
	filter->set_view(*picviz_view);
	filter->set_output(&picviz_view->volatile_selection);
	filter->operator()(picviz_view->layer_stack_output_layer.get_selection());

	/* We process the view from the selection */
//	picviz_view->process_from_selection();
#endif
	/* We refresh the listing */
// 	message.function = PVSDK_MESSENGER_REFRESH_LISTING;
// 	message.pv_view = picviz_view;
// 	pv_com->post_message_to_qt(message);

	return true;
// 	return false;
}

/******************************************************************************
 *
 * PVGL::PVScatter::mouse_up
 *
 *****************************************************************************/
bool PVGL::PVScatter::mouse_up(int /*button*/, int /*x*/, int /*y*/, int /*modifiers*/)
{
	Picviz::PVStateMachine *state_machine;
	vec2 plotted_mouse;
	PVSDK::PVMessage message;

	PVLOG_DEBUG("PVGL::PVScatter::%s\n", __FUNCTION__);

	if (!picviz_view) {
		return false;
	}
	if (!picviz_view->is_consistent()) {
		return false;
	}
	state_machine = picviz_view->state_machine;


	PVCore::PVArgumentList arguments;
	arguments["x1_min"] = selection_square.get_bottom_left_point().x;
	arguments["x1_max"] = selection_square.get_top_right_point().x;
	arguments["x2_min"] = selection_square.get_bottom_left_point().y;
	arguments["x2_max"] = selection_square.get_top_right_point().y;
	arguments["x1_axis_index"] = first_axis;
	arguments["x2_axis_index"] = second_axis;

	Picviz::PVSelectionFilterScatterPlotSelectionSquare filter(arguments);
	filter.set_view(*picviz_view);
	filter.operator()(picviz_view->layer_stack_output_layer.get_selection(), picviz_view->volatile_selection);

	/* We process the view from the selection */
	picviz_view->process_from_selection();

	message.function = PVSDK_MESSENGER_FUNCTION_UPDATE_OTHER_SELECTIONS;
	message.pv_view = get_libview();
	message.int_1 = get_window_id();
	pv_message->post_message_to_gl(message);


	/* We update the view */
	PVGL::wtk_window_need_redisplay();
	/* We update the listing */
	message.function = PVSDK_MESSENGER_FUNCTION_REFRESH_LISTING;
	message.pv_view = picviz_view;
	pv_message->post_message_to_qt(message);

// 	message.function = PVSDK_MESSENGER_SELECTION_CHANGED;
// 	message.pv_view = picviz_view;
// 	pv_com->post_message_to_qt(message);


	return true;
}

/******************************************************************************
 *
 * PVGL::PVScatter::mouse_wheel
 *
 *****************************************************************************/
void PVGL::PVScatter::mouse_wheel(int /*delta_zoom_level*/, int /*x*/, int /*y*/)
{
	PVLOG_DEBUG("PVGL::PVScatter::%s\n", __FUNCTION__);
}

/******************************************************************************
 *
 * PVGL::PVScatter::passive_motion
 *
 *****************************************************************************/
bool PVGL::PVScatter::passive_motion(int /*x*/, int /*y*/, int /*modifiers*/)
{
	PVLOG_DEBUG("PVGL::PVScatter::%s\n", __FUNCTION__);

	return false;
}

/******************************************************************************
 *
 * PVGL::PVScatter::screen_to_plotted
 *
 *****************************************************************************/
vec2 PVGL::PVScatter::screen_to_plotted(vec2 screen)
{
	PVLOG_DEBUG("PVGL::PVScatter::%s\n", __FUNCTION__);
	return vec2((xmin + xmax) / 2.0 + (xmax - xmin) * (screen.x / width - 0.5) / pow(1.2, zoom_level_x) - translation.x,
	            (ymin + ymax) / 2.0 + (ymin - ymax) * (screen.y /height - 0.5) / pow(1.2, zoom_level_y) - translation.y);
}
/******************************************************************************
 *
 * PVGL::PVScatter::set_size
 *
 *****************************************************************************/
void PVGL::PVScatter::set_size(int w, int h)
{
	PVLOG_DEBUG("PVGL::PVScatter::%s\n", __FUNCTION__);

	width = w;
	height = h;
	top_bar->set_size(width, 60);
	glViewport(0, 0, width, height);
}

/******************************************************************************
 *
 * PVGL::PVScatter::special_keys
 *
 *****************************************************************************/
void PVGL::PVScatter::special_keys(int key, int, int)
{
	Picviz::PVStateMachine *state_machine;

	if (!picviz_view) { // The view isn't finished to be read and parsed
		return;
	}
	if (!picviz_view->is_consistent()) {
		return;
	}
	state_machine = picviz_view->state_machine;

	switch (key)
		{
			case GLUT_KEY_HOME:
					break;
			case GLUT_KEY_LEFT:
					first_axis = (first_axis + picviz_view->get_axes_count() - 1) % picviz_view->get_axes_count();
					update_arrays_positions();
					break;
			case GLUT_KEY_RIGHT:
					first_axis = (first_axis + 1) % picviz_view->get_axes_count();
					update_arrays_positions();
					break;
			case GLUT_KEY_DOWN:
					second_axis = (second_axis + picviz_view->get_axes_count() - 1) % picviz_view->get_axes_count();
					update_arrays_positions();
					break;
			case GLUT_KEY_UP:
					second_axis = (second_axis + 1) % picviz_view->get_axes_count();
					update_arrays_positions();
					break;
			case GLUT_KEY_DELETE:
					break;
			case GLUT_KEY_F1:
					if (glutGetModifiers() & GLUT_ACTIVE_SHIFT) {
					} else {
					}
					break;
		}
}

/******************************************************************************
 *
 * PVGL::PVScatter::update_arrays_colors
 *
 *****************************************************************************/
void PVGL::PVScatter::update_arrays_colors(void)
{
	int nb_lines;

	PVLOG_DEBUG("PVGL::PVScatter::%s\n", __FUNCTION__);

	if (!picviz_view->is_consistent()) {
		return;
	}
	nb_lines = picviz_view->get_row_count();
	// Update the color vbo.
	glBindBuffer(GL_ARRAY_BUFFER, vbo_color); PRINT_OPENGL_ERROR();
	glBufferData(GL_ARRAY_BUFFER, nb_lines * sizeof(ubvec4), &picviz_view->post_filter_layer.get_lines_properties().table.at(0), GL_DYNAMIC_DRAW); PRINT_OPENGL_ERROR();
}

/******************************************************************************
 *
 * PVGL::PVScatter::update_arrays_selection
 *
 *****************************************************************************/
void PVGL::PVScatter::update_arrays_selection(void)
{
	PVLOG_DEBUG("PVGL::PVScatter::%s\n", __FUNCTION__);

	if (!picviz_view->is_consistent()) {
		return;
	}
	// Update the selection TBO
	glBindBuffer(GL_TEXTURE_BUFFER, tbo_selection); PRINT_OPENGL_ERROR();
	glBufferData(GL_TEXTURE_BUFFER, PICVIZ_SELECTION_NUMBER_OF_BYTES,
	             picviz_view->post_filter_layer.get_selection().get_buffer(), GL_DYNAMIC_DRAW); PRINT_OPENGL_ERROR();
	update_arrays_colors();
}

/******************************************************************************
 *
 * PVGL::PVScatter::update_arrays_z
 *
 *****************************************************************************/
void PVGL::PVScatter::update_arrays_z(void)
{
	int nb_lines;

	PVLOG_DEBUG("PVGL::PVScatter::%s\n", __FUNCTION__);

	if (!picviz_view->is_consistent()) {
		return;
	}
	nb_lines = picviz_view->get_row_count();
	glBindBuffer(GL_ARRAY_BUFFER, vbo_zla); PRINT_OPENGL_ERROR();
	glBufferData(GL_ARRAY_BUFFER, nb_lines * sizeof(GLfloat), &(picviz_view->z_level_array.get_value(0)), GL_DYNAMIC_DRAW); PRINT_OPENGL_ERROR();
}

/******************************************************************************
 *
 * PVGL::PVScatter::update_arrays_zombies
 *
 *****************************************************************************/
void PVGL::PVScatter::update_arrays_zombies(void)
{
	PVLOG_DEBUG("PVGL::PVScatter::%s\n", __FUNCTION__);

	if (!picviz_view->is_consistent()) {
		return;
	}
	// Update the zombie TBO
	glBindBuffer(GL_TEXTURE_BUFFER, tbo_zombie); PRINT_OPENGL_ERROR();
	glBufferData(GL_TEXTURE_BUFFER, PICVIZ_SELECTION_NUMBER_OF_BYTES,
	             picviz_view->layer_stack_output_layer.get_selection().get_buffer(), GL_DYNAMIC_DRAW); PRINT_OPENGL_ERROR();
}

/******************************************************************************
 *
 * PVGL::PVScatter::update_positions
 *
 *****************************************************************************/
void PVGL::PVScatter::update_arrays_positions(void)
{
	vec2  *mapped_positions_array;
	int    plotted_row_size;
	float *plotted_array = 0;

	PVLOG_DEBUG("PVGL::PVScatter::%s\n", __FUNCTION__);

	if (!picviz_view->is_consistent()) {
		return;
	}
	glBindVertexArray(main_vao); PRINT_OPENGL_ERROR();
	glBindBuffer(GL_ARRAY_BUFFER, vbo_position); PRINT_OPENGL_ERROR();
	mapped_positions_array = reinterpret_cast<vec2*>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY)); PRINT_OPENGL_ERROR();

	plotted_array = &picviz_view->get_plotted_parent()->table[0];
	plotted_row_size = picviz_view->get_original_axes_count();

	for (PVRow i = 0; i < picviz_view->get_row_count(); i++) {
		float *current_row = plotted_array + i * plotted_row_size;
		mapped_positions_array[i] = vec2(current_row[picviz_view->axes_combination.get_axis_column_index_fast(first_axis)],
																		 current_row[picviz_view->axes_combination.get_axis_column_index_fast(second_axis)]);
	}

	// Unmap the positions buffer.
	glUnmapBuffer (GL_ARRAY_BUFFER); PRINT_OPENGL_ERROR();
	glBindVertexArray(0); PRINT_OPENGL_ERROR();
}

/******************************************************************************
 *
 * PVGL::PVScatter::reinit_picviz_view
 *
 *****************************************************************************/
void PVGL::PVScatter::reinit_picviz_view()
{
	glBindVertexArray(main_vao); PRINT_OPENGL_ERROR();
	glBindBuffer(GL_ARRAY_BUFFER, vbo_position); PRINT_OPENGL_ERROR();
	glBufferData(GL_ARRAY_BUFFER, picviz_view->get_row_count() * sizeof(vec2), 0, GL_DYNAMIC_DRAW); PRINT_OPENGL_ERROR();
	glBindVertexArray(0); PRINT_OPENGL_ERROR();
	update_arrays_z();
	update_arrays_colors();
	update_arrays_selection();
	update_arrays_zombies();
	update_arrays_positions();
}
