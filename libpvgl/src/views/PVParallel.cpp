//! \file PVView.cpp
//! $Id: PVView.cpp 3139 2011-06-14 17:55:24Z stricaud $
//! Copyright (C) Sebastien Tricaud 2009, 2010
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



#include <picviz/PVView.h>

#include <pvgl/general.h>
#include <pvgl/PVConfig.h>
#include <pvgl/PVUtils.h>
#include <pvgl/PVCom.h>
#include <pvgl/PVHBox.h>
#include <pvgl/PVVBox.h>
#include <pvgl/PVEventLine.h>

#include <pvgl/views/PVParallel.h>

/******************************************************************************
 *
 * PVGL::PVView::PVView
 *
 *****************************************************************************/
PVGL::PVView::PVView(int win_id, PVCom *com) : PVGL::PVDrawable(win_id, com),
		selection_square(this),
		widget_manager(),
		lines(this),
		map(this, &widget_manager, &lines, width, height),
		axes(this)
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);
	translation = vec2(0.0f, 0.0f);
	zoom_level_x = 0;
	zoom_level_y = 0;
	selection_dirty = false;
	update_line_dirty = false;
	size_dirty = true;
	max_lines_per_redraw = pvconfig.value("pvgl/lpr", MAX_LINES_PER_REDRAW).toInt();

	// Creation of the ui.
	top_bar = new PVLayout(&widget_manager);
	top_bar->move(0, 0);
	top_bar->set_size(width, 60);

	label_nb_lines = new PVLabel(&widget_manager, "");
	label_nb_lines->set_shadow(true);

	label_axis_mode = new PVLabel(&widget_manager, "Axis mode");
	label_axis_mode->set_shadow(true);
	label_axis_mode->set_color(ubvec4(255, 0, 0, 255));
	label_axis_mode->hide();

	PVHBox *hbox = new PVHBox(&widget_manager);
	top_bar->add(hbox, 0, 0, -1, -1);
	PVVBox *vbox = new PVVBox(&widget_manager);
	hbox->pack_start(vbox);
	vbox->pack_start(label_axis_mode);
	vbox->pack_start(label_nb_lines);
	event_line = new PVEventLine(&widget_manager, this, com);
	hbox->pack_start(event_line, false);

	label_lpr = new PVLabel(&widget_manager, "LPR: 25000"); // FIXME this should use a 'const' ?
	label_lpr->set_color(ubvec4(0, 0, 0, 255));
	vbox->pack_start(label_lpr);
        
}

PVGL::PVView::~PVView()
{
	delete top_bar;
	delete label_nb_lines;
	delete label_axis_mode;
	delete event_line;
	delete label_lpr;
}

/******************************************************************************
 *
 * PVGL::PVView::init
 *
 *****************************************************************************/
void PVGL::PVView::init(Picviz::PVView_p view)
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);

	PVGL::PVDrawable::init(view);

	// We fix the initial mathematical extent of the PVGL::PVView
	xmin = -0.5f;
	ymin = -0.1f;
	xmax = picviz_view->get_column_count_as_float () - 0.5f;
	ymax = 1.3f;

	// We initialize the 5 fundamental objects of the PVView
	axes.init(view);
	selection_square.init(view);
	lines.init(view);
	map.init(view);
	event_line->set_view(view);

	/* We fill the arrays for later use */
	lines.update_arrays_positions();
	lines.update_arrays_colors();
	lines.update_arrays_z();
	lines.update_arrays_zombies();
	lines.update_arrays_selection();
	map.update_arrays_positions();
	map.update_arrays_colors();
	map.update_arrays_z();
	map.update_arrays_zombies();
	map.update_arrays_selection();

	axes.update_arrays();
	axes.update_arrays_bg();
	selection_square.update_arrays();
}

/******************************************************************************
 *
 * PVGL::PVView::draw
 *
 *****************************************************************************/
void PVGL::PVView::draw(void)
{
	float zoom_x;
	float zoom_y;
	Picviz::PVStateMachine *state_machine;

	PVLOG_HEAVYDEBUG("PVGL::PVView::%s\n", __FUNCTION__);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	if (!picviz_view || !picviz_view->is_consistent()) {
		int current_time = (glutGet(GLUT_ELAPSED_TIME) / 250) % 4;
		const char *text = "";
		switch (current_time) {
			case 0:
					text = "Loading";
					break;
			case 1:
					text = "Loading.";
					break;
			case 2:
					text = "Loading..";
					break;
			case 3:
					text = "Loading...";
					break;
		}
		glOrtho(0,width, height,0, -1,1);

		glColor4ubv(&PVGL_VIEW_LOADING_COLOR.x);
		widget_manager.draw_text(50, 50, text, 22);
		return;
	}

	if (size_dirty) {
		//return;
	}
	glEnable(GL_DEPTH_TEST);
	zoom_x = pow(1.2, zoom_level_x);
	zoom_y = pow(1.2, zoom_level_y);
	glScalef(zoom_x, zoom_y, 1.0);
	glOrtho(xmin, xmax, ymin, ymax, -350.0, 350.0);
	glTranslatef(translation.x, translation.y, 0.0);

	lines.draw();
	state_machine = picviz_view->state_machine;
	axes.draw(state_machine->is_axes_mode());
	axes.draw_bg();
	selection_square.draw();
	// screen space
	axes.draw_names();
	map.draw();

	glDisable(GL_DEPTH_TEST);

	glLoadIdentity ();
	glOrtho(0,width, height,0, -1,1);

	// Draw number of selected lines
		{
			std::stringstream ss;
			ss << "LPR: " << get_max_lines_per_redraw();
			label_lpr->set_text(ss.str());
		}
		{
			std::stringstream ss;
			ss << "Lines selected: " << picviz_view->get_number_of_selected_lines() <<
					" / " << picviz_view->eventline.get_current_index() - picviz_view->eventline.get_first_index() + 1<<
					" / " << picviz_view->eventline.get_row_count();
			label_nb_lines->set_text(ss.str());
		}
	if (picviz_view->state_machine->is_axes_mode()) {
		label_axis_mode->show();
	} else {
		label_axis_mode->hide();
	}

	top_bar->draw();
}

/******************************************************************************
 *
 * PVGL::PVView::update_all
 *
 *****************************************************************************/
void PVGL::PVView::update_all(void)
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);
	if (!picviz_view) { // Sanity check
		return;
	}
	if (!picviz_view->is_consistent()) {
		return;
	}
	axes.update_arrays();
	axes.update_arrays_bg();
	lines.update_arrays_colors();
	lines.update_arrays_z();
	lines.update_arrays_selection();
	lines.update_arrays_zombies();
	lines.update_arrays_positions();
	map.update_arrays_colors();
	map.update_arrays_z();
	map.update_arrays_selection();
	map.update_arrays_zombies();
	map.update_arrays_positions();
	selection_square.update_arrays();
}

#if 0
void PVGL::PVView::update_axes()
{
	if (!picviz_view) { // Sanity check
		return;
	}
	if (!picviz_view->is_consistent()) {
		return;
	}

	change_axes_count();
	update_all();
}
#endif

void PVGL::PVView::update_colors()
{
	if (!picviz_view) { // Sanity check
		return;
	}
	if (!picviz_view->is_consistent()) {
		return;
	}

	get_lines().update_arrays_colors();
	get_map().update_arrays_colors();
}

void PVGL::PVView::update_z()
{
	if (!picviz_view) { // Sanity check
		return;
	}
	if (!picviz_view->is_consistent()) {
		return;
	}

	get_lines().update_arrays_z();
	get_map().update_arrays_z();
}

void PVGL::PVView::update_positions()
{
	if (!picviz_view) { // Sanity check
		return;
	}
	if (!picviz_view->is_consistent()) {
		return;
	}

	get_lines().update_arrays_positions();
	get_map().update_arrays_positions();
}

void PVGL::PVView::update_zombies()
{
	if (!picviz_view) { // Sanity check
		return;
	}
	if (!picviz_view->is_consistent()) {
		return;
	}

	get_lines().update_arrays_zombies();
	get_map().update_arrays_zombies();
}

void PVGL::PVView::update_selections()
{
	if (!picviz_view) { // Sanity check
		return;
	}
	if (!picviz_view->is_consistent()) {
		return;
	}

	get_lines().update_arrays_selection();
	get_map().update_arrays_selection();
}

/******************************************************************************
 *
 * PVGL::PVView::update_axes
 *
 *****************************************************************************/
void PVGL::PVView::update_axes(void)
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);
	if (!picviz_view) { // Sanity check
		return;
	}
	if (!picviz_view->is_consistent()) {
		return;
	}
	axes.update_arrays();
	axes.update_arrays_bg();
	lines.update_arrays_selection();
	lines.update_arrays_zombies();
	map.update_arrays_selection();
	map.update_arrays_zombies();
}

/******************************************************************************
 *
 * PVGL::PVView::reset_to_home
 *
 *****************************************************************************/
void PVGL::PVView::reset_to_home(void)
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);
	if (!picviz_view) { // Sanity check
		return;
	}
	xmin = -0.5f;
	ymin = -0.1f;
	xmax = picviz_view->get_column_count_as_float() - 0.5f;
	ymax = 1.3f;
	last_mouse_press_position_x = last_mouse_press_position_y = 0;
	translation = vec2 (0.0f, 0.0f);
	zoom_level_x = 0;
	zoom_level_y = 0;
	set_dirty();
}

/******************************************************************************
 *
 * PVGL::PVView::change_axes_count
 *
 *****************************************************************************/
void PVGL::PVView::change_axes_count()
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);
	if (!picviz_view) { // Sanity check
		return;
	}
	xmax = picviz_view->get_axes_count() - 0.5f;
	lines.change_axes_count();
}

/******************************************************************************
 *
 * PVGL::PVView::screen_to_plotted
 *
 *****************************************************************************/
vec2 PVGL::PVView::screen_to_plotted(vec2 screen)
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);
	return vec2((xmin + xmax) / 2.0 + (xmax - xmin) * (screen.x / width - 0.5) / pow(1.2, zoom_level_x) - translation.x,
	            (ymin + ymax) / 2.0 + (ymin - ymax) * (screen.y /height - 0.5) / pow(1.2, zoom_level_y) - translation.y);
}

/******************************************************************************
 *
 * PVGL::PVView::update_displacement
 *
 *****************************************************************************/
void PVGL::PVView::update_displacement(vec2 screen)
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);
	/* We fetch the displacement stored in that event */
	translation.x += (screen.x - last_mouse_press_position_x) * (xmax - xmin) / (width * pow(1.2, zoom_level_x));
	translation.y -= (screen.y - last_mouse_press_position_y) * (ymax - ymin) / (height* pow(1.2, zoom_level_y));

	lines.move_offset(screen - vec2(last_mouse_press_position_x, last_mouse_press_position_y));
	last_mouse_press_position_x = screen.x;
	last_mouse_press_position_y = screen.y;
}

/******************************************************************************
 *
 * PVGL::PVView::toggle_map
 *
 *****************************************************************************/
void PVGL::PVView::toggle_map()
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);

	map.toggle_map();
}

/******************************************************************************
 *
 * PVGL::PVView::keyboard
 *
 *****************************************************************************/
void PVGL::PVView::keyboard(unsigned char key, int, int)
{
	Picviz::PVStateMachine *state_machine;
	PVGL::PVMessage       message;

	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);

	if (!picviz_view) { // The view isn't finished to be read and parsed
		return;
	}
	if (!picviz_view->is_consistent()) {
		return;
	}
	state_machine = picviz_view->state_machine;

	switch (key) {
		case ' ':
				update_all ();
				break;
		case '#': // Toggle antialiasing
				state_machine->toggle_antialiased();
				// We refresh the view.
				set_dirty();
				break;
		case 'a': case 'A': // Select all
#if 0
				if (glutGetModifiers() & GLUT_ACTIVE_SHIFT) {
					picviz_view->floating_selection.select_all();
				} else {
					//picviz_view->volatile_selection = picviz_view->layer_stack_output_layer.get_selection();
					//picviz_view->layer_stack_output_layer->selection.A2B_copy(,
					//                          picviz_view->volatile_selection);
				}
#endif
				picviz_view->select_all_nonzb_lines();
				/* We refresh the listing */
				message.function = PVGL_COM_FUNCTION_REFRESH_LISTING;
				message.pv_view = picviz_view;
				pv_com->post_message_to_qt(message);
				update_selections();
				//update_colors();
				break;
		case 'c': case 'C': // Choose a color.
				message.function = PVGL_COM_FUNCTION_SET_COLOR;
				message.pv_view = picviz_view;
				pv_com->post_message_to_qt(message);
				break;
		case 'f': case 'F':
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
		case 'k': // Commit the selection.
				if (glutGetModifiers() & GLUT_ACTIVE_ALT) {
					message.function = PVGL_COM_FUNCTION_COMMIT_SELECTION_IN_NEW_LAYER;
					message.pv_view = picviz_view;
					pv_com->post_message_to_qt(message);
				} else {
					message.function = PVGL_COM_FUNCTION_COMMIT_SELECTION_IN_CURRENT_LAYER;
					message.pv_view = picviz_view;
					pv_com->post_message_to_qt(message);
				}
				break;
		case 'm':
				toggle_map();
				break;
		case 's':
				message.function = PVGL_COM_FUNCTION_SCREENSHOT_CHOOSE_FILENAME;
				message.pv_view = picviz_view;
				message.int_1 = glutGetWindow();
				message.int_2 = glutGetModifiers();
				pv_com->post_message_to_qt(message);
				break;
		case 'u': case 'U':
				if (glutGetModifiers() & GLUT_ACTIVE_SHIFT) {
					/* We toggle */
					state_machine->toggle_gl_unselected_visibility();
					/* We refresh the view */
					//picviz_view_process_visibility(pv_view);
					get_lines().set_main_fbo_dirty();
					map.set_main_fbo_dirty();
				} else if (glutGetModifiers() & GLUT_ACTIVE_ALT) {
					/* We toggle*/
					state_machine->toggle_listing_unselected_visibility();
					/* We refresh the listing */
					message.function = PVGL_COM_FUNCTION_REFRESH_LISTING;
					message.pv_view = picviz_view;
					pv_com->post_message_to_qt(message);
				} else	{
					/* We toggle the unselected listing visibility first */
					state_machine->toggle_gl_unselected_visibility();
					// We make sure the gl is the same
					state_machine->set_listing_unselected_visible(state_machine->are_gl_unselected_visible());
					/* We refresh the view */
					//picviz_view_process_visibility(pv_view);
					get_lines().set_main_fbo_dirty();
					map.set_main_fbo_dirty();
					/* We refresh the listing */
					message.function = PVGL_COM_FUNCTION_REFRESH_LISTING;
					message.pv_view = picviz_view;
					pv_com->post_message_to_qt(message);
				}
				break;
		case 'x': case 'X':
				state_machine->toggle_axes_mode();

				// if we enter in AXES_MODE we must disable SQUARE_AREA_MODE
				if (state_machine->is_axes_mode()) {
					/* We turn SQUARE AREA mode OFF */
					state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_OFF);
					//current_view->update_axes();
				}
				break;
		case 'y': case 'Y':
				axes.toggle_show_limits();
				break;
		case 'z': case 'Z':
				if (glutGetModifiers() & GLUT_ACTIVE_SHIFT) {
					/* We toggle*/
					state_machine->toggle_gl_zombie_visibility();
					/* We refresh the view */
					get_lines().set_main_fbo_dirty();
					map.set_main_fbo_dirty();
				} else if (glutGetModifiers() & GLUT_ACTIVE_ALT) {
					/* We toggle*/
					state_machine->toggle_listing_zombie_visibility();
					/* We refresh the listing */
					message.function = PVGL_COM_FUNCTION_REFRESH_LISTING;
					message.pv_view = picviz_view;
					pv_com->post_message_to_qt(message);
				} else {
					/* We toggle the zombie listing visilibity first */
					state_machine->toggle_gl_zombie_visibility();
					// We make sure the gl is the same
					state_machine->set_listing_zombie_visible(state_machine->are_gl_zombie_visible());
					/* We refresh the view */
					get_lines().set_main_fbo_dirty();
					map.set_main_fbo_dirty();
					/* We refresh the listing */
					message.function = PVGL_COM_FUNCTION_REFRESH_LISTING;
					message.pv_view = picviz_view;
					pv_com->post_message_to_qt(message);
				}
				break;
			case 127: // Delete key from the main keyboard. In axes mode, delete the selected axis.
					if (!state_machine->is_axes_mode()) {
						break;
					}

					/* We decide to leave at least two axes... */
					if (picviz_view->get_axes_count() <= 2 ) {
						break;
					}
					picviz_view->axes_combination.remove_axis(picviz_view->active_axis);
					/* We check if we have just removed the rightmost axis */
					if (picviz_view->axes_combination.get_axes_count() == picviz_view->active_axis ) {
						picviz_view->active_axis -= 1;
					}

					change_axes_count();
					update_all();
					message.function = PVGL_COM_FUNCTION_REFRESH_LISTING; // WITH_HORIZONTAL_HEADER?
					message.pv_view = picviz_view;
					pv_com->post_message_to_qt(message);
					break;
	}
}

/******************************************************************************
 *
 * PVGL::PVView::special_keys
 *
 *****************************************************************************/
void PVGL::PVView::special_keys(int key, int, int)
{
	Picviz::PVStateMachine *state_machine;
	PVGL::PVMessage       message;

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
					reset_to_home();
					break;
			case GLUT_KEY_LEFT:
					if (state_machine->is_axes_mode()) {
						// move the selected axis itself (with shift) or just change which axis is selected.
						// Do nothing if already at the leftmost position.
						if (picviz_view->active_axis == 0) {
							break;
						}
						if (glutGetModifiers() & GLUT_ACTIVE_SHIFT) {
							picviz_view->axes_combination.move_axis_left_one_position(picviz_view->active_axis);
							picviz_view->active_axis -= 1;
						} else {
							picviz_view->active_axis -= 1;
						}
						update_all();
						update_listing();
					} else { // Move/zoom the selection square
						float x_start, x_end, x_range, x_middle;
						x_start = picviz_view->square_area.get_start_x();
						x_end = picviz_view->square_area.get_end_x();
						x_range = std::fabs(x_end - x_start);
						if (glutGetModifiers() & GLUT_ACTIVE_SHIFT) { // zoom out the selection in the x direction
							x_middle = (x_start + x_end) / 2.0f;
							x_start = x_middle + (x_start - x_middle) / 1.2f;
							x_end = x_middle + (x_end - x_middle) / 1.2f;
						} else if (glutGetModifiers() & GLUT_ACTIVE_CTRL) { // Move the selection to the left.
							x_start -= x_range;
							x_end -= x_range;
						} else { // Move the selection to left by one pixel.
							x_start -= (xmax - xmin) / get_width() / pow(1.2, zoom_level_x);
							x_end   -= (xmax - xmin) / get_width() / pow(1.2, zoom_level_x);
						}
						picviz_view->square_area.set_start_x(x_start);
						picviz_view->square_area.set_end_x(x_end);

						picviz_view->square_area.set_dirty();
						set_update_line_dirty();
						update_selection();
					}
					break;
			case GLUT_KEY_RIGHT:
					if (state_machine->is_axes_mode()) {
						/* We test if we are at the leftmost axis */
						if (picviz_view->active_axis == picviz_view->get_axes_count() - 1) {
							break;
						}
						if (glutGetModifiers() & GLUT_ACTIVE_SHIFT) { // Move axis.
							picviz_view->axes_combination.move_axis_right_one_position(picviz_view->active_axis);
							picviz_view->active_axis += 1;
						} else {
							picviz_view->active_axis += 1;
						}
						update_all();
						update_listing();
					} else {
						float x_start, x_end, x_range, x_middle;
						x_start = picviz_view->square_area.get_start_x();
						x_end = picviz_view->square_area.get_end_x();
						x_range = std::fabs(x_end - x_start);
						if (glutGetModifiers() & GLUT_ACTIVE_SHIFT) { // zoom out the selection in the x direction
							x_middle = (x_start + x_end) / 2.0f;
							x_start = x_middle + (x_start - x_middle) * 1.2f;
							x_end = x_middle + (x_end - x_middle) * 1.2f;
						} else if (glutGetModifiers() & GLUT_ACTIVE_CTRL) { // Move the selection to the right
							x_start += x_range;
							x_end += x_range;
						} else { // Move the selection to the right by one pixel
							x_start += (xmax - xmin) / get_width() / pow(1.2, zoom_level_x);
							x_end   += (xmax - xmin) / get_width() / pow(1.2, zoom_level_x);
						}
						picviz_view->square_area.set_start_x(x_start);
						picviz_view->square_area.set_end_x(x_end);

						picviz_view->square_area.set_dirty();
						set_update_line_dirty();
						update_selection();
					}
					break;
			case GLUT_KEY_DOWN:
						{
							float y_start, y_end, y_range, y_middle;
							y_start = picviz_view->square_area.get_start_y();
							y_end = picviz_view->square_area.get_end_y();
							y_range = std::fabs(y_end - y_start);
							if (glutGetModifiers() & GLUT_ACTIVE_SHIFT) { // zoom out the selection in the y direction
								y_middle = (y_start + y_end) / 2.0f;
								y_start = y_middle + (y_start - y_middle) / 1.2f;
								y_end = y_middle + (y_end - y_middle) / 1.2f;
							} else if (glutGetModifiers() & GLUT_ACTIVE_CTRL) { // Move the selection down
								y_start -= y_range;
								y_end -= y_range;
							} else { // Move the selection down by one pixel
								y_start -= (ymax - ymin) / get_height() / pow(1.2, zoom_level_y);
								y_end   -= (ymax - ymin) / get_height() / pow(1.2, zoom_level_y);
							}
							picviz_view->square_area.set_start_y(y_start);
							picviz_view->square_area.set_end_y(y_end);

							picviz_view->square_area.set_dirty();
							set_update_line_dirty();
							update_selection();
						}
					break;
			case GLUT_KEY_UP:
						{
							float y_start, y_end, y_range, y_middle;
							y_start = picviz_view->square_area.get_start_y();
							y_end = picviz_view->square_area.get_end_y();
							y_range = std::fabs(y_end - y_start);
							if (glutGetModifiers() & GLUT_ACTIVE_SHIFT) { // zoom in the selection in the y direction
								y_middle = (y_start + y_end) / 2.0f;
								y_start = y_middle + (y_start - y_middle) * 1.2f;
								y_end = y_middle + (y_end - y_middle) * 1.2f;
							} else if (glutGetModifiers() & GLUT_ACTIVE_CTRL) { // Move the selection up
								y_start += y_range;
								y_end += y_range;
							} else { // Move the selection up by one pixel
								y_start += (ymax - ymin) / get_height() / pow(1.2, zoom_level_y);
								y_end   += (ymax - ymin) / get_height() / pow(1.2, zoom_level_y);
							}
							picviz_view->square_area.set_start_y(y_start);
							picviz_view->square_area.set_end_y(y_end);

							picviz_view->square_area.set_dirty();
							set_update_line_dirty();
							update_selection();
						}
					break;
			case GLUT_KEY_DELETE:
					if (!state_machine->is_axes_mode()) {
						break;
					}

					/* We decide to leave at least two axes... */
					if (picviz_view->get_axes_count() <= 2 ) {
						break;
					}
					picviz_view->axes_combination.remove_axis(picviz_view->active_axis);
					/* We check if we have just removed the rightmost axis */
					if (picviz_view->axes_combination.get_axes_count() == picviz_view->active_axis ) {
						picviz_view->active_axis -= 1;
					}

					change_axes_count();
					update_all();
					message.function = PVGL_COM_FUNCTION_REFRESH_LISTING; // WITH_HORIZONTAL_HEADER?
					message.pv_view = picviz_view;
					pv_com->post_message_to_qt(message);
					break;
			case GLUT_KEY_F1:
					if (glutGetModifiers() & GLUT_ACTIVE_SHIFT) {
						max_lines_per_redraw += 1000;
					} else {
						max_lines_per_redraw -= 1000;
						if (max_lines_per_redraw < 1000) {
							max_lines_per_redraw = 1000;
						}
					}
					break;
		}
}

/******************************************************************************
 *
 * PVGL::PVView::mouse_wheel
 *
 *****************************************************************************/
void PVGL::PVView::mouse_wheel(int delta_zoom_level, int x, int y)
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);
	int old_zoom_level_x, old_zoom_level_y;

	/* We need to refresh the pixel dimension of the view */
	int MX = std::max(1, width);
	int MY = std::max(1, height);

	if (!picviz_view) { // Sanity check
		return;
	}
	if (!picviz_view->is_consistent()) {
		return;
	}
	/* We store the old zoom_levels */
	old_zoom_level_x = zoom_level_x;
	old_zoom_level_y = zoom_level_y;

	/* We set the new zoom_levels */
	switch (glutGetModifiers()) {
		case GLUT_ACTIVE_ALT:
				zoom_level_x += delta_zoom_level;
				break;

		case GLUT_ACTIVE_SHIFT:
				zoom_level_y += delta_zoom_level;
				break;

		case 0:
				zoom_level_x += delta_zoom_level;
				zoom_level_y += delta_zoom_level;
				break;
	}

	/* We teste whether we are zooming IN or OUT */
	if (delta_zoom_level > 0) {
		// We are zooming IN
		// We compute the new position of the translation-shift of the view:
		translation.x += (xmax - xmin) * (float(x) / MX - 0.5) * (1.0 / pow(1.2, zoom_level_x) - 1.0 / pow(1.2, old_zoom_level_x));
		translation.y += (ymin - ymax) * (float(y) / MY - 0.5) * (1.0 / pow(1.2, zoom_level_y) - 1.0 / pow(1.2, old_zoom_level_y));
	}
	lines.reset_offset();
	set_dirty();
}

/******************************************************************************
 *
 * PVGL::PVView::mouse_down
 *
 *****************************************************************************/
void PVGL::PVView::mouse_down(int button, int x, int y, int modifiers)
{
	vec2 plotted_mouse;
	Picviz::PVStateMachine *state_machine;

	PVLOG_INFO("PVGL::PVView::%s\n", __FUNCTION__);

	if (!picviz_view) { // Sanity check
		return;
	}
	if (!picviz_view->is_consistent()) {
		return;
	}
	state_machine = picviz_view->state_machine;
	if (map.mouse_down(x, y)) {
		return;
	}
	if (top_bar->is_visible() && event_line->mouse_down(button, x, y, modifiers)) {
		return;
	}
	/* We test whether AXES_MODE is active or not */
	if (state_machine->is_axes_mode() && button != 2) {
		/* We are in AXES_MODE */
		// Compute the x position of the mouse click in the plotted coordinates.
		plotted_mouse = screen_to_plotted(vec2(x, y));
		// set the active_axis according to the position of the click.
		picviz_view->set_active_axis_closest_to_position(plotted_mouse.x);
		update_axes();
	} else {
		/* We are NOT in AXES_MODE */
		/* We test if it is a RightButton click */
		if (button == 2) {
			/* We activate GRAB_MODE */
			state_machine->set_grabbed(true);
			lines.reset_offset();
		}
		/* We test if we are in GRAB mode */
		if (state_machine->is_grabbed()) {
			/* We are in GRAB mode */
			last_mouse_press_position_x = x;
			last_mouse_press_position_y = y;

			/* We are in SELECTION mode */
		} else {
			/* We start by clearing the selection in the listing */
			PVGL::PVMessage message;
			message.function = PVGL_COM_FUNCTION_CLEAR_SELECTION;
			message.pv_view = picviz_view;
			pv_com->post_message_to_qt(message);

			/* We might need to commit volatile_selection with floating_selection, depending on the previous Square_area_mode */
			switch (state_machine->get_square_area_mode()) {
				case Picviz::PVStateMachine::AREA_MODE_ADD_VOLATILE:
						picviz_view->floating_selection |= picviz_view->volatile_selection;
//						picviz_view->floating_selection.AB2A_or(picviz_view->volatile_selection);
						break;

				case Picviz::PVStateMachine::AREA_MODE_INTERSECT_VOLATILE:
						picviz_view->floating_selection &= picviz_view->volatile_selection;
//						picviz_view->floating_selection.AB2A_and(picviz_view->volatile_selection);
						break;

				case Picviz::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE:
						picviz_view->floating_selection = picviz_view->volatile_selection;
//						picviz_view->volatile_selection.A2B_copy(picviz_view->floating_selection);
						break;

				case Picviz::PVStateMachine::AREA_MODE_SUBSTRACT_VOLATILE:
						picviz_view->floating_selection -= picviz_view->volatile_selection;
//						picviz_view->floating_selection.AB2A_substraction(picviz_view->volatile_selection);
						break;

				case Picviz::PVStateMachine::AREA_MODE_OFF:
						;
			}

			// Compute the x and y position of the mouse click in the plotted coordinates.
			plotted_mouse = screen_to_plotted(vec2(x, y));

			// Store the position of this initial mouse press in the plotted coordinates system.
			picviz_view->square_area.set_start_x(plotted_mouse.x);
			picviz_view->square_area.set_start_y(plotted_mouse.y);
			// And reset the end_x/end_y value.
			picviz_view->square_area.set_end_x(plotted_mouse.x);
			picviz_view->square_area.set_end_y(plotted_mouse.y);

			/* We set AREA_MODE_* according to the modifiers */
			switch (modifiers) {
				/* INTERSECT */
				case (GLUT_ACTIVE_SHIFT | GLUT_ACTIVE_CTRL):
						state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_INTERSECT_VOLATILE);
						picviz_view->volatile_selection.select_none();
						break;

						/* SUBSTRACT */
				case GLUT_ACTIVE_CTRL:
						state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_SUBSTRACT_VOLATILE);
						picviz_view->volatile_selection.select_none();
						break;

						/* ADD */
				case GLUT_ACTIVE_SHIFT:
						state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_ADD_VOLATILE);
						picviz_view->volatile_selection.select_none();
						break;

						/* SET */
				default:
						state_machine->set_square_area_mode(Picviz::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE);
						picviz_view->volatile_selection.select_none();
						picviz_view->floating_selection.select_none();
						break;
			}
		}
	}
}

/******************************************************************************
 *
 * PVGL::PVView::mouse_move
 *
 *****************************************************************************/
bool PVGL::PVView::mouse_move(int x, int y, int modifiers)
{
	Picviz::PVStateMachine   *state_machine;

	vec2 plotted_mouse;

	if (!picviz_view) { // The view isn't finished to be read and parsed
		return false;
	}
	if (!picviz_view->is_consistent()) {
		return false;
	}
	state_machine = picviz_view->state_machine;
	if (map.mouse_move(x, y)) {
		return false;
	}
	if (top_bar->is_visible() && event_line->mouse_move(x, y, modifiers)) {
		return false;
	}
	/* We test if we are in GRAB mode */
	if (state_machine->is_grabbed()) {
		// We are in GRAB mode.
		update_displacement(vec2(x, y));
	} else if (state_machine->is_axes_mode()) {
		// We are in AXES_MODE.
		// Compute the position of the mouse click in the plotted coordinates.
		plotted_mouse = screen_to_plotted(vec2(x, y));
		// Move the active axis.
		if (!picviz_view->move_active_axis_closest_to_position(plotted_mouse.x)) {
			/* the axis really moved... */
			update_all();
		}
		return false;
	} else { // We are in SELECTION mode.
		// Store the position of this last square_area position in the plotted coordinates system.
		plotted_mouse = screen_to_plotted(vec2(x, y));
		picviz_view->square_area.set_end_x(plotted_mouse.x);
		picviz_view->square_area.set_end_y(plotted_mouse.y);

		picviz_view->square_area.set_dirty();
		set_update_line_dirty();
		update_selection_except_listing();
		return true; // Tell all the windows to redraw.
	}
	return false;
}

/******************************************************************************
 *
 * PVGL::PVView::mouse_up
 *
 *****************************************************************************/
bool PVGL::PVView::mouse_up(int button, int x, int y, int modifiers)
{
	Picviz::PVStateMachine *state_machine;
        
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);

	if (!picviz_view) {
		return false;
	}
	if (!picviz_view->is_consistent()) {
		return false;
	}
	state_machine = picviz_view->state_machine;

	if (map.mouse_up(x, y)) {
		return true;
	}

	if (top_bar->is_visible() && event_line->mouse_up(button, x, y, modifiers)) {
		return true;
	}
	/* We test if we are NOT in GRAB mode */
	if (!state_machine->is_grabbed()) { // We are in SELECTION mode.
		/* AG: if the square area is empty (that is the user has just click and release the mouse
		 * with no mouvements), we need to restore the previous selection. */
		if (picviz_view->square_area.is_empty()) {
			/* Get the selection back from real_output_selection from the picviz view */
			picviz_view->volatile_selection = picviz_view->get_real_output_selection();
		}
		/* We update the view */
		glutPostRedisplay ();
		/* We update the listing */
		update_listing();
	}

	/* We test if it is a RightButton click */
	if (button == 2) {
		/* We deactivate GRAB_MODE */
		state_machine->set_grabbed(false);
		get_lines().reset_offset();
		get_lines().set_main_fbo_dirty();
		//map.set_lines_fbo_dirty();
		get_lines().set_zombie_fbo_dirty();
		//map.set_zombie_fbo_dirty();
	}
	return true;
}

/******************************************************************************
 *
 * PVGL::PVView::passive_motion
 *
 *****************************************************************************/
bool PVGL::PVView::passive_motion(int x, int y, int modifiers)
{
	PVLOG_HEAVYDEBUG("PVGL::PVView::%s\n", __FUNCTION__);

	if (top_bar->is_visible() && event_line->passive_motion(x, y, modifiers)) {
		return true;
	}
	return false;
}

/******************************************************************************
 *
 * PVGL::PVView::set_size
 *
 *****************************************************************************/
void PVGL::PVView::set_size(int w, int h)
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);

	width = w;
	height = h;
	size_dirty = true;
	top_bar->set_size(width, 60);
	event_line->set_size(width/3 +1, 60);
	glViewport(0, 0, width, height);
}

/******************************************************************************
 *
 * PVGL::PVView::update_set_size
 *
 *****************************************************************************/
void PVGL::PVView::update_set_size()
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);

	size_dirty = false;
	lines.set_size(width, height);
	map.set_size(width, height);
}

/******************************************************************************
 *
 * PVGL::PVView::is_set_size_dirty
 *
 *****************************************************************************/
bool PVGL::PVView::is_set_size_dirty() const
{
	PVLOG_HEAVYDEBUG("PVGL::PVView::%s\n", __FUNCTION__);

	return size_dirty;
}

/******************************************************************************
 *
 * PVGL::PVView::reinit_picviz_view
 *
 *****************************************************************************/
void PVGL::PVView::reinit_picviz_view()
{
	PVDrawable::init(picviz_view);
	lines.reinit_picviz_view();
	update_all();
}

/******************************************************************************
 *
 * PVGL::PVView::update_listing
 *
 *****************************************************************************/
void PVGL::PVView::update_listing(void)
{
                PVLOG_INFO("PVGL::PVView::update_listing\n");
	PVGL::PVMessage message;

	message.function = PVGL_COM_FUNCTION_CLEAR_SELECTION;
	message.pv_view = get_libview();
	pv_com->post_message_to_qt(message);

	message.function = PVGL_COM_FUNCTION_SELECTION_CHANGED;
	message.pv_view = get_libview();
	pv_com->post_message_to_qt(message);

	message.function = PVGL_COM_FUNCTION_UPDATE_OTHER_SELECTIONS;
	message.pv_view = get_libview();
	message.int_1 = get_window_id();
	pv_com->post_message_to_gl(message);
}

/******************************************************************************
 *
 * PVGL::PVView::update_selection
 *
 *****************************************************************************/
void PVGL::PVView::update_selection(void)
{
	update_listing();
	selection_square.update_arrays();
}
