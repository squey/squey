//! \file PVView.cpp
//! $Id$
//! Copyright (C) Sebastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>

#define GLEW_STATIC 1
#include <GL/glew.h>
#include <GL/freeglut.h>

// #include <pvkernel/core/PVUtils.h>

#include <picviz/PVView.h>

#include <pvsdk/PVMessenger.h>

#include <pvgl/general.h>
#include <pvgl/PVConfig.h>
#include <pvgl/PVUtils.h>
#include <pvgl/PVHBox.h>
#include <pvgl/PVVBox.h>
#include <pvgl/PVEventLine.h>
#include <pvgl/PVWTK.h>

#include <pvgl/views/PVParallel.h>

/******************************************************************************
 *
 * PVGL::PVView::PVView
 *
 *****************************************************************************/
PVGL::PVView::PVView(int win_id, PVSDK::PVMessenger *message) : PVGL::PVDrawable(win_id, message),
		selection_square(this),
		widget_manager(),
		lines(this),
		// Disabled for now map(this, &widget_manager, &lines, width, height),
		axes(this)
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);
	show_axes = true;
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

	label_fps = new PVLabel(&widget_manager, "FPS: 0");
	label_fps->set_shadow(true);
	label_fps->hide();

	PVHBox *hbox = new PVHBox(&widget_manager);
	top_bar->add(hbox, 0, 0, -1, -1);
	PVVBox *vbox = new PVVBox(&widget_manager);
	hbox->pack_start(vbox);
	vbox->pack_start(label_axis_mode);
	vbox->pack_start(label_nb_lines);
	vbox->pack_start(label_fps);
	event_line = new PVEventLine(&widget_manager, this, message);
	hbox->pack_start(event_line, false);

	label_lpr = new PVLabel(&widget_manager, "LPR: 25000"); // FIXME this should use a 'const' ?
	label_lpr->set_color(ubvec4(0, 0, 0, 255));
	vbox->pack_start(label_lpr);
}

/******************************************************************************
 *
 * PVGL::PVView::~PVView
 *
 *****************************************************************************/
PVGL::PVView::~PVView()
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);

	delete top_bar;
	delete label_nb_lines;
	delete label_axis_mode;
	delete event_line;
	delete label_lpr;
	delete label_fps;
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
	xmax = ((float) picviz_view->get_axes_count()) - 0.5f;
	ymax = 1.3f;

	// We initialize the 5 fundamental objects of the PVView
	axes.init(view);
	selection_square.init(view);
	lines.init(view);
	// map.init(view);
	event_line->set_view(view);

	/* We fill the arrays for later use */
	lines.update_arrays_positions();
	lines.update_arrays_colors();
	lines.update_arrays_z();
	lines.update_arrays_zombies();
	lines.update_arrays_selection();
	// map.update_arrays_positions();
	// map.update_arrays_colors();
	// map.update_arrays_z();
	// map.update_arrays_zombies();
	// map.update_arrays_selection();

	axes.update_arrays();
	axes.update_arrays_bg();
	selection_square.update_arrays();
	
	update_label_lines_selected_eventline();
	update_label_lpr();
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
		int current_time = (PVGL::wtk_time_ms_elapsed_since_init() / 250) % 4;
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
		glOrtho(0, width, height, 0, -1,1);

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
	if (show_axes) {
		axes.draw_names();
	}
	// map.draw();

	glDisable(GL_DEPTH_TEST);

	glLoadIdentity ();
	glOrtho(0,width, height,0, -1,1);

	// Draw number of selected lines
	// if (PVCore::PVUtils::isCapsLockActivated()) {
	// 	PVLOG_INFO("CAPS LOCK is Activated\n");
	// }
		// {
		// 	if (state_machine->is_caps_lock_activated()) {
		// 		PVLOG_INFO("Caps lock activated!\n");
		// 		// std::stringstream ss;
		// 		// // ss << "LPR: " << get_max_lines_per_redraw();
		// 		// ss << "Caps lock activated!";
		// 		// label_lpr->set_text(ss.str());
		// 	}
		// }
	{
		if (label_fps->is_visible()) {
			std::stringstream ss;
			ss << "FPS: " << current_fps;
			label_fps->set_text(ss.str());
		}
	}

	if (picviz_view->state_machine->is_axes_mode()) {
		label_axis_mode->show();
	} else {
		label_axis_mode->hide();
	}

	top_bar->draw();

	if (label_fps->is_visible()) {
		compute_fps();
	}
}

/******************************************************************************
 *
 * PVGL::PVView::get_center_position
 *
 *****************************************************************************/
vec2 PVGL::PVView::get_center_position()
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);
	
	return vec2((xmin + xmax) / 2.0,
	            (ymin + ymax) / 2.0);
}

/******************************************************************************
 *
 * PVGL::PVView::get_leftmost_visible_axis
 *
 *****************************************************************************/
PVCol PVGL::PVView::get_leftmost_visible_axis()
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);

	vec2 topleft_plotted = screen_to_plotted(vec2(0, 0));
	
	return picviz_view->get_active_axis_closest_to_position(topleft_plotted.x);
}

/******************************************************************************
 *
 * PVGL::PVView::get_most_centered_visible_axis
 *
 *****************************************************************************/
PVCol PVGL::PVView::get_most_centered_visible_axis()
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);

	vec2 centered_plotted = get_center_position();
	
	return picviz_view->get_active_axis_closest_to_position(centered_plotted.x);
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
 * PVGL::PVView::keyboard
 *
 *****************************************************************************/
void PVGL::PVView::keyboard(unsigned char key, int, int)
{
	Picviz::PVStateMachine *state_machine;
	PVSDK::PVMessage       message;

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
				picviz_view->select_all_nonzb_lines();
				/* We refresh the listing */
				message.function = PVSDK_MESSENGER_FUNCTION_REFRESH_LISTING;
				message.pv_view = picviz_view;
				pv_message->post_message_to_qt(message);
				update_selections();
				break;
		case 'c': case 'C': // Choose a color.
				message.function = PVSDK_MESSENGER_FUNCTION_SET_COLOR;
				message.pv_view = picviz_view;
				pv_message->post_message_to_qt(message);
				break;
		case 'f':
				if (is_fullscreen()) {
					PVGL::wtk_window_resize(old_width, old_height);
					toggle_fullscreen(false);
				} else {
					old_width = get_width();
					old_height= get_height();
					PVGL::wtk_window_fullscreen();
					toggle_fullscreen(true);
				}
				break;
		case 'F':
				if (label_fps->is_visible()) {
					label_fps->hide();
				}
				else {
					label_fps->show();
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
					message.function = PVSDK_MESSENGER_FUNCTION_COMMIT_SELECTION_IN_NEW_LAYER;
					message.pv_view = picviz_view;
					pv_message->post_message_to_qt(message);
				} else {
					message.function = PVSDK_MESSENGER_FUNCTION_COMMIT_SELECTION_IN_CURRENT_LAYER;
					message.pv_view = picviz_view;
					pv_message->post_message_to_qt(message);
				}
				break;
		case 'i': case 'I': // Select all
				picviz_view->select_inv_lines();
				/* We refresh the listing */
				message.function = PVSDK_MESSENGER_FUNCTION_REFRESH_LISTING;
				message.pv_view = picviz_view;
				pv_message->post_message_to_qt(message);
				update_selections();
				break;
		case 'm':
				// toggle_map();
				break;
		case 'r':
				message.function = PVSDK_MESSENGER_FUNCTION_REPORT_CHOOSE_FILENAME;
				message.pv_view = picviz_view;
				message.int_1 = glutGetWindow();
				message.int_2 = glutGetModifiers();
				pv_message->post_message_to_qt(message);
				break;
		case 's':
				message.function = PVSDK_MESSENGER_FUNCTION_SCREENSHOT_CHOOSE_FILENAME;
				message.pv_view = picviz_view;
				message.int_1 = glutGetWindow();
				message.int_2 = glutGetModifiers();
				pv_message->post_message_to_qt(message);
				break;
		case 'u': case 'U':
				if (glutGetModifiers() & GLUT_ACTIVE_SHIFT) {
					/* We toggle*/
					state_machine->toggle_listing_unselected_visibility();
					/* We refresh the listing */
					message.function = PVSDK_MESSENGER_FUNCTION_REFRESH_LISTING;
					message.pv_view = picviz_view;
					pv_message->post_message_to_qt(message);
				} else	{
					/* We toggle */
					state_machine->toggle_gl_unselected_visibility();
					/* We refresh the view */
					//picviz_view_process_visibility(pv_view);
					get_lines().set_main_fbo_dirty();
					//map.set_main_fbo_dirty();
				}
				break;
		case 'w': case 'W':
				show_axes = !show_axes;
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
					state_machine->toggle_listing_zombie_visibility();
					/* We refresh the listing */
					message.function = PVSDK_MESSENGER_FUNCTION_REFRESH_LISTING;
					message.pv_view = picviz_view;
					pv_message->post_message_to_qt(message);
				} else {
					/* We toggle*/
					state_machine->toggle_gl_zombie_visibility();
					/* We refresh the view */
					get_lines().set_main_fbo_dirty();
					// map.set_main_fbo_dirty();
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
				message.function = PVSDK_MESSENGER_FUNCTION_REFRESH_LISTING; // WITH_HORIZONTAL_HEADER?
				message.pv_view = picviz_view;
				pv_message->post_message_to_qt(message);
				break;
	}
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

	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);

	if (!picviz_view) { // Sanity check
		return;
	}
	if (!picviz_view->is_consistent()) {
		return;
	}
	state_machine = picviz_view->state_machine;
/*	if (map.mouse_down(x, y)) {
		return;
	}*/
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
			PVSDK::PVMessage message;
			message.function = PVSDK_MESSENGER_FUNCTION_CLEAR_SELECTION;
			message.pv_view = picviz_view;
			pv_message->post_message_to_qt(message);

			/* We might need to commit volatile_selection with floating_selection, depending on the previous Square_area_mode */
			picviz_view->commit_volatile_in_floating_selection();

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

	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);

	if (!picviz_view) { // The view isn't finished to be read and parsed
		return false;
	}
	if (!picviz_view->is_consistent()) {
		return false;
	}
	state_machine = picviz_view->state_machine;
/*	if (map.mouse_move(x, y)) {
		return false;
	}*/
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

/*	if (map.mouse_up(x, y)) {
		return true;
	}*/

	if (top_bar->is_visible() && event_line->mouse_up(button, x, y, modifiers)) {
		return true;
	}
	/* We test if we are in GRAB mode */
	if (state_machine->is_grabbed()) {
		PVCol first_axis = get_most_centered_visible_axis();
// 		PVCol first_axis = get_leftmost_visible_axis();
		/* Send a message to Qt */
		PVSDK::PVMessage message;
		message.function = PVSDK_MESSENGER_FUNCTION_MAY_ENSURE_AXIS_VIEWABLE;
		message.pv_view = picviz_view;
		message.int_1 = (int) first_axis;
		pv_message->post_message_to_qt(message);
	}
	else { // We are in SELECTION mode.
		/* AG: if the square area is empty (that is the user has just click and release the mouse
		 * with no mouvements), we need to restore the previous selection. */
		if (picviz_view->square_area.is_empty()) {
			/* Get the selection back from real_output_selection from the picviz view */
			picviz_view->volatile_selection = picviz_view->get_real_output_selection();
		}
		/* We update the view */
		PVGL::wtk_window_need_redisplay();
		/* Trying to solve a bug */
		if (picviz_view->square_area.is_dirty()) {
			PVLOG_DEBUG("PVGL::PVView::%s : picviz_view->process_from_selection\n", __FUNCTION__);
			//picviz_view->gl_cdlocker.lock();
			picviz_view->selection_A2B_select_with_square_area(picviz_view->layer_stack_output_layer.get_selection(), picviz_view->volatile_selection);
			picviz_view->process_from_selection();
			picviz_view->square_area.set_clean();
			//picviz_view->gl_call_locker.unlock();
			PVLOG_DEBUG("PVGL::PVView::%s : pv_view->update_lines\n", __FUNCTION__);
			get_lines().update_arrays_selection();
			//	get_map().update_arrays_selection();
			update_lines();
		}

		
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
 * PVGL::PVView::mouse_wheel
 *
 *****************************************************************************/
void PVGL::PVView::mouse_wheel(int delta_zoom_level, int x, int y)
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);
	int old_zoom_level_x, old_zoom_level_y;

	/* We need to refresh the pixel dimension of the view */
	int MX = picviz_max(1, width);
	int MY = picviz_max(1, height);

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
 * PVGL::PVView::passive_motion
 *
 *****************************************************************************/
bool PVGL::PVView::passive_motion(int x, int y, int modifiers)
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);

	if (top_bar->is_visible() && event_line->passive_motion(x, y, modifiers)) {
		return true;
	}
	return false;
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
	xmax = (float)picviz_view->get_axes_count() - 0.5f;
	ymax = 1.3f;
	last_mouse_press_position_x = last_mouse_press_position_y = 0;
	translation = vec2 (0.0f, 0.0f);
	zoom_level_x = 0;
	zoom_level_y = 0;
	set_dirty();
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
 * PVGL::PVView::special_keys
 *
 *****************************************************************************/
void PVGL::PVView::special_keys(int key, int, int)
{
	Picviz::PVStateMachine *state_machine;
	PVSDK::PVMessage       message;

	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);

	if (!picviz_view) { // The view isn't finished to be read and parsed
		return;
	}
	if (!picviz_view->is_consistent()) {
		return;
	}
	state_machine = picviz_view->state_machine;

	// if (glutGetModifiers() & GLUT_ACTIVE_SHIFT) {
	// 	state_machine->toggle_caps_lock_activated();
	// }

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
						update_axes_combination();
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
						update_axes_combination();
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
					message.function = PVSDK_MESSENGER_FUNCTION_REFRESH_LISTING; // WITH_HORIZONTAL_HEADER?
					message.pv_view = picviz_view;
					pv_message->post_message_to_qt(message);
					update_axes_combination();
					break;
			case GLUT_KEY_F1:
					if (glutGetModifiers() & GLUT_ACTIVE_SHIFT) {
						max_lines_per_redraw += 1000;
						// We should talk PVLines about this!
						lines.update_lpr();
					} else {
						max_lines_per_redraw -= 1000;
						if (max_lines_per_redraw < 1000) {
							max_lines_per_redraw = 1000;
						}
					}
					update_label_lpr();
					break;
		}
}

/******************************************************************************
 *
 * PVGL::PVView::toggle_map
 *
 *****************************************************************************/
#if 0
void PVGL::PVView::toggle_map()
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);

	//map.toggle_map();
}
#endif

/******************************************************************************
 *
 * PVGL::PVView::reinit_picviz_view
 *
 *****************************************************************************/
void PVGL::PVView::reinit_picviz_view()
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);

	PVDrawable::init(picviz_view);
	lines.reinit_picviz_view();
	update_all();
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
	lines.update_lpr();
	lines.update_arrays_colors();
	lines.update_arrays_z();
	lines.update_arrays_selection();
	lines.update_arrays_zombies();
	lines.update_arrays_positions();
	//map.update_arrays_colors();
	//map.update_arrays_z();
	//map.update_arrays_selection();
	//map.update_arrays_zombies();
	//map.update_arrays_positions();
	selection_square.update_arrays();
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
	//map.update_arrays_selection();
	//map.update_arrays_zombies();
}


#if 0
/******************************************************************************
 *
 * PVGL::PVView::update_axes
 *
 *****************************************************************************/
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

/******************************************************************************
 *
 * PVGL::PVView::update_axes_combination
 *
 *****************************************************************************/
void PVGL::PVView::update_axes_combination(void)
{
	PVSDK::PVMessage message;

	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);
	
	message.function = PVSDK_MESSENGER_FUNCTION_UPDATE_AXES_COMBINATION;
	message.pv_view = get_libview();
	message.int_1 = get_window_id();
	pv_message->post_message_to_qt(message);
}

/******************************************************************************
 *
 * PVGL::PVView::update_colors
 *
 *****************************************************************************/
void PVGL::PVView::update_colors()
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);

	if (!picviz_view) { // Sanity check
		return;
	}
	if (!picviz_view->is_consistent()) {
		return;
	}

	get_lines().update_arrays_colors();
	//get_map().update_arrays_colors();
}

/******************************************************************************
 *
 * PVGL::PVView::update_current_layer
 *
 *****************************************************************************/
void PVGL::PVView::update_current_layer()
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);

	if (!picviz_view) { // Sanity check
		return;
	}
	if (!picviz_view->is_consistent()) {
		return;
	}
	axes.update_current_layer(picviz_view->get_layer_stack().get_selected_layer());
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
 * PVGL::PVView::update_listing
 *
 *****************************************************************************/
void PVGL::PVView::update_listing(void)
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);

	PVSDK::PVMessage message;

	message.function = PVSDK_MESSENGER_FUNCTION_CLEAR_SELECTION;
	message.pv_view = get_libview();
	pv_message->post_message_to_qt(message);

	message.function = PVSDK_MESSENGER_FUNCTION_SELECTION_CHANGED;
	message.pv_view = get_libview();
	pv_message->post_message_to_qt(message);

	message.function = PVSDK_MESSENGER_FUNCTION_UPDATE_OTHER_SELECTIONS;
	message.pv_view = get_libview();
	message.int_1 = get_window_id();
	pv_message->post_message_to_gl(message);
}

/******************************************************************************
 *
 * PVGL::PVView::update_positions
 *
 *****************************************************************************/
void PVGL::PVView::update_positions()
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);

	if (!picviz_view) { // Sanity check
		return;
	}
	if (!picviz_view->is_consistent()) {
		return;
	}

	get_lines().update_arrays_positions();
	//get_map().update_arrays_positions();
}

/******************************************************************************
 *
 * PVGL::PVView::update_selection
 *
 *****************************************************************************/
void PVGL::PVView::update_selection(void)
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);

	update_listing();
	selection_square.update_arrays();
}

/******************************************************************************
 *
 * PVGL::PVView::update_selections
 *
 *****************************************************************************/
void PVGL::PVView::update_selections()
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);

	if (!picviz_view) { // Sanity check
		return;
	}
	if (!picviz_view->is_consistent()) {
		return;
	}

	get_lines().update_arrays_selection();
	//get_map().update_arrays_selection();
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
	//map.set_size(width, height);
}

/******************************************************************************
 *
 * PVGL::PVView::update_text_lines_selected_eventline
 *
 *****************************************************************************/
void PVGL::PVView::update_label_lines_selected_eventline()
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);

	float selected_float = ((float)picviz_view->get_number_of_selected_lines() / (float)picviz_view->eventline.get_row_count());
	int selected_percent = (int)(selected_float*100);

	std::stringstream ss;
	ss << "Events selected: " << picviz_view->get_number_of_selected_lines() << " (" << selected_percent << "%)" <<
		" / " << picviz_view->eventline.get_current_index() - picviz_view->eventline.get_first_index() <<
		" / " << picviz_view->eventline.get_row_count();
	label_nb_lines->set_text(ss.str());
}

/******************************************************************************
 *
 * PVGL::PVView::update_lpr
 *
 *****************************************************************************/
void PVGL::PVView::update_label_lpr()
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);

	std::stringstream ss;
	ss << "LPR: " << get_max_lines_per_redraw();
	label_lpr->set_text(ss.str());
}

/******************************************************************************
 *
 * PVGL::PVView::update_z
 *
 *****************************************************************************/
void PVGL::PVView::update_z()
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);

	if (!picviz_view) { // Sanity check
		return;
	}
	if (!picviz_view->is_consistent()) {
		return;
	}

	get_lines().update_arrays_z();
	//get_map().update_arrays_z();
}

/******************************************************************************
 *
 * PVGL::PVView::update_zombies
 *
 *****************************************************************************/
void PVGL::PVView::update_zombies()
{
	PVLOG_DEBUG("PVGL::PVView::%s\n", __FUNCTION__);

	if (!picviz_view) { // Sanity check
		return;
	}
	if (!picviz_view->is_consistent()) {
		return;
	}

	get_lines().update_arrays_zombies();
	//get_map().update_arrays_zombies();
}

