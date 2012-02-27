//! \file PVMain.cpp
//! $Id: PVMain.cpp 3191 2011-06-23 13:47:36Z stricaud $
//! Copyright (C) SÃÂÃÂ©bastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <iostream>
#include <list>
#include <map>
#include <sstream>
#include <string>
#include <cmath>
#include <algorithm>

#ifdef WIN32 // sleep()
#include <windows.h>
#else
#include <unistd.h>
#endif // WIN32

#include <QtCore>
#include <QImage>

#define GLEW_STATIC 1
#include <GL/glew.h>

#include <picviz/general.h>
#include <pvgl/general.h>

#include <picviz/PVSelection.h>
#include <picviz/PVView.h>

#include <pvsdk/PVMessenger.h>

#include <pvgl/PVFonts.h>
#include <pvgl/PVUtils.h>
#include <pvgl/PVConfig.h>
#include <pvgl/views/PVParallel.h>
#include <pvgl/views/PVScatter.h>

#include <pvgl/PVMain.h>

#include <pvgl/PVWTK.h>



static std::list<PVGL::PVDrawable*> all_drawables;
// static PVGL::PVCom *pvgl_com = 0;
static PVSDK::PVMessenger *pvsdk_messenger = 0;

static PVGL::PVView *transient_view = 0;
static int last_key_pressed_time = 0;
static int last_reshape_time_time = 0;

static bool _should_stop = false;

PVGL::PVIdleManager idle_manager;

int view_resize_update_timer = 0;
int window_timer_refresh = 0;
int mouse_motion_refresh_timer = 0;

/******************************************************************************
 *
 * PVGL::PVMain::idle_callback
 *
 *****************************************************************************/
void PVGL::PVMain::idle_callback(void)
{
	PVLOG_HEAVYDEBUG("PVGL::PVMain::%s\n", __FUNCTION__);

	idle_manager.callback();
}

/******************************************************************************
 *
 * PVGL::PVMain::get_drawable_from_id
 *
 *****************************************************************************/
PVGL::PVDrawable *PVGL::PVMain::get_drawable_from_id(int window_id)
{
	PVLOG_HEAVYDEBUG("PVGL::PVMain::%s\n", __FUNCTION__);

	for (std::list<PVGL::PVDrawable*>::iterator it = all_drawables.begin(); it != all_drawables.end(); ++it) {
		if ((*it)->get_window_id() == window_id) {
			return *it;
		}
	}
	if (transient_view && transient_view->get_window_id() == window_id) {
		return transient_view;
	}
	return 0;
}

/******************************************************************************
 *
 * PVGL::PVMain::display_callback
 *
 *****************************************************************************/
void PVGL::PVMain::display_callback()
{
	PVGL::PVDrawable *current_drawable;

	PVLOG_HEAVYDEBUG("PVGL::PVMain::%s\n", __FUNCTION__);

	current_drawable = PVGL::PVMain::get_drawable_from_id(wtk_get_current_window());

	if (current_drawable) {
		current_drawable->draw();
		PVGL::wtk_buffers_swap();
	}
}

/******************************************************************************
 *
 * PVGL::PVMain::keyboard_callback
 *
 *****************************************************************************/
void PVGL::PVMain::keyboard_callback(unsigned char key, int x, int y)
{
	PVGL::PVDrawable *current_drawable;

	PVLOG_DEBUG("PVGL::PVMain::%s\n", __FUNCTION__);

	last_key_pressed_time = PVGL::wtk_time_ms_elapsed_since_init();
	current_drawable = get_drawable_from_id(wtk_get_current_window());

	if (current_drawable) {
		current_drawable->keyboard(key, x, y);
		PVGL::wtk_window_need_redisplay();
	}
}

/******************************************************************************
 *
 * PVGL::PVMain::special_callback
 *
 *****************************************************************************/
void PVGL::PVMain::special_callback(int key, int x, int y)
{
	PVGL::PVDrawable *current_drawable;

	PVLOG_DEBUG("PVGL::PVMain::%s\n", __FUNCTION__);

	last_key_pressed_time = PVGL::wtk_time_ms_elapsed_since_init();
	current_drawable = get_drawable_from_id(wtk_get_current_window());
	if (current_drawable) {
		current_drawable->special_keys(key, x, y);
		PVGL::wtk_window_need_redisplay();
	}
}

/******************************************************************************
 *
 * PVGL::PVMain::mouse_wheel
 *
 *****************************************************************************/
void PVGL::PVMain::mouse_wheel(int delta_zoom_level, int x, int y)
{
	PVGL::PVDrawable *current_drawable;

	PVLOG_DEBUG("PVGL::PVMain::%s\n", __FUNCTION__);

	current_drawable = get_drawable_from_id(wtk_get_current_window());
	if (current_drawable) {
		current_drawable->mouse_wheel(delta_zoom_level, x, y);
	}
}

/******************************************************************************
 *
 * PVGL::PVMain::mouse_down
 *
 *****************************************************************************/
void PVGL::PVMain::mouse_down(int button, int x, int y)
{
        
	PVGL::PVDrawable *current_drawable;

	PVLOG_DEBUG("PVGL::PVMain::%s\n", __FUNCTION__);
        
        
        
	current_drawable = get_drawable_from_id(wtk_get_current_window());
	if (current_drawable) {
		current_drawable->mouse_down(button, x, y, PVGL::wtk_get_keyboard_modifiers());
	}
	// FIXME: Is this a non-sense?
        mouse_is_moving =true;
}

/******************************************************************************
 *
 * PVGL::PVMain::mouse_release
 *
 *****************************************************************************/
void PVGL::PVMain::mouse_release(int button, int x, int y)
{
	PVGL::PVDrawable *current_drawable;

	PVLOG_DEBUG("PVGL::PVMain::%s\n", __FUNCTION__);
        mouse_is_moving = false;
	current_drawable = get_drawable_from_id(wtk_get_current_window());
	if (current_drawable) {
		current_drawable->mouse_up(button, x, y, PVGL::wtk_get_keyboard_modifiers());
         }
}

/******************************************************************************
 *
 * PVGL::PVMain::button_callback
 *
 *****************************************************************************/
void PVGL::PVMain::button_callback(int button, int state, int x, int y)
{
	PVLOG_DEBUG("PVGL::PVMain::%s\n", __FUNCTION__);
        //moving_locker_mutex.lock();
	switch (state) {
		case GLUT_DOWN:
					{
						switch (button) {
							case 3: // mouse wheel up
									mouse_wheel (1, x, y);
									break;
							case 4: // mouse wheel down
									mouse_wheel (-1, x, y);
									break;
							default:
									mouse_down (button, x, y);
						}
					}
				break;
		case GLUT_UP:
				switch (button) {
					case 3: // mouse wheel up
							break;
					case 4: // mouse wheel down
							break;
					default:
							mouse_release (button, x, y);
				}
				break;
	}
        //moving_locker_mutex.unlock();
	PVGL::wtk_window_need_redisplay();
}

/******************************************************************************
 *
 * PVGL::PVMain::motion_callback
 *
 *****************************************************************************/
void PVGL::PVMain::motion_callback(int x, int y)
{
	PVGL::PVDrawable *current_drawable;

	PVLOG_HEAVYDEBUG("PVGL::PVMain::%s\n", __FUNCTION__);

	last_key_pressed_time = PVGL::wtk_time_ms_elapsed_since_init();

	current_drawable = get_drawable_from_id(wtk_get_current_window());
	if (current_drawable) {
		if (current_drawable->mouse_move(x, y, PVGL::wtk_get_keyboard_modifiers())) {
		}
		PVGL::wtk_window_need_redisplay();
	}
	//FIXME: sync code, ugly!
	// It only works with PVView(s)
// 	if (1) {
// 		for (std::list<PVGL::PVDrawable*>::iterator it = all_drawables.begin(); it != all_drawables.end(); ++it) {
// 			if ((*it)->get_libview() == current_drawable->get_libview() && (*it)->get_window_id() != current_drawable->get_window_id()) {
// 				(*it)->update_selection();
// 				(*it)->get_lines().update_arrays_selection();
// 				(*it)->set_update_line_dirty();
// 				PVGL::wtk_set_current_window((*it)->get_window_id());
// 				PVGL::wtk_window_need_redisplay();
// 			}
// 		}
// 	}
}

/******************************************************************************
 *
 * PVGL::PVMain::reshape_callback
 *
 *****************************************************************************/
void PVGL::PVMain::reshape_callback(int width, int height)
{
	PVGL::PVDrawable *current_drawable = get_drawable_from_id(wtk_get_current_window());

	PVLOG_DEBUG("PVGL::PVMain::%s\n", __FUNCTION__);

	last_reshape_time_time = PVGL::wtk_time_ms_elapsed_since_init();

	if (current_drawable) {
		current_drawable->set_size(width, height);
	}
	glViewport(0, 0, width, height);
	PVGL::wtk_window_need_redisplay();
}

/******************************************************************************
 *
 * PVGL::PVMain::close_callback
 *
 *****************************************************************************/
void PVGL::PVMain::close_callback(void)
{
	int window_id = wtk_get_current_window();
	PVLOG_DEBUG("PVGL::PVMain::%s\n", __FUNCTION__);

	// Let's remove the window from our pool.
	for (std::list<PVGL::PVDrawable*>::iterator it = all_drawables.begin(); it != all_drawables.end(); ++it) {
		if ((*it)->get_window_id() == window_id) {
			// Delete tasks for that drawable
			idle_manager.remove_tasks_for_drawable((*it));

			// Then tell Qt that this drawable is destroyed
			QString *name = new QString((*it)->get_name());
			PVSDK::PVMessage message;

			message.function = PVSDK_MESSENGER_FUNCTION_ONE_VIEW_DESTROYED;
			message.pv_view = (*it)->get_libview();
			if (!(*it)->get_libview()) {
				transient_view = 0;
			}
			message.pointer_1 = name;

			pvsdk_messenger->post_message_to_qt(message);

			// And delete its resources
			delete (*it);

			all_drawables.erase(it);
			break;
		}
	}
}

/******************************************************************************
 *
 * PVGL::PVMain::entry_callback
 *
 *****************************************************************************/
void PVGL::PVMain::entry_callback(int/* state*/)
{
#if 0
	PVLOG_DEBUG("PVGL::PVMain::%s\n", __FUNCTION__);

	if (state == GLUT_ENTERED) {
		//std::cerr << "Got in!" << std::endl;
	} else { // GLUT_LEFT
		//std::cerr << "Got out!" << std::endl;
	}
#endif
}

/******************************************************************************
 *
 * PVGL::PVMain::passive_motion_callback
 *
 *****************************************************************************/
void PVGL::PVMain::passive_motion_callback(int x, int y)
{
	PVGL::PVDrawable *current_drawable;

	PVLOG_HEAVYDEBUG("PVGL::PVMain::%s\n", __FUNCTION__);

	current_drawable = get_drawable_from_id(wtk_get_current_window());

	if (current_drawable) {
	current_drawable->passive_motion(x, y, PVGL::wtk_get_keyboard_modifiers());
	}
}

/******************************************************************************
 *
 * PVGL::PVMain::create_view
 *
 *****************************************************************************/
void PVGL::PVMain::create_view(QString *name)
{
	PVGL::PVView *current_view;
	int       window_id;
	int       index = 0;
	QString   base_name (*name);
	int default_width = pvconfig.value("pvgl/parallel_view_width", PVGL_VIEW_DEFAULT_WIDTH).toInt();
	int default_height = pvconfig.value("pvgl/parallel_view_height", PVGL_VIEW_DEFAULT_HEIGHT).toInt();

	PVLOG_DEBUG("PVGL::PVMain::%s\n", __FUNCTION__);

	for (std::list<PVGL::PVDrawable*>::iterator it = all_drawables.begin(); it != all_drawables.end(); ++it) {
		if ((*it)->get_base_name() == name) {
			index = picviz_max(index, (*it)->get_index() + 1);
		}
	}
	name->append(QString(":%1 (parallel view)").arg(index));
	window_id = PVGL::wtk_window_int_create(qPrintable(*name), default_width, default_height);
	current_view = new PVGL::PVView(window_id, pvsdk_messenger);
	current_view->set_index(index);
	current_view->set_base_name(base_name);
	current_view->set_name(*name);

	PRINT_OPENGL_ERROR ();
	glewInit ();
	// XXX The following is only necessary when using OpenGL 3.3 Core.
		//{
			//std::cout << "The following error (if any) is due to a bug in the GLEW library" << std::endl;
			//PRINT_OPENGL_ERROR ();
			//fixing_glew_bugs ();
		//}
	float win_r = pvconfig.value("pvgl/window_r", 0.2f).toFloat();
	float win_g = pvconfig.value("pvgl/window_g", 0.2f).toFloat();
	float win_b = pvconfig.value("pvgl/window_b", 0.2f).toFloat();
	float win_a = pvconfig.value("pvgl/window_a", 1.0f).toFloat();
	glClearColor(win_r, win_g, win_b, win_a);
	glEnable(GL_DEPTH_TEST);
	glClear(GL_COLOR_BUFFER_BIT);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	PVGL::wtk_buffers_swap();

	PVGL::wtk_set_display_func(PVGL::PVMain::display_callback);
	PVGL::wtk_set_keyboard_func(PVGL::PVMain::keyboard_callback);
	PVGL::wtk_set_special_func(PVGL::PVMain::special_callback);
	PVGL::wtk_set_mouse_func(PVGL::PVMain::button_callback);
	PVGL::wtk_set_motion_func(PVGL::PVMain::motion_callback);
	PVGL::wtk_set_reshape_func(PVGL::PVMain::reshape_callback);
	PVGL::wtk_set_close_func(PVGL::PVMain::close_callback);
	// PVGL::wtk_set_entry_func(PVGL::PVMain::entry_callback);
	PVGL::wtk_set_passive_motion_func(PVGL::PVMain::passive_motion_callback);

	transient_view = current_view;
}

/******************************************************************************
 *
 * PVGL::PVMain::create_scatter
 *
 *****************************************************************************/
void PVGL::PVMain::create_scatter(QString *name, Picviz::PVView_p pv_view)
{
	PVGL::PVScatter *new_scatter;
	int       window_id;
	int       index = 0;
	QString   base_name (*name);
	int default_width = pvconfig.value("pvgl/scatter_view_width", PVGL_VIEW_DEFAULT_WIDTH).toInt();
	int default_height = pvconfig.value("pvgl/scatter_view_height", PVGL_VIEW_DEFAULT_HEIGHT).toInt();

	PVLOG_DEBUG("PVGL::PVMain::%s\n", __FUNCTION__);

	for (std::list<PVGL::PVDrawable*>::iterator it = all_drawables.begin(); it != all_drawables.end(); ++it) {
		if ((*it)->get_base_name() == name) {
			index = picviz_max(index, (*it)->get_index() + 1);
		}
	}
	name->append(QString(":%1 (scatter view)").arg(index));
	window_id = PVGL::wtk_window_int_create(qPrintable(*name), default_width, default_height);
	new_scatter = new PVGL::PVScatter(window_id, pvsdk_messenger);
	new_scatter->set_index(index);
	new_scatter->set_base_name(base_name);
	new_scatter->set_name(*name);


	PRINT_OPENGL_ERROR ();
	glewInit ();
	float win_r = pvconfig.value("pvgl/window_r", 0.2f).toFloat();
	float win_g = pvconfig.value("pvgl/window_g", 0.2f).toFloat();
	float win_b = pvconfig.value("pvgl/window_b", 0.2f).toFloat();
	float win_a = pvconfig.value("pvgl/window_a", 1.0f).toFloat();
	glClearColor(win_r, win_g, win_b, win_a);
	// glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
	glEnable(GL_DEPTH_TEST);
	glClear(GL_COLOR_BUFFER_BIT);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	PVGL::wtk_buffers_swap();
	new_scatter->init(pv_view);

	PVGL::wtk_set_display_func(PVGL::PVMain::display_callback);
	PVGL::wtk_set_keyboard_func(PVGL::PVMain::keyboard_callback);
	PVGL::wtk_set_special_func(PVGL::PVMain::special_callback);
	PVGL::wtk_set_mouse_func(PVGL::PVMain::button_callback);
	PVGL::wtk_set_motion_func(PVGL::PVMain::motion_callback);
	PVGL::wtk_set_reshape_func(PVGL::PVMain::reshape_callback);
	PVGL::wtk_set_close_func(PVGL::PVMain::close_callback);
	PVGL::wtk_set_entry_func(PVGL::PVMain::entry_callback);
	PVGL::wtk_set_passive_motion_func(PVGL::PVMain::passive_motion_callback);

	all_drawables.push_back(new_scatter);
}

/******************************************************************************
 *
 * PVGL::PVMain::timer_func
 *
 *****************************************************************************/
void PVGL::PVMain::timer_func(int)
{
	PVSDK::PVMessage message;

	PVLOG_HEAVYDEBUG("PVGL::PVMain::%s\n", __FUNCTION__);

	if (_should_stop) {
		PVLOG_ERROR("PVGL::PVMain::%s: we are exiting, don't do too much!\n", __FUNCTION__);
		return;
	}
        

	// Do we have mail?
	if (pvsdk_messenger->get_message_for_gl(message)) {
		switch (message.function) {
			case PVSDK_MESSENGER_FUNCTION_PLEASE_WAIT:
						{
							QString *name = reinterpret_cast<QString *>(message.pointer_1);
							create_view(name);
						}
					break;
			case PVSDK_MESSENGER_FUNCTION_CREATE_VIEW:
						{
							all_drawables.push_back(transient_view);
							PVGL::wtk_set_current_window(transient_view->get_window_id());
							transient_view->init(message.pv_view);
							message.pointer_1 = new QString(transient_view->get_name());
							transient_view = 0;
							message.function = PVSDK_MESSENGER_FUNCTION_VIEW_CREATED;
							pvsdk_messenger->post_message_to_qt(message);
						}
					break;
			case PVSDK_MESSENGER_FUNCTION_ENSURE_VIEW:
						{
							// Ensure that a GL view has been created for a given Picviz::PVView
							bool found_drawable = false;
							for (std::list<PVGL::PVDrawable*>::iterator it = all_drawables.begin(); it != all_drawables.end(); ++it) {
								PVGL::PVView *pv_view = dynamic_cast<PVGL::PVView*>(*it);
								if (pv_view && pv_view->get_libview() == message.pv_view) {
									found_drawable = true;
									break;
								}
							}
							if (found_drawable) {
								// Found one, so do nothing.
								break;
							}

							// Create a parallel view
							QString *name = reinterpret_cast<QString *>(message.pointer_1);
							create_view(name);
							all_drawables.push_back(transient_view);
							PVGL::wtk_set_current_window(transient_view->get_window_id());
							transient_view->init(message.pv_view);
							message.pointer_1 = new QString(transient_view->get_name());
							transient_view = 0;
							message.function = PVSDK_MESSENGER_FUNCTION_VIEW_CREATED;
							pvsdk_messenger->post_message_to_qt(message);
						}
					break;
			case PVSDK_MESSENGER_FUNCTION_ICONIFY_ALL_DRAWABLES:
					for (std::list<PVGL::PVDrawable*>::iterator it = all_drawables.begin(); it != all_drawables.end(); ++it) {
						PVGL::wtk_set_current_window((*it)->get_window_id());
						// TODO: wtk_iconify_window
						glutIconifyWindow();
					}
					break;
			case PVSDK_MESSENGER_FUNCTION_SHOW_DRAWABLES_OF_PVVIEW:
					for (std::list<PVGL::PVDrawable*>::iterator it = all_drawables.begin(); it != all_drawables.end(); ++it) {
						if ((*it)->get_libview() == message.pv_view) {
							PVGL::wtk_set_current_window((*it)->get_window_id());
							glutShowWindow();
						}
					}
					break;
			case PVSDK_MESSENGER_FUNCTION_CREATE_SCATTER_VIEW:
						{
							QString *name = reinterpret_cast<QString *>(message.pointer_1);
							create_scatter(name, message.pv_view);
							message.function = PVSDK_MESSENGER_FUNCTION_VIEW_CREATED;
							pvsdk_messenger->post_message_to_qt(message);
						}
					break;
				case PVSDK_MESSENGER_FUNCTION_DESTROY_TRANSIENT:
							{
								if (transient_view) {
									PVGL::wtk_destroy_window(transient_view->get_window_id());
									delete transient_view;
									transient_view = 0;
								}
							}
						break;
			case PVSDK_MESSENGER_FUNCTION_REFRESH_VIEW:
					for (std::list<PVGL::PVDrawable*>::iterator it = all_drawables.begin(); it != all_drawables.end(); ++it) {
						PVGL::PVView *pv_view = dynamic_cast<PVGL::PVView*>(*it);
						if (pv_view) {
							if (pv_view->get_libview() == message.pv_view) {
								PVGL::wtk_set_current_window((*it)->get_window_id());
								if (message.int_1 & PVSDK_MESSENGER_REFRESH_AXES) {
									pv_view->update_axes();
								}
								if (message.int_1 & PVSDK_MESSENGER_REFRESH_AXES_COUNT) {
									pv_view->change_axes_count();
									pv_view->update_axes();
								}
								if (message.int_1 & PVSDK_MESSENGER_REFRESH_COLOR) {
									pv_view->update_colors();
								}
								if (message.int_1 & PVSDK_MESSENGER_REFRESH_Z) {
									pv_view->update_z();
								}
								if (message.int_1 & PVSDK_MESSENGER_REFRESH_POSITIONS) {
									pv_view->update_positions();
								}
								if (message.int_1 & PVSDK_MESSENGER_REFRESH_ZOMBIES) {
									pv_view->update_zombies();
								}
								if (message.int_1 & PVSDK_MESSENGER_REFRESH_SELECTION) {
									pv_view->update_selections();
								}
								if (message.int_1 & PVSDK_MESSENGER_REFRESH_SELECTED_LAYER) {
									pv_view->update_current_layer();
								}
								//(*it)->update_all();
								PVGL::wtk_window_need_redisplay();
							}
						}
						PVGL::PVScatter *pv_scatter = dynamic_cast<PVGL::PVScatter*>(*it);
						if (pv_scatter) {
							if (pv_scatter->get_libview() == message.pv_view) {
								PVGL::wtk_set_current_window((*it)->get_window_id());
								if (message.int_1 & PVSDK_MESSENGER_REFRESH_COLOR) {
									pv_scatter->update_arrays_colors();
								}
								if (message.int_1 & PVSDK_MESSENGER_REFRESH_Z) {
									pv_scatter->update_arrays_z();
								}
								if (message.int_1 & PVSDK_MESSENGER_REFRESH_POSITIONS) {
									pv_scatter->update_arrays_positions();
								}
								if (message.int_1 & PVSDK_MESSENGER_REFRESH_ZOMBIES) {
									pv_scatter->update_arrays_zombies();
								}
								if (message.int_1 & PVSDK_MESSENGER_REFRESH_SELECTION) {
									pv_scatter->update_arrays_selection();
								}
									pv_scatter->update_arrays_colors();
									pv_scatter->update_arrays_z();
									pv_scatter->update_arrays_positions();
									pv_scatter->update_arrays_zombies();
									pv_scatter->update_arrays_selection();
								PVGL::wtk_window_need_redisplay();
							}
						}
					}
					break;
			case PVSDK_MESSENGER_FUNCTION_TAKE_SCREENSHOT:
						{
							int rank = 0;
							for (std::list<PVGL::PVDrawable*>::iterator it = all_drawables.begin(); it != all_drawables.end(); ++it) {
								if ((*it)->get_libview() == message.pv_view && ((*it)->get_window_id() == message.int_1 || message.int_1 == -1)) {
									unsigned char *image_data;
									PVGL::wtk_set_current_window((*it)->get_window_id());
									QString *filename = NULL;
									QImage* image_ret = NULL;
									if (message.int_2 == true) {
										// We have a filename, so save it;
										filename = reinterpret_cast<QString *>(message.pointer_1);
									}
									else {
										// Just save the QImage object
										image_ret = reinterpret_cast<QImage*>(message.pointer_1);
									}

									if (filename) {
										if (message.int_1 == -1) {
											filename->insert(picviz_max(0,filename->lastIndexOf('.')),QString("_%1").arg(rank));
											rank++;
										}
									}
									//std::cout << "should take a screenshot and save in: " << filename->toUtf8().data() << std::endl;
									image_data = new unsigned char[3 * (*it)->get_width() * (*it)->get_height()];
									glPixelStorei(GL_PACK_LSB_FIRST, GL_FALSE);
									glPixelStorei(GL_PACK_ROW_LENGTH, 0);
									glPixelStorei(GL_PACK_ALIGNMENT, 1);

									glPixelStorei(GL_UNPACK_LSB_FIRST, GL_FALSE);
									glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
									glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

									glReadPixels(0, 0, (*it)->get_width(), (*it)->get_height(), GL_RGB, GL_UNSIGNED_BYTE, image_data);

									{
										QImage image(image_data, (*it)->get_width(), (*it)->get_height(), 3 * (*it)->get_width(),
												QImage::Format_RGB888);
										if (filename) {
											QImage flipped_image = image.mirrored(false, true);
											flipped_image.save(*filename);
										}
										else {
											*image_ret = image.mirrored(false, true);
										}
									}

									delete[] image_data;
									if (message.int_1 != -1) {
										break;
									}
								}
							}
							message.function = PVSDK_MESSENGER_FUNCTION_SCREENSHOT_TAKEN;
							pvsdk_messenger->post_message_to_qt(message);
						}
					break;
			case PVSDK_MESSENGER_FUNCTION_DESTROY_VIEWS:
						{
							std::list<PVGL::PVDrawable*>::iterator it = all_drawables.begin();
							while (it != all_drawables.end()) {
								if ((*it)->get_libview() == message.pv_view) {
									std::list<PVGL::PVDrawable*>::iterator it_rem = it;

									// First remove the tasks for that drawable
									idle_manager.remove_tasks_for_drawable((*it));

									// Then destroy the window
									PVGL::wtk_set_current_window((*it)->get_window_id());
									QString *name = new QString((*it)->get_name());
									PVSDK::PVMessage message;

									message.function = PVSDK_MESSENGER_FUNCTION_ONE_VIEW_DESTROYED;
									message.pv_view = (*it)->get_libview();
									message.pointer_1 = name;

									pvsdk_messenger->post_message_to_qt(message);
									PVGL::wtk_destroy_window((*it)->get_window_id());

									it++;

									// Delete the resources of that view
									PVGL::PVDrawable *pvv = *it_rem;
									all_drawables.erase(it_rem);
									delete pvv;

									continue;
								}
								it++;
							}
							message.function = PVSDK_MESSENGER_FUNCTION_VIEWS_DESTROYED;
							pvsdk_messenger->post_message_to_qt(message);
						}
					break;
			case PVSDK_MESSENGER_FUNCTION_UPDATE_OTHER_SELECTIONS:
					for (std::list<PVGL::PVDrawable*>::iterator it = all_drawables.begin(); it != all_drawables.end(); ++it) {
						if ((*it)->get_libview() == message.pv_view && (*it)->get_window_id() != message.int_1) {
							PVLOG_DEBUG("my id: %d, message id: %d\n", (*it)->get_window_id(), message.int_1);
							PVGL::PVView *pv_view = dynamic_cast<PVGL::PVView*>(*it);
							if (pv_view) {
								PVGL::wtk_set_current_window(pv_view->get_window_id());
								pv_view->update_selection_except_listing();
								pv_view->get_lines().update_arrays_selection();
								// disabled for now pv_view->get_map().update_arrays_selection();
								pv_view->set_update_line_dirty();
								PVGL::wtk_window_need_redisplay();
							}
						}
					}
					break;
			case PVSDK_MESSENGER_FUNCTION_REINIT_PVVIEW:
					for (std::list<PVGL::PVDrawable*>::iterator it = all_drawables.begin(); it != all_drawables.end(); ++it) {
						if ((*it)->get_libview() == message.pv_view) {
							PVGL::wtk_set_current_window((*it)->get_window_id());
							(*it)->reinit_picviz_view();
							PVGL::wtk_window_need_redisplay();
						}
					}
					break;
			case PVSDK_MESSENGER_FUNCTION_TOGGLE_DISPLAY_EDGES:
					for (std::list<PVGL::PVDrawable*>::iterator it = all_drawables.begin(); it != all_drawables.end(); ++it) {
						PVGL::PVView *pv_view = dynamic_cast<PVGL::PVView*>(*it);
						if (pv_view) {
							pv_view->axes_toggle_show_limits();
						}
					}
					break;
		        case PVSDK_MESSENGER_FUNCTION_SET_VIEW_WINDOWTITLE:
				{
					PVLOG_INFO("Message to set the window title no handled today!\n");
				// QString *name = reinterpret_cast<QString *>(message.pointer_1);
				// for (std::list<PVGL::PVDrawable*>::iterator it = all_drawables.begin(); it != all_drawables.end(); ++it) {
				// 	PVGL::PVView *pv_view = dynamic_cast<PVGL::PVView*>(*it);

				// 	pv_view->set_name(*name);
				// }
				}
				break;
			default:
					PVLOG_ERROR("PVGL::%s unknown function in a message: %d\n", __FUNCTION__, message.function);
		}
	}
	// Check if we need to reselect
	if (PVGL::wtk_time_ms_elapsed_since_init() - last_key_pressed_time > mouse_motion_refresh_timer) {

		//PVLOG_DEBUG("   we need to reselect\n");
		for (std::list<PVGL::PVDrawable*>::iterator it = all_drawables.begin(); it != all_drawables.end(); ++it) {
			PVGL::PVView *pv_view = dynamic_cast<PVGL::PVView*>(*it);
			if (pv_view) {
				if (pv_view->is_update_line_dirty()) {

					PVLOG_DEBUG("   reselect\n");
					PVGL::wtk_set_current_window(pv_view->get_window_id());
					Picviz::PVView_p picviz_view = pv_view->get_libview();
					if (!picviz_view)
						continue;
					if (picviz_view->square_area.is_dirty()) {
						PVLOG_DEBUG("   picviz_view->process_from_selection\n");
						//picviz_view->gl_cdlocker.lock();
						picviz_view->selection_A2B_select_with_square_area(picviz_view->layer_stack_output_layer.get_selection(), picviz_view->volatile_selection);
						picviz_view->process_from_selection();
						picviz_view->square_area.set_clean();
						//picviz_view->gl_call_locker.unlock();
						PVLOG_DEBUG("   pv_view->update_lines\n");
						pv_view->get_lines().update_arrays_selection();
						// disabled for now pv_view->get_map().update_arrays_selection();
						pv_view->update_lines();
					}
					//                                        PVLOG_DEBUG("   pv_view->update_lines\n");
					//					pv_view->get_lines().update_arrays_selection();
					//					pv_view->get_map().update_arrays_selection();
					//					pv_view->update_lines();
				}
			}
			PVGL::PVScatter *pv_scatter = dynamic_cast<PVGL::PVScatter*>(*it);
			if (pv_scatter) {
				PVGL::wtk_set_current_window(pv_scatter->get_window_id());
				pv_scatter->update_arrays_selection();
			}
		}

	}
	// Check if we need to resize
	if (PVGL::wtk_time_ms_elapsed_since_init() - last_reshape_time_time > view_resize_update_timer) {
		for (std::list<PVGL::PVDrawable*>::iterator it = all_drawables.begin(); it != all_drawables.end(); ++it) {
			PVGL::PVView *pv_view = dynamic_cast<PVGL::PVView*>(*it);
			if (pv_view) {
				if (pv_view->is_set_size_dirty()) {
					PVGL::wtk_set_current_window(pv_view->get_window_id());
					pv_view->update_set_size();
					glViewport(0, 0, pv_view->get_width(), pv_view->get_height());
					PVGL::wtk_window_need_redisplay();
				}
			}
		}
	}
	// Refresh every window every 20 ms
	for (std::list<PVGL::PVDrawable*>::iterator it = all_drawables.begin(); it != all_drawables.end(); ++it) {
		PVGL::wtk_set_current_window((*it)->get_window_id());
		PVGL::wtk_window_need_redisplay();
	}
	if (transient_view) {
		PVGL::wtk_set_current_window(transient_view->get_window_id());
		PVGL::wtk_window_need_redisplay();
	}
        
	PVGL::wtk_set_timer_func(window_timer_refresh, timer_func, 0);
}

/******************************************************************************
 *
 * PVGL::PVMain::stop
 *
 *****************************************************************************/

void PVGL::PVMain::stop()
{
	PVLOG_INFO("PVGL::%s: stopping\n", __FUNCTION__);
	_should_stop = true;
}

/******************************************************************************
 *
 * pvgl_init
 *
 *****************************************************************************/
bool pvgl_init(PVSDK::PVMessenger *messenger)
{
	int argc = 1;
	char *argv[] = { const_cast<char*>("PVGL"), NULL };

	int main_loop_timer_refresh = pvconfig.value("pvgl/main_loop_timer_refresh", 5/*20*/).toInt();

	view_resize_update_timer = pvconfig.value("pvgl/view_resize_update_timer", PVGL_VIEW_RESIZE_UPDATE_TIMER).toInt();
	window_timer_refresh = pvconfig.value("pvgl/window_timer_refresh", 20).toInt();
	mouse_motion_refresh_timer = pvconfig.value("pvgl/mouse_motion_refresh_timer", 100).toInt();

	pvsdk_messenger = messenger;

	if (pvgl_share_path_exists() == false) {
		PVLOG_FATAL("Cannot open PVGL share directory %s!\n", pvgl_get_share_path().c_str());
		return false;
	} else {
		PVLOG_INFO("Using PVGL share directory %s\n", pvgl_get_share_path().c_str());
	}

	PVGL::wtk_init(argc, argv);

	// Wait for the first message
	PVLOG_DEBUG("PVGL::%s Everything created, waiting for message.\n", __FUNCTION__);
	for(;;) { // FIXME! Dont eat all my cpu!
		PVSDK::PVMessage message;
		if (messenger->get_message_for_gl(message)) {
			switch (message.function) {
				case PVSDK_MESSENGER_FUNCTION_PLEASE_WAIT:
							{
								QString *name = reinterpret_cast<QString *>(message.pointer_1);
								PVGL::PVMain::create_view(name);
								//message.function = PVSDK_MESSENGER_FUNCTION_VIEW_CREATED;
								//pvsdk_messenger->post_message_to_qt(message);
								PVGL::wtk_set_timer_func(main_loop_timer_refresh, PVGL::PVMain::timer_func, 0);
								PVGL::wtk_main_loop();

								PVGL::wtk_init(argc, argv);
							}
						break;
				case PVSDK_MESSENGER_FUNCTION_DESTROY_TRANSIENT:
							{
								if (transient_view) {
									PVGL::wtk_destroy_window(transient_view->get_window_id());
									delete transient_view;
									transient_view = 0;
								}
							}
						break;
				case PVSDK_MESSENGER_FUNCTION_CREATE_VIEW:
							{
								all_drawables.push_back(transient_view);
								PVGL::wtk_set_current_window(transient_view->get_window_id());
								transient_view->init(message.pv_view);
								message.pointer_1 = new QString(transient_view->get_name());
								transient_view = 0;
								message.function = PVSDK_MESSENGER_FUNCTION_VIEW_CREATED;
								pvsdk_messenger->post_message_to_qt(message);
								PVGL::wtk_set_timer_func(main_loop_timer_refresh, PVGL::PVMain::timer_func, 0);
								PVGL::wtk_main_loop();

								PVGL::wtk_init(argc, argv);
							}
						break;
				case PVSDK_MESSENGER_FUNCTION_CREATE_SCATTER_VIEW:
							{
								QString *name = reinterpret_cast<QString *>(message.pointer_1);
								PVGL::PVMain::create_scatter(name, message.pv_view);
								PVLOG_INFO("PVGL::%s scatter view created\n", __FUNCTION__);
								message.function = PVSDK_MESSENGER_FUNCTION_VIEW_CREATED;
								message.pointer_1 = new QString(*name);
								pvsdk_messenger->post_message_to_qt(message);
								PVGL::wtk_set_timer_func(main_loop_timer_refresh, PVGL::PVMain::timer_func, 0);
								PVGL::wtk_main_loop();

								PVGL::wtk_init(argc, argv);
							}
						break;
				default:
						PVLOG_ERROR("PVGL::%s unknown function in a message: %d\n", __FUNCTION__, message.function);
			}
		} else {
#ifdef WIN32
			Sleep(1000);
#else
			sleep(1);
#endif
		}
	}

	return true;
}

QList<Picviz::PVView_p> PVGL::PVMain::list_displayed_picviz_views()
{
	QList<Picviz::PVView_p> ret;
	std::list<PVGL::PVDrawable*>::const_iterator it;
	for (it = all_drawables.begin(); it != all_drawables.end(); it++) {
		Picviz::PVView_p v = (*it)->get_libview();
		if (!ret.contains(v)) {
			ret << v;
		}
	}
	return ret;
}
