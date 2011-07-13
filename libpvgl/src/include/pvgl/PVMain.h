//! \file PVMain.h
//! $Id: PVMain.h 2888 2011-05-19 07:29:43Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009, 2010
//! Copyright (C) Philippe Saade 2009, 2010
//! Copyright (C) Picviz Labs 2011

#ifndef LIBPVGL_PVMAIN_H
#define LIBPVGL_PVMAIN_H

#include <pvcore/general.h>

#include <pvgl/PVIdleManager.h>
#include <QtCore>

namespace PVGL {

class PVDrawable;
/**
	*
	*/
struct LibExport PVMain {

	/**
	*
	*/
	static void display_callback();

	/**
	*
	*/
	static void idle_callback();

	/**
	*
	* @param window_id
	*/
	static PVGL::PVDrawable *get_drawable_from_id(int window_id);
	
	/**
	*
	* @param key
	* @param x
	* @param y
	*/
	static void keyboard_callback(unsigned char key, int x, int y);
	
	/**
	*
	* @param key
	* @param x
	* @param y
	*/
	static void special_callback(int key, int x, int y);
	
	/**
	*
	* @param delta_zoom_level
	* @param x
	* @param y
	*/
	static void mouse_wheel(int delta_zoom_level, int x, int y);
	
	/**
	*
	* @param button
	* @param x
	* @param y
	*/
	static void mouse_down(int button, int x, int y);
	
	/**
	*
	* @param button
	* @param x
	* @param y
	*/
	static void mouse_release(int button, int x, int y);
	
	/**
	*
	* @param button
	* @param state
	* @param x
	* @param y
	*/
	static void button_callback(int button, int state, int x, int y);
	
	/**
	*
	* @param x
	* @param y
	*/
	static void motion_callback(int x, int y);
	
	/**
	*
	* @param width
	* @param height
	*/
	static void reshape_callback(int width, int height);
	
	/**
	*
	*/
	static void close_callback(void);
	
	/**
	*
	* @param state
	*/
	static void entry_callback(int state);
	
	/**
	*
	* @param x
	* @param y
	*/
	static void passive_motion_callback(int x, int y);
	
	/**
	*
	* @param name
	*/
	static void create_view(QString *name);
	
	/**
	*
	* @param name
	* @param pv_view
	*/
	static void create_scatter(QString *name, Picviz::PVView_p pv_view);
	
	/**
	*
	*/
	static void timer_func(int);

	
	/**
	*
	*/
	static void stop();
    
        

};
#ifndef DEF_passive_motion_locker_mutex
#define DEF_passive_motion_locker_mutex
static QMutex moving_locker_mutex;//!<
static bool mouse_is_moving;//!<
#endif
}
extern PVGL::PVIdleManager idle_manager;
#endif
