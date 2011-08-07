//! \file PVDrawable.h
//! $Id: PVDrawable.h 2985 2011-05-26 09:01:11Z dindinx $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef LIBPVGL_PVDRAWABLE_H
#define LIBPVGL_PVDRAWABLE_H

#include <QtCore>

#include <picviz/PVView.h>

namespace PVGL {
class PVView;
}

#include <pvgl/PVCom.h>
#include <pvgl/PVWidgetManager.h>
#include <pvgl/PVIdleManager.h>

namespace PVGL {
/**
 *
 */
class LibGLDecl PVDrawable {
protected:
	Picviz::PVView_p picviz_view;    //!<
	PVCom           *pv_com;         //!<

	int              width;          //!< The width of the drawable window.
	int              height;         //!< The height of the drawable window.

	PVWidgetManager *widget_manager; //!<
	int max_lines_per_redraw;

private:
	int              index;          //!< Index number, which is equal to the number of #PVDrawable opened with the same Picviz::PVView.
	QString          base_name;      //!< Name of this view (filename only).
	QString          name;           //!< Name of this view (filename + index).
	bool             fullscreen;     //!< Is the current view fullscreened?
	int              window_id;      //!<
public:
	/**
	 * Constructor.
	 *
	 * @param window_id
	 * @param com
	 */
	PVDrawable(int window_id, PVCom *com);

	virtual ~PVDrawable();

	/**
	 *
	 */
	virtual void init(Picviz::PVView_p view);

	//! \name getters
	int get_max_lines_per_redraw() const;

	/**
	 *
	 *
	 * @return The current width of the drawable window.
	 */
	int get_width()const{return width;}

	/**
	 *
	 *
	 * @return
	 */
	int get_height()const{return height;}

	/**
	 *
	 *
	 * @return
	 */
	Picviz::PVView_p get_libview() const {return picviz_view;}

	/**
	 *
	 *
	 * @return
	 */
	int get_window_id()const{return window_id;}

	/**
	 *
	 * @return
	 */
	int get_index()const{return index;}

	/**
	 *
	 * @return
	 */
	QString get_base_name() const { return base_name; }
	//! \}

	/**
	 *
	 * @return
	 */
	QString get_name()const{return name;}

	/**
	 *
	 * @return true if the view is fullscreened, false otherwise
	 */
	bool is_fullscreen()const{return fullscreen;}
	//! \}

	/**
	 * @param on true if the view is to be fullscreened, false otherwise.
	 */
	void toggle_fullscreen(bool on){fullscreen = on;}

	/**
	 * @param i
	 */
	void set_index(int i){index = i;}

	/**
	 * @param name
	 */
	void set_base_name(const QString &name){base_name = name;}

	/**
	 * @param name_
	 */
	void set_name(const QString &name_){name = name_;}


	//! \name drawing
	//! \{
	/**
	 *
	 */
	virtual void draw(void) = 0;
  //! \}

	/**
	 *
	 * @param width
	 * @param height
	 */
	virtual void set_size(int width, int height);

	/**
	 * @param key
	 * @param x
	 * @param y
	 */
	virtual void keyboard(unsigned char key, int x, int y);

	/**
	 * @param key
	 * @param x
	 * @param y
	 */
	virtual void special_keys(int key, int x, int y);

	/**
	 *
	 * @param delta_zoom_level
	 * @param x
	 * @param y
	 */
	virtual void mouse_wheel(int delta_zoom_level, int x, int y);

	/**
	 *
	 * @param button     A bitfield indicating which mouse buttons are pressed.
	 * @param x          The x coordinate of the mouse pointer, in window space.
	 * @param y          The y coordinate of the mouse pointer, in window space.
	 * @param modifiers
	 */
	virtual void mouse_down(int button, int x, int y, int modifiers);

	/**
	 * @param x          The x coordinate of the mouse pointer, in window space.
	 * @param y          The y coordinate of the mouse pointer, in window space.
	 * @param modifiers
	 */
	virtual bool mouse_move(int x, int y, int modifiers);

	/**
	 *
	 * @param button     A bitfield indicating which mouse buttons are pressed.
	 * @param x          The x coordinate of the mouse pointer, in window space.
	 * @param y          The y coordinate of the mouse pointer, in window space.
	 * @param modifiers
	 */
	virtual bool mouse_up(int button, int x, int y, int modifiers);

	/**
	 *
	 * @param x          The x coordinate of the mouse pointer, in window space.
	 * @param y          The y coordinate of the mouse pointer, in window space.
	 * @param modifiers
	 */
	virtual bool passive_motion(int x, int y, int modifiers);

	/**
	 * A scheduler implementation.
	 *
	 * @param kind
	 *
	 * @return
	 */
	int dumb_scheduler(PVGL::PVIdleTaskKinds kind);

	/**
	 * A scheduler implementation.
	 *
	 * @param kind
	 *
	 * @return
	 */
	int small_files_scheduler(PVGL::PVIdleTaskKinds kind);

	/**
	 *
	 * @param kind
	 *
	 * @return
	 */
	int (PVGL::PVDrawable::*current_scheduler)(PVGL::PVIdleTaskKinds kind);

	virtual void reinit_picviz_view() = 0;
};
}
#endif
