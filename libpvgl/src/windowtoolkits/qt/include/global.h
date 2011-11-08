#ifndef PVGL_WTK_QT_GLOBAL
#define PVGL_WTK_QT_GLOBAL

#include <pvkernel/core/stdint.h>
#include <tbb/tick_count.h>

namespace PVGL {
namespace WTKQt {

class PVGLWidget;

class Global
{
	friend class PVGLWidget;

public:
	static pvgl_callback_close_t _callback_close;
	static pvgl_callback_display_t _callback_display;
	static pvgl_callback_entry_t _callback_entry;
	static pvgl_callback_idle_t _callback_idle;
	static pvgl_callback_keyboard_t _callback_keyboard;
	static pvgl_callback_motion_t _callback_motion;
	static pvgl_callback_passive_motion_t _callback_passive_motion;
	static pvgl_callback_mouse_t _callback_mouse;
	static pvgl_callback_reshape_t _callback_reshape;
	static pvgl_callback_special_t _callback_special;

public:
	static int create_window(const char* name, int widget, int height);
	static void set_current_window_id(int id);

public:
	static inline int get_current_window_id() { return _cur_win_id; }
	static inline PVGLWidget* get_current_window() { return _cur_win; }
	static void launch_timer(unsigned int msecs, pvgl_callback_timer_t f, int value);
	static void set_ms_start();
	static tbb::tick_count get_ms_start() { return _init_start; }
	static PVGLWidget* get_widget_from_id(int id);

protected:
	static void set_current_window(PVGLWidget* widget);

private:
	static int _cur_win_id;
	static PVGLWidget* _cur_win;
	static map_glwindows _windows;
	static tbb::tick_count _init_start;
};

}
}

#endif
