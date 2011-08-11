//! \file PVWidgetManager.h
//! $Id: PVWidgetManager.h 2480 2011-04-24 11:51:30Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef LIBPVGL_GUI_H
#define LIBPVGL_GUI_H

#define GLEW_STATIC 1
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <picviz/PVView.h>

#include <pvgl/PVFonts.h>
#include <pvgl/PVCom.h>


/**
 * \enum PVGLIconNames
 */
enum PVGLIconNames
{
	// Eventline icons.
  PVGL_ICON_EVENTLINE_BACK                  = 0,
	PVGL_ICON_EVENTLINE_LINE                  = 1,
	PVGL_ICON_EVENTLINE_LEFT_SLIDER           = 2,
	PVGL_ICON_EVENTLINE_LEFT_SLIDER_PRELIGHT  = 3,
	PVGL_ICON_EVENTLINE_RIGHT_SLIDER          = 4,
	PVGL_ICON_EVENTLINE_RIGHT_SLIDER_PRELIGHT = 5,
	PVGL_ICON_EVENTLINE_THUMB                 = 6,
	PVGL_ICON_EVENTLINE_THUMB_PRELIGHT        = 7,

	// Window border icons.
	PVGL_ICON_WINDOW_LT  = 8,
	PVGL_ICON_WINDOW_T   = 9,
	PVGL_ICON_WINDOW_RT  = 10,
	PVGL_ICON_WINDOW_L   = 11,
	PVGL_ICON_WINDOW_R   = 12,
	PVGL_ICON_WINDOW_LB  = 13,
	PVGL_ICON_WINDOW_B   = 14,
	PVGL_ICON_WINDOW_RB  = 15
};

namespace PVGL {
	class PVView;
struct LibGLDecl PVIcon
{
	int x;
	int y;
	int width;
	int height;
	int hot_x;
	int hot_y;

	PVIcon (int x_, int y_, int width_, int height_, int hot_x, int hot_y);
};

/**
 * \class PVWidgetManager
 */
class LibGLDecl PVWidgetManager {
	PVFont              pvgl_font;           //!< The PVFont attached to this PVWidgetManager
	GLuint              icon_texture;        //!< The Texture providing the icons of this PVWidgetManager
	std::vector<PVIcon> icons;               //!< 

public:
	/**
	 * Constructor.
	 */
	PVWidgetManager();

	/**
	 *
	 * @param x
	 * @param y
	 * @param text
	 * @param font_size
	 */
	void draw_text(int x, int y, const std::string &text, float font_size);

	/**
	 *
	 */
	void get_text_size(const std::string &text, int font_height, int &string_width, int &string_height, int &string_ascent);
	/**
	 *
	 * @param x
	 * @param y
	 * @param name
	 */
	void draw_icon(int x, int y, PVGLIconNames name);

	/**
	 *
	 * @param x
	 * @param y
	 * @param width
	 * @param height
	 * @param name
	 */
	void draw_icon_streched(int x, int y, int width, int height, PVGLIconNames name);
};
}
#endif
