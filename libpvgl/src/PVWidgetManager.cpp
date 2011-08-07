//! \file PVWidgetManager.cpp
//! $Id: PVWidgetManager.cpp 2520 2011-04-30 12:26:48Z stricaud $
//! Copyright (C) SÃ©bastien Tricaud 2009, 2010
//! Copyright (C) Philippe Saade 2009,2010
//! Copyright (C) Picviz Labs 2011

#include <sstream>

#include <QImage>

#define GLEW_STATIC 1
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <picviz/PVView.h>

#include <pvgl/PVConfig.h>
#include <pvgl/PVUtils.h>
#include <pvgl/PVCom.h>
#include <pvgl/PVLayout.h>
#include <pvgl/PVHBox.h>
#include <pvgl/PVVBox.h>
#include <pvgl/views/PVParallel.h>

#include <pvgl/PVWidgetManager.h>

/******************************************************************************
 *
 * PVGL::PVIcon::PVIcon
 *
 *****************************************************************************/
PVGL::PVIcon::PVIcon (int x_, int y_, int width_, int height_, int hot_x_, int hot_y_)
		: x(x_), y(y_), width(width_), height(height_), hot_x(hot_x_), hot_y(hot_y_)
{
	PVLOG_DEBUG("PVGL::PVIcon::%s\n", __FUNCTION__);
}

/******************************************************************************
 *
 * PVGL::PVWidgetManager::PVWidgetManager
 *
 *****************************************************************************/
PVGL::PVWidgetManager::PVWidgetManager()
{
	PVLOG_DEBUG("PVGL::PVWidgetManager::%s\n", __FUNCTION__);

	QImage image(QString((pvgl_get_share_path() + "gui.png").c_str()));

	glGenTextures(1, &icon_texture); PRINT_OPENGL_ERROR();
	glBindTexture(GL_TEXTURE_RECTANGLE, icon_texture); PRINT_OPENGL_ERROR();
	glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA, image.width(), image.height(), 0, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, image.bits()); PRINT_OPENGL_ERROR();
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST); PRINT_OPENGL_ERROR();
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST); PRINT_OPENGL_ERROR();
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); PRINT_OPENGL_ERROR();
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); PRINT_OPENGL_ERROR();

	// Eventline
	// The Eventline background
	icons.push_back(PVIcon(0, 0, 6, 40, 0, 0));
	// The Eventline middle line
	icons.push_back(PVIcon(7, 0, 5, 40, 0, 0));
	// The left slider
	icons.push_back(PVIcon(13, 0, 15, 31, 15, 16));
	// The prelighted left slider
	icons.push_back(PVIcon(29, 0, 15, 31, 15, 16));
	// The right slider
	icons.push_back(PVIcon(45, 0, 15, 31, 0, 16));
	// The prelighted right slider
	icons.push_back(PVIcon(61, 0, 15, 31, 0, 16));
	// The eventline thumb
	icons.push_back(PVIcon(77, 0, 21, 31, 11, 16));
	// The prelighted eventline thumb
	icons.push_back(PVIcon(99, 0, 21, 31, 11, 16));

	// The map (and other?) window
	// Top-Left
	icons.push_back(PVIcon(0, 64, 32, 32, 6, 14));
	// Top
	icons.push_back(PVIcon(32, 64, 32, 32, 0, 14));
	// Top-Right
	icons.push_back(PVIcon(64, 64, 32, 32, 22, 14));
	// Left
	icons.push_back(PVIcon(0, 96, 32, 32, 6, 0));
	// Right
	icons.push_back(PVIcon(64, 96, 32, 32, 22, 0));
	// Bottom-Left
	icons.push_back(PVIcon(0, 128, 32, 32, 6, 21));
	// Bottom
	icons.push_back(PVIcon(32, 128, 32, 32, 0, 21));
	// Bottom right
	icons.push_back(PVIcon(64, 128, 32, 32, 22, 21));

}

/******************************************************************************
 *
 * PVGL::PVWidgetManager::draw_icon
 *
 *****************************************************************************/
void PVGL::PVWidgetManager::draw_icon(int x, int y, PVGLIconNames name)
{
	PVLOG_HEAVYDEBUG("PVGL::PVWidgetManager::%s\n", __FUNCTION__);

	glEnable(GL_BLEND);
	glEnable(GL_TEXTURE_RECTANGLE);
	glBindTexture(GL_TEXTURE_RECTANGLE, icon_texture); PRINT_OPENGL_ERROR();
	glColor3f(1,1,1);
	glBegin(GL_QUADS);
		glTexCoord2f(icons[name].x,                     icons[name].y);                      glVertex2f(x - icons[name].hot_x,                     y - icons[name].hot_y);
		glTexCoord2f(icons[name].x,                     icons[name].y + icons[name].height); glVertex2f(x - icons[name].hot_x,                     y - icons[name].hot_y + icons[name].height);
		glTexCoord2f(icons[name].x + icons[name].width, icons[name].y + icons[name].height); glVertex2f(x - icons[name].hot_x + icons[name].width, y - icons[name].hot_y + icons[name].height);
		glTexCoord2f(icons[name].x + icons[name].width, icons[name].y);                      glVertex2f(x - icons[name].hot_x + icons[name].width, y - icons[name].hot_y);
	glEnd();
	glDisable(GL_TEXTURE_RECTANGLE);
	glDisable(GL_BLEND);
}

/******************************************************************************
 *
 * PVGL::PVWidgetManager::draw_icon_streched
 *
 *****************************************************************************/
void PVGL::PVWidgetManager::draw_icon_streched(int x, int y, int width, int height, PVGLIconNames name)
{
	PVLOG_HEAVYDEBUG("PVGL::PVWidgetManager::%s\n", __FUNCTION__);

	glEnable(GL_BLEND);
	glEnable(GL_TEXTURE_RECTANGLE);
	glBindTexture(GL_TEXTURE_RECTANGLE, icon_texture); PRINT_OPENGL_ERROR();
	glColor3f(1,1,1);
	glBegin(GL_QUADS);
		glTexCoord2f(icons[name].x,                     icons[name].y);                      glVertex2f(x - icons[name].hot_x,         y - icons[name].hot_y);
		glTexCoord2f(icons[name].x,                     icons[name].y + icons[name].height); glVertex2f(x - icons[name].hot_x,         y - icons[name].hot_y + height);
		glTexCoord2f(icons[name].x + icons[name].width, icons[name].y + icons[name].height); glVertex2f(x - icons[name].hot_x + width, y - icons[name].hot_y + height);
		glTexCoord2f(icons[name].x + icons[name].width, icons[name].y);                      glVertex2f(x - icons[name].hot_x + width, y - icons[name].hot_y);
	glEnd();
	glDisable(GL_TEXTURE_RECTANGLE);
	glDisable(GL_BLEND);
}

/******************************************************************************
 *
 * PVGL::PVWidgetManager::draw_text
 *
 *****************************************************************************/
void PVGL::PVWidgetManager::draw_text(int x, int y, const std::string &text, float font_size)
{
	PVLOG_HEAVYDEBUG("PVGL::PVWidgetManager::%s\n", __FUNCTION__);

	pvgl_font.draw_text(x, y, text.c_str(), font_size);
}

/******************************************************************************
 *
 * PVGL::PVWidgetManager::get_text_size
 *
 *****************************************************************************/
void PVGL::PVWidgetManager::get_text_size(const std::string &text, int font_height, int &string_width, int &string_height, int &string_ascent)
{
  pvgl_font.get_text_size(text, font_height, string_width, string_height, string_ascent);
}

