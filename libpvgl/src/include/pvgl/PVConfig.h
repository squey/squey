//! \file PVConfig.h
//! $Id: PVConfig.h 2456 2011-04-22 11:44:54Z dindinx $
//! Copyright (C) Sébastien Tricaud 2009, 2010
//! Copyright (C) Philippe Saade 2009, 2010
//! Copyright (C) Picviz Labs 2011

#ifndef LIBPVGL_CONFIG_H
#define LIBPVGL_CONFIG_H

#include <pvkernel/core/general.h>

// View configuration
//! The default width of a newly create pvgl_view.
const int PVGL_VIEW_DEFAULT_WIDTH  = 600;
//! The default height of a newly create pvgl_view.
const int PVGL_VIEW_DEFAULT_HEIGHT = 500;
//! The color of the "Loading..." text;
const ubvec4 PVGL_VIEW_LOADING_COLOR(255, 255, 255, 255);
//! The percentage of the horizontal border zone of fbo outside the screen (between 0.0 and 100.0)
const float PVGL_VIEW_OUTSIDE_HORIZONTAL_BORDER = 100.0f;
//! The percentage of the vertical border zone of fbo outside the screen (between 0.0 and 100.0)
const float PVGL_VIEW_OUTSIDE_VERTICAL_BORDER = 25.0f;

//! Time (in ms) between the last resize event and the real resizing
const int PVGL_VIEW_RESIZE_UPDATE_TIMER = 5/*50*/;

// Map configuration
//! The default position of the map (X coordinate).
const int PVGL_MAP_DEFAULT_X = 5;
//! The default position of the map (Y coordinate).
const int PVGL_MAP_DEFAULT_Y = 70;

// Axes configuration
//! The color used to draw the names of the axes.
const ubvec4 PVGL_AXES_NAMES_COLOR(255, 255, 255, 255);

#endif
