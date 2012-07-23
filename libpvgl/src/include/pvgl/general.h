/**
 * \file general.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef LIBPVGL_GENERAL_H
#define LIBPVGL_GENERAL_H

#include <pvkernel/core/general.h>

#include <pvsdk/PVMessenger.h>

#define MAX_LINES_PER_REDRAW 75000
#define MAX_LINES_FOR_INTERACTIVITY 10000 /* Used to decide whether we can interactively do things or not, such as the behavior of the event line */

/** Init the GL subsystem, and wait for the creation of a view.
 *
 * This is the main entry point of the PVGL library.
 * @param com The communication link between the Qt interface and the GL one. See #PVCom.
 */
bool LibGLDecl pvgl_init (PVSDK::PVMessenger *message);

#endif
