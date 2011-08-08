//! \file WtkWindow.cpp
//! Functions that hide the window to the pvgl code
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011

#include "include/WtkWindow.h"

PVGL::WTK::WtkWindow::WtkWindow(int win_id) 
{
	_win_id = win_id;
}

PVGL::WTK::WtkWindow::WtkWindow(void *win_ptr) 
{
	_win_ptr = win_ptr;
}
