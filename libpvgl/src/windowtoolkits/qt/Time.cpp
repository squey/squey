//! \file Buffer.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011

#ifdef USE_WTK_QT

#include <pvkernel/core/general.h>

#include "../core/include/Time.h"
#include "include/global.h"

#include <tbb/tick_count.h>

int PVGL::wtk_time_ms_elapsed_since_init()
{
	double seconds = (tbb::tick_count::now() - PVGL::WTKQt::Global::get_ms_start()).seconds();
	return (int) (seconds * 1000.0);
}

#endif	// USE_WTK_QT
