/**
 * \file PVRoot.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVHIVE_WAXES_PICVIZ_PVROOT__
#define __PVHIVE_WAXES_PICVIZ_PVROOT__

#include <pvhive/PVWax.h>
#include <picviz/PVRoot.h>

DECLARE_WAX(Picviz::PVRoot::select_view)
DECLARE_WAX(Picviz::PVRoot::select_source)
DECLARE_WAX(Picviz::PVRoot::select_scene)

DECLARE_WAX(Picviz::PVRoot::add_correlation)
DECLARE_WAX(Picviz::PVRoot::delete_correlation)
DECLARE_WAX(Picviz::PVRoot::process_correlation)


#endif /* __PVHIVE_WAXES_PICVIZ_PVROOT__ */
