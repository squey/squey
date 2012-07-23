/**
 * \file PVTFViewRowFilterFastIPv4.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PICVIZ_PVTFVIEWROWFILTERFASTIPV4_H
#define PICVIZ_PVTFVIEWROWFILTERFASTIPV4_H

#include <pvkernel/core/general.h>
#include <picviz/PVSelection.h>
#include <picviz/PVTFViewRowFiltering.h>

#include <QList>

namespace Picviz {

class PVView;

class LibPicvizDecl PVTFViewRowFilterFastIPv4: public PVTFViewRowFiltering
{
public:
	PVSelection operator()(PVView const& view_src, PVView const& view_dst, PVSelection const& sel_org) const;
};

}

#endif
