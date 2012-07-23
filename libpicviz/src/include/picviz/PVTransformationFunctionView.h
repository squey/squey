/**
 * \file PVTransformationFunctionView.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PICVIZ_PVTRANSFORMATIONFUNCTIONVIEW_H
#define PICVIZ_PVTRANSFORMATIONFUNCTIONVIEW_H

#include <pvkernel/core/general.h>
#include <picviz/PVSelection.h>

namespace Picviz {

class PVView;

/*! \brief Interface for view's selection transformation
 */
class PVTransformationFunctionView
{
public:
	PVTransformationFunctionView() { }

public:
	virtual void pre_process(PVView const& view_src, PVView const& view_dst) = 0;

	virtual PVSelection operator()(PVView const& view_org, PVView const& view_dst, PVSelection const& sel_org) const = 0;
};

}

#endif
