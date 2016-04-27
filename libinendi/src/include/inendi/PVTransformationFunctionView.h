/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVTRANSFORMATIONFUNCTIONVIEW_H
#define INENDI_PVTRANSFORMATIONFUNCTIONVIEW_H

#include <pvkernel/core/general.h>
#include <inendi/PVSelection.h>

namespace Inendi
{

class PVView;

/*! \brief Interface for view's selection transformation
 */
class PVTransformationFunctionView
{
  public:
	PVTransformationFunctionView() {}

  public:
	virtual void pre_process(PVView const& view_src, PVView const& view_dst) = 0;

	virtual PVSelection operator()(PVView const& view_org, PVView const& view_dst,
	                               PVSelection const& sel_org) const = 0;
};
}

#endif
