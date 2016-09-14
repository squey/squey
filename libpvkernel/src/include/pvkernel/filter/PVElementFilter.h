/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVELEMENTFILTER_H
#define PVFILTER_PVELEMENTFILTER_H

#include <pvkernel/filter/PVFilterFunction.h>
#include <pvkernel/core/PVElement.h>

namespace PVFilter
{

class PVElementFilter : public PVFilterFunctionBase<PVCore::PVElement&, PVCore::PVElement&>
{
  public:
	virtual PVCore::PVElement& operator()(PVCore::PVElement& in) { return in; }

	CLASS_FILTER_NONREG_NOPARAM(PVElementFilter)
};
} // namespace PVFilter

#endif
