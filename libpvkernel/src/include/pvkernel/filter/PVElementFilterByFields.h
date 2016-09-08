/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVELEMENTFILTERBYFIELDS_H
#define PVFILTER_PVELEMENTFILTERBYFIELDS_H

#include <pvkernel/filter/PVElementFilter.h>
#include <pvkernel/filter/PVFieldsFilter.h>

#include <vector>

namespace PVCore
{
class PVElement;
}

namespace PVFilter
{

class PVElementFilterByFields : public PVElementFilter
{
  public:
	PVCore::PVElement& operator()(PVCore::PVElement& elt) override;
	void add_filter(PVFieldsBaseFilter_p&& f) { _ff.push_back(f); }

  protected:
	std::vector<PVFieldsBaseFilter_p> _ff;

	CLASS_FILTER_NONREG_NOPARAM(PVElementFilterByFields)
};
}

#endif
