/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVELEMENTFILTERBYFIELDS_H
#define PVFILTER_PVELEMENTFILTERBYFIELDS_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVElementFilter.h>
#include <pvkernel/filter/PVFilterFunction.h>
#include <pvkernel/filter/PVFieldsFilter.h>
#include <pvkernel/core/PVElement.h>

namespace PVFilter
{

class PVElementFilterByFields : public PVElementFilter
{
  public:
	PVElementFilterByFields(PVFieldsBaseFilter_f fields_f);

  public:
	PVCore::PVElement& operator()(PVCore::PVElement& elt);

  protected:
	PVFieldsBaseFilter_f _ff;

	CLASS_FILTER_NONREG_NOPARAM(PVElementFilterByFields)
};
}

#endif
