/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVFIELDSMAPPINGFILTER_H
#define PVFILTER_PVFIELDSMAPPINGFILTER_H

#include <pvkernel/filter/PVFieldsFilter.h>

namespace PVFilter
{

class PVFieldsMappingFilter : public PVFieldsFilter<many_to_many>
{
  public:
	PVFieldsMappingFilter(size_t idx, PVFieldsBaseFilter_p func);

  public:
	PVCore::list_fields& many_to_many(PVCore::list_fields& fields);

  protected:
	size_t _idx;
	PVFieldsBaseFilter_p _func;
};
}

#endif
