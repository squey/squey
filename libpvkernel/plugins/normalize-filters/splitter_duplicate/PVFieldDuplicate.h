/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVFIELDDUPLICATE_H
#define PVFILTER_PVFIELDDUPLICATE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>

namespace PVFilter
{

class PVFieldDuplicate : public PVFieldsFilter<one_to_many>
{

  public:
	PVFieldDuplicate(PVCore::PVArgumentList const& args = PVFieldDuplicate::default_args());

  public:
	void set_args(PVCore::PVArgumentList const& args);
	PVCore::list_fields::size_type one_to_many(PVCore::list_fields& l,
	                                           PVCore::list_fields::iterator it_ins,
	                                           PVCore::PVField& field);

  private:
	size_t _n;

	CLASS_FILTER(PVFilter::PVFieldDuplicate)
};
}

#endif // PVFILTER_PVFIELDDUPLICATE_H
