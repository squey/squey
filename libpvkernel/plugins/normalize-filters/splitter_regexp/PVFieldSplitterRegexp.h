/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVFIELDSPLITTERREGEXP_H
#define PVFILTER_PVFIELDSPLITTERREGEXP_H

#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>

#include <regex>

namespace PVFilter
{

class PVFieldSplitterRegexp : public PVFieldsFilter<one_to_many>
{
  public:
	PVFieldSplitterRegexp(
	    PVCore::PVArgumentList const& args = PVFieldSplitterRegexp::default_args());
	PVFieldSplitterRegexp(const PVFieldSplitterRegexp& src);

  protected:
	PVCore::list_fields::size_type one_to_many(PVCore::list_fields& l,
	                                           PVCore::list_fields::iterator it_ins,
	                                           PVCore::PVField& field);

  public:
	virtual void set_args(PVCore::PVArgumentList const& args);

  protected:
	std::regex _regexp;
	bool _full_line;

	CLASS_FILTER(PVFilter::PVFieldSplitterRegexp)
};
}

#endif
