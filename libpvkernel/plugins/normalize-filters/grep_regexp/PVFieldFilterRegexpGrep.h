/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVFIELDFILTERREGEXPGREP_H
#define PVFILTER_PVFIELDFILTERREGEXPGREP_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVFieldsFilter.h>
#include <regex>

namespace PVFilter
{

class PVFieldFilterRegexpGrep : public PVFieldsFilter<one_to_one>
{
  public:
	PVFieldFilterRegexpGrep(
	    PVCore::PVArgumentList const& args = PVFieldFilterRegexpGrep::default_args());
	void set_args(PVCore::PVArgumentList const& args) override;

  public:
	PVCore::PVField& one_to_one(PVCore::PVField& obj) override;

  private:
	std::regex _rx;
	bool _inverse;

	CLASS_FILTER(PVFilter::PVFieldFilterRegexpGrep)
};
}

#endif
