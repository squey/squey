/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVFIELDFILTERGREP_H
#define PVFILTER_PVFIELDFILTERGREP_H

#include <pvkernel/filter/PVFieldsFilter.h>

#include <QString>

namespace PVFilter
{

class PVFieldFilterGrep : public PVFieldsFilter<one_to_one>
{
  public:
	PVFieldFilterGrep(PVCore::PVArgumentList const& args = PVFieldFilterGrep::default_args());

  public:
	void set_args(PVCore::PVArgumentList const& args) override;

  public:
	PVCore::PVField& one_to_one(PVCore::PVField& obj) override;

  protected:
	QString _str;
	bool _inverse;

	CLASS_FILTER(PVFilter::PVFieldFilterGrep)
};
}

#endif
