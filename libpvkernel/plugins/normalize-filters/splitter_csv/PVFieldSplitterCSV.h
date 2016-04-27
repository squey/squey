/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef PVFILTER_PVFIELDSPLITTERCSV_FILE_H
#define PVFILTER_PVFIELDSPLITTERCSV_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>
#include <pvkernel/filter/PVFieldsFilterParamWidget.h>

namespace PVFilter
{

class PVFieldSplitterCSV : public PVFieldsFilter<one_to_many>
{
  public:
	PVFieldSplitterCSV(PVCore::PVArgumentList const& args = PVFieldSplitterCSV::default_args());

  public:
	PVCore::list_fields::size_type one_to_many(PVCore::list_fields& l,
	                                           PVCore::list_fields::iterator it_ins,
	                                           PVCore::PVField& field);
	void set_args(PVCore::PVArgumentList const& args);

  private:
	char _sep;
	char _quote;

	CLASS_FILTER(PVFilter::PVFieldSplitterCSV)
};
}

#endif
