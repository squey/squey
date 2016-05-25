/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef PVFILTER_PVFIELDCONVERTERSUBSTITUTION_H
#define PVFILTER_PVFIELDCONVERTERSUBSTITUTION_H

#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>

#include <unordered_map>

namespace PVFilter
{

class PVFieldConverterStruct : public PVFieldsConverter
{

  public:
	PVFieldConverterStruct(
	    PVCore::PVArgumentList const& args = PVFieldConverterStruct::default_args());

  public:
	void set_args(PVCore::PVArgumentList const& args) override;
	PVCore::PVField& one_to_one(PVCore::PVField& field) override;

	CLASS_FILTER(PVFilter::PVFieldConverterStruct)
};
}

#endif // PVFILTER_PVFIELDCONVERTERSUBSTITUTION_H
