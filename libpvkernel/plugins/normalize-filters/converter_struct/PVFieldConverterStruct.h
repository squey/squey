/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef PVFILTER_PVFIELDCONVERTERSTRUCT_H
#define PVFILTER_PVFIELDCONVERTERSTRUCT_H

#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>

namespace PVFilter
{

class PVFieldConverterStruct : public PVFieldsConverter
{

  public:
	PVFieldConverterStruct();

  public:
	PVCore::PVField& one_to_one(PVCore::PVField& field) override;

	CLASS_FILTER_NOPARAM(PVFilter::PVFieldConverterStruct)
};
}

#endif // PVFILTER_PVFIELDCONVERTERSTRUCT_H
