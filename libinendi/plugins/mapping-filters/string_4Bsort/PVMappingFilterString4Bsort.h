/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVMAPPINGFILTERSTRING4BSORT_H
#define PVFILTER_PVMAPPINGFILTERSTRING4BSORT_H

#include <pvkernel/core/general.h>
#include <inendi/PVMappingFilter.h>

#include <tbb/atomic.h>

namespace Inendi {

struct string_mapping
{
	static Inendi::PVMappingFilter::decimal_storage_type process_utf8(const char* buf, size_t size, PVMappingFilter* m);
};

class PVMappingFilterString4Bsort: public PVMappingFilter
{
public:
	Inendi::PVMappingFilter::decimal_storage_type process_cell(const char* buf, size_t size) override
	{
		return string_mapping::process_utf8(buf, size, this);
	}
	QString get_human_name() const { return QString("Pseudo-sort on the first 4 bytes"); }
	PVCore::DecimalType get_decimal_type() const override { return PVCore::UnsignedIntegerType; }

protected:
	CLASS_FILTER(PVMappingFilterString4Bsort)
};

}

#endif	/* PVFILTER_PVMAPPINGFILTERSTRING4BSORT_H */
