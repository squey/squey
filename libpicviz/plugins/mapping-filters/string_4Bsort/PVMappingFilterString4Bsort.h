/**
 * \file PVMappingFilterString4Bsort.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVMAPPINGFILTERSTRING4BSORT_H
#define PVFILTER_PVMAPPINGFILTERSTRING4BSORT_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/stdint.h>
#include <picviz/PVPureMappingFilter.h>

#include <tbb/atomic.h>

namespace Picviz {

struct string_mapping
{
	static Picviz::PVMappingFilter::decimal_storage_type process_utf8(const char* buf, size_t size, PVMappingFilter* m);
	static Picviz::PVMappingFilter::decimal_storage_type process_utf16(uint16_t const* buf, size_t size, PVMappingFilter* m);
};

class PVMappingFilterString4Bsort: public PVPureMappingFilter<string_mapping>
{
public:
	QString get_human_name() const { return QString("Pseudo-sort on the first 4 bytes"); }
	PVCore::DecimalType get_decimal_type() const override { return PVCore::UnsignedIntegerType; }

protected:
	CLASS_FILTER(PVMappingFilterString4Bsort)
};

}

#endif	/* PVFILTER_PVMAPPINGFILTERSTRING4BSORT_H */
