/**
 * \file PVMappingFilterFloatDefault.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVMAPPINGFILTERFLOAT_H
#define PVFILTER_PVMAPPINGFILTERFLOAT_H

#include <pvkernel/core/general.h>
#include <picviz/PVPureMappingFilter.h>

#include <tbb/enumerable_thread_specific.h>

namespace Picviz {

struct float_mapping
{
	static Picviz::PVMappingFilter::decimal_storage_type process_utf8(const char* buf, size_t size, PVMappingFilter* m);
	static Picviz::PVMappingFilter::decimal_storage_type process_utf16(uint16_t const* buf, size_t size, PVMappingFilter* m);
};

class PVMappingFilterFloatDefault: public PVPureMappingFilter<float_mapping>
{
	friend class float_mapping;
public:
	QString get_human_name() const { return QString("Default"); }
	PVCore::DecimalType get_decimal_type() const override { return PVCore::FloatType; }

protected:
	tbb::enumerable_thread_specific<QString>& th_qs() { return _th_qs; }

private:
	tbb::enumerable_thread_specific<QString> _th_qs;

	CLASS_FILTER(PVMappingFilterFloatDefault)
};

}

#endif
