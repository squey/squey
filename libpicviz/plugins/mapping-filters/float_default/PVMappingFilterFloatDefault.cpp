/**
 * \file PVMappingFilterFloatDefault.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVMappingFilterFloatDefault.h"

Picviz::PVMappingFilter::decimal_storage_type Picviz::float_mapping::process_utf8(const char* buf, size_t size, PVMappingFilter*)
{
	Picviz::PVMappingFilter::decimal_storage_type ret_ds;
#ifdef NDEBUG
	PV_UNUSED(size);
#else
	assert(buf[size] == '\0');
#endif
	ret_ds.storage_as_float() = strtof(buf, NULL);
	return ret_ds;
}

Picviz::PVMappingFilter::decimal_storage_type Picviz::float_mapping::process_utf16(const uint16_t* buf, size_t size, PVMappingFilter* m)
{
	QString& qstr = static_cast<Picviz::PVMappingFilterFloatDefault*>(m)->th_qs().local();
	qstr.setRawData((QChar const*) buf, size);
	Picviz::PVMappingFilter::decimal_storage_type ret_ds;
	ret_ds.storage_as_float() = qstr.toFloat();
	return ret_ds;
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterFloatDefault)
