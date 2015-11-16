/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVMappingFilterFloatDefault.h"

Inendi::PVMappingFilter::decimal_storage_type Inendi::float_mapping::process_utf8(const char* buf, size_t size, PVMappingFilter*)
{
	Inendi::PVMappingFilter::decimal_storage_type ret_ds;
#ifdef NDEBUG
	PV_UNUSED(size);
#else
	assert(buf[size] == '\0');
#endif
	ret_ds.storage_as_float() = strtof(buf, NULL);
	return ret_ds;
}

Inendi::PVMappingFilter::decimal_storage_type Inendi::float_mapping::process_utf16(const uint16_t* buf, size_t size, PVMappingFilter* m)
{
	QString& qstr = static_cast<Inendi::PVMappingFilterFloatDefault*>(m)->th_qs().local();
	qstr.setRawData((QChar const*) buf, size);
	Inendi::PVMappingFilter::decimal_storage_type ret_ds;
	ret_ds.storage_as_float() = qstr.toFloat();
	return ret_ds;
}

IMPL_FILTER_NOPARAM(Inendi::PVMappingFilterFloatDefault)
