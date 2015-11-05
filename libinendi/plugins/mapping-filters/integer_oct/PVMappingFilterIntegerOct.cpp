/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVMappingFilterIntegerOct.h"
#include <tbb/enumerable_thread_specific.h>

#include <QString>

Inendi::PVMappingFilterIntegerOct::PVMappingFilterIntegerOct(PVCore::PVArgumentList const& args):
	PVPureMappingFilter<integer_mapping>(),
	_signed(true) // This will be changed by set_args anyway
{
	INIT_FILTER(PVMappingFilterIntegerOct, args);
}

DEFAULT_ARGS_FILTER(Inendi::PVMappingFilterIntegerOct)
{
	PVCore::PVArgumentList args;
	//args[PVCore::PVArgumentKey("signed", "Signed integers")].setValue<bool>(true);
	return args;
}

void Inendi::PVMappingFilterIntegerOct::set_args(PVCore::PVArgumentList const& args)
{
	Inendi::PVMappingFilter::set_args(args);
}

PVCore::DecimalType Inendi::PVMappingFilterIntegerOct::get_decimal_type() const
{
	//return (_signed) ? PVCore::IntegerType : PVCore::UnsignedIntegerType;
	return PVCore::UnsignedIntegerType;
}

Inendi::PVMappingFilter::decimal_storage_type Inendi::integer_mapping::process_utf8(const char* buf, const size_t /*size*/, PVMappingFilter*)
{
	char* end;
	Inendi::PVMappingFilter::decimal_storage_type ret_ds;
	ret_ds.storage_as_uint() = strtoul(buf, &end, 8);

	return ret_ds;
}

Inendi::PVMappingFilter::decimal_storage_type Inendi::integer_mapping::process_utf16(const uint16_t* buf, size_t size, PVMappingFilter*)
{
	static tbb::enumerable_thread_specific<QString, tbb::scalable_allocator<QString>, tbb::ets_key_per_instance> tls_qs;

	Inendi::PVMappingFilter::decimal_storage_type ret_ds;
	QString& local_qs = tls_qs.local();
	local_qs.setRawData((const QChar*) buf, size);
	bool ok;
	ret_ds.storage_as_uint() = local_qs.toULong(&ok, 8);

	return ret_ds;
}

IMPL_FILTER_NOPARAM(Inendi::PVMappingFilterIntegerOct)
