/**
 * \file PVMappingFilterIntegerHex.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVMappingFilterIntegerHex.h"
#include <tbb/enumerable_thread_specific.h>

#include <QString>

Picviz::PVMappingFilterIntegerHex::PVMappingFilterIntegerHex(PVCore::PVArgumentList const& args):
	PVPureMappingFilter<integer_mapping>(),
	_signed(true) // This will be changed by set_args anyway
{
	INIT_FILTER(PVMappingFilterIntegerHex, args);
}

DEFAULT_ARGS_FILTER(Picviz::PVMappingFilterIntegerHex)
{
	PVCore::PVArgumentList args;
	//args[PVCore::PVArgumentKey("signed", "Signed integers")].setValue<bool>(true);
	return args;
}

void Picviz::PVMappingFilterIntegerHex::set_args(PVCore::PVArgumentList const& args)
{
	Picviz::PVMappingFilter::set_args(args);
}

PVCore::DecimalType Picviz::PVMappingFilterIntegerHex::get_decimal_type() const
{
	//return (_signed) ? PVCore::IntegerType : PVCore::UnsignedIntegerType;
	return PVCore::UnsignedIntegerType;
}

Picviz::PVMappingFilter::decimal_storage_type Picviz::integer_mapping::process_utf8(const char* buf, const size_t size, PVMappingFilter* m)
{
	char* end;
	Picviz::PVMappingFilter::decimal_storage_type ret_ds;
	ret_ds.storage_as_uint() = strtoul(buf, &end, 16);

	return ret_ds;
}

Picviz::PVMappingFilter::decimal_storage_type Picviz::integer_mapping::process_utf16(const uint16_t* buf, size_t size, PVMappingFilter* m)
{
	static tbb::enumerable_thread_specific<QString, tbb::scalable_allocator<QString>, tbb::ets_key_per_instance> tls_qs;

	Picviz::PVMappingFilter::decimal_storage_type ret_ds;
	QString& local_qs = tls_qs.local();
	local_qs.setRawData((const QChar*) buf, size);
	bool ok;
	ret_ds.storage_as_uint() = local_qs.toULong(&ok, 16);

	return ret_ds;
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterIntegerHex)
