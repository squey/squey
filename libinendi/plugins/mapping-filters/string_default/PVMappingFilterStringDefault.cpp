/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVMappingFilterStringDefault.h"
#include <pvkernel/core/PVTBBMaxArray.h>
#include <pvkernel/core/PVStringUtils.h>

#include <tbb/parallel_reduce.h>

#include <omp.h>


Inendi::PVMappingFilterStringDefault::PVMappingFilterStringDefault(PVCore::PVArgumentList const& args):
	PVMappingFilter(),
	_case_sensitive(true) // This will be changed by set_args anyway
{
	INIT_FILTER(PVMappingFilterStringDefault, args);
}

DEFAULT_ARGS_FILTER(Inendi::PVMappingFilterStringDefault)
{
	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey("convert-lowercase", "Convert strings to lower case")].setValue<bool>(false);
	return args;
}

void Inendi::PVMappingFilterStringDefault::set_args(PVCore::PVArgumentList const& args)
{
	Inendi::PVMappingFilter::set_args(args);
	_case_sensitive = !args.at("convert-lowercase").toBool();
}

Inendi::PVMappingFilter::decimal_storage_type Inendi::string_mapping::process_utf16(const uint16_t* buf, size_t size, PVMappingFilter* m)
{
	Inendi::PVMappingFilter::decimal_storage_type ret_ds;
	ret_ds.storage_as_uint() = (uint32_t) PVCore::PVStringUtils::compute_str_factor16(buf, size, static_cast<PVMappingFilterStringDefault*>(m)->case_sensitive());
	return ret_ds;
}

Inendi::PVMappingFilter::decimal_storage_type Inendi::string_mapping::process_utf8(const char* buf, size_t size, PVMappingFilter* m)
{
	Inendi::PVMappingFilter::decimal_storage_type ret_ds;
	ret_ds.storage_as_uint() = (uint32_t) PVCore::PVStringUtils::compute_str_factor(PVCore::PVUnicodeString((PVCore::PVUnicodeString::utf_char*) buf, size), static_cast<PVMappingFilterStringDefault*>(m)->case_sensitive());
	return ret_ds;
}

IMPL_FILTER(Inendi::PVMappingFilterStringDefault)
