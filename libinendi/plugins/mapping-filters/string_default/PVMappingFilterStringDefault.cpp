/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVMappingFilterStringDefault.h"
#include <pvkernel/core/PVStringUtils.h>

Inendi::PVMappingFilterStringDefault::PVMappingFilterStringDefault(PVCore::PVArgumentList const& args):
	PVMappingFilter(),
	_case_sensitive(false)
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

Inendi::PVMappingFilter::decimal_storage_type Inendi::PVMappingFilterStringDefault::process_cell(const char* buf, size_t size)
{
	Inendi::PVMappingFilter::decimal_storage_type ret_ds;
	ret_ds.storage_as_uint() = PVCore::PVStringUtils::compute_str_factor(PVCore::PVUnicodeString((PVCore::PVUnicodeString::utf_char*) buf, size), _case_sensitive);
	return ret_ds;
}

IMPL_FILTER(Inendi::PVMappingFilterStringDefault)
