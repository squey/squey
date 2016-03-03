/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include "PVMappingFilterString4Bsort.h"

/**
 * Return a values which sort strings based on theirs 4 first chars.
 *
 * @warning this is a duplication from host mapping.
 */
static uint32_t compute_str_factor(const char* str, size_t len)
{
	uint32_t res = *reinterpret_cast<const uint32_t*>(str);
	switch(len) {
		case 0:
			return 0;
		case 1:
			return res & 0xFF000000;
		case 2:
			return res & 0xFFFF0000;
		case 3:
			return res & 0xFFFFFF00;
		default:
			return res;
	}
}

Inendi::PVMappingFilter::decimal_storage_type Inendi::PVMappingFilterString4Bsort::process_cell(const char* buf, size_t size)
{
	Inendi::PVMappingFilter::decimal_storage_type ret_ds;
	ret_ds.storage_as_uint() = compute_str_factor(buf, size);;
	return ret_ds;
}

IMPL_FILTER_NOPARAM(Inendi::PVMappingFilterString4Bsort)
