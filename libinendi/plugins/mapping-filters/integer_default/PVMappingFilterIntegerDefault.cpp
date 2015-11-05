/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVMappingFilterIntegerDefault.h"

Inendi::PVMappingFilterIntegerDefault::PVMappingFilterIntegerDefault(bool signed_,
                                                                     PVCore::PVArgumentList const& args):
	PVPureMappingFilter<integer_mapping>(),
	_signed(signed_)
{
	INIT_FILTER(PVMappingFilterIntegerDefault, args);
}

DEFAULT_ARGS_FILTER(Inendi::PVMappingFilterIntegerDefault)
{
	PVCore::PVArgumentList args;
	return args;
}

void Inendi::PVMappingFilterIntegerDefault::set_args(PVCore::PVArgumentList const& args)
{
	Inendi::PVMappingFilter::set_args(args);
}

PVCore::DecimalType Inendi::PVMappingFilterIntegerDefault::get_decimal_type() const
{
	return (_signed) ? PVCore::IntegerType : PVCore::UnsignedIntegerType;
}

Inendi::PVMappingFilter::decimal_storage_type Inendi::integer_mapping::process_utf8(const char* buf, const size_t size, PVMappingFilter* m)
{
	ssize_t i = 0;
	Inendi::PVMappingFilter::decimal_storage_type ret_ds;
	ret_ds.storage_as_uint() = 0;
	while (isspace(buf[i]) && i < (ssize_t) size) {
		i++;
	}

	ssize_t start = i;
	uint32_t pow10 = 1;
	const char first_char = buf[start];
	bool set_negative = false;
	if (first_char == '+' || first_char == '-') {
		start++;
		if (first_char == '-') {
			if (static_cast<PVMappingFilterIntegerDefault*>(m)->is_signed()) {
				set_negative = true;
			}
			else {
				return ret_ds;
			}
		}
	}

	for (i = size-1; i >= start; i--) {
		const char c = buf[i];
		if (isspace(c)) {
			continue;
		}
		if (c >= '0' && c <= '9') {
			ret_ds.storage_as_uint() += pow10*(c-'0');
			pow10 *= 10;
		}
		else {
			break;
		}
	}

	if (set_negative) {
		ret_ds.storage_as_int() = -(ret_ds.storage_as_int());
	}

	return ret_ds;
}

Inendi::PVMappingFilter::decimal_storage_type Inendi::integer_mapping::process_utf16(const uint16_t* buf, size_t size, PVMappingFilter* m)
{
	ssize_t i = 0;
	Inendi::PVMappingFilter::decimal_storage_type ret_ds;
	ret_ds.storage_as_uint() = 0;
	for (; i < (ssize_t) size; i++) {
		const uint16_t c = buf[i];
		if ((c & 0xFF00) != 0) {
			return ret_ds;
		}
		if (!isspace((char)c)) {
			break;
		}
	}

	ssize_t start = i;
	uint32_t pow10 = 1;
	const uint16_t first_char = buf[start];
	if ((first_char & 0xFF00) != 0) {
		return ret_ds;
	}
	bool set_negative = false;
	if (first_char == '+' || first_char == '-') {
		start++;
		if (first_char == '-') {
			if (static_cast<PVMappingFilterIntegerDefault*>(m)->is_signed()) {
				set_negative = true;
			}
			else {
				return ret_ds;
			}
		}
	}
	for (i = size-1; i >= start; i--) {
		const uint16_t c = buf[i];
		if ((c & 0xFF00) != 0) {
			return ret_ds;
		}
		if (isspace(c)) {
			continue;
		}
		if (c >= '0' && c <= '9') {
			ret_ds.storage_as_uint() += pow10*(c-'0');
			pow10 *= 10;
		}
		else {
			break;
		}
	}

	if (set_negative) {
		ret_ds.storage_as_int() = -(ret_ds.storage_as_int());
	}

	return ret_ds;
}

IMPL_FILTER_NOPARAM(Inendi::PVMappingFilterIntegerDefault)
