/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVMappingFilterHostDefault.h"
#include <inendi/limits.h>
#include <pvkernel/core/PVUnicodeString.h>
#include <pvkernel/core/PVUnicodeString16.h>
#include <pvkernel/core/PVStringUtils.h>
#include <pvkernel/core/network.h>

#include <QVector>
#include <QByteArray>

#include <algorithm>
#include <tbb/concurrent_vector.h>

#include <pvkernel/core/dumbnet.h>

static uint32_t compute_str16_factor(PVCore::PVUnicodeString16 const& str)
{
	char b1_c = 0;
	char b2_c = 0;
	char b3_c = 0;
	char b4_c = 0;

	const size_t len = str.len();
	PVCore::PVUnicodeString16::utf_char const* const buf = str.buffer();
	if (len >= 1) {
		b1_c = QChar(buf[0]).toLatin1();
		if (len >= 2) {
			b2_c = QChar(buf[1]).toLatin1();
			if (len >= 3) {
				b3_c = QChar(buf[2]).toLatin1();
				if (len >= 4) {
					b4_c = QChar(buf[3]).toLatin1();
				}
			}
		}
	}

	return ((uint32_t)(b4_c) << 0)  | ((uint32_t)(b3_c) << 8) |
	       ((uint32_t)(b2_c) << 16) | ((uint32_t)(b1_c) << 24);
}

static uint32_t compute_str_factor(PVCore::PVUnicodeString const& str)
{
	char b1_c = 0;
	char b2_c = 0;
	char b3_c = 0;
	char b4_c = 0;

	const size_t len = str.len();
	PVCore::PVUnicodeString::utf_char const* const buf = str.buffer();
	// TODO: check for UTF8 real chars!
	if (len >= 1) {
		b1_c = buf[0];
		if (len >= 2) {
			b2_c = buf[1];
			if (len >= 3) {
				b3_c = buf[2];
				if (len >= 4) {
					b4_c = buf[3];
				}
			}
		}
	}

	return ((uint32_t)(b4_c) << 0)  | ((uint32_t)(b3_c) << 8) |
	       ((uint32_t)(b2_c) << 16) | ((uint32_t)(b1_c) << 24);
}


Inendi::PVMappingFilterHostDefault::PVMappingFilterHostDefault(PVCore::PVArgumentList const& args):
	PVPureMappingFilter<host_mapping>(),
	_case_sensitive(false) // This will be changed by set_args anyway
{
	INIT_FILTER(PVMappingFilterHostDefault, args);
}

DEFAULT_ARGS_FILTER(Inendi::PVMappingFilterHostDefault)
{
	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey("convert-domain-lowercase", "Convert domain strings to lower case")].setValue<bool>(true);
	return args;
}

void Inendi::PVMappingFilterHostDefault::set_args(PVCore::PVArgumentList const& args)
{
	Inendi::PVMappingFilter::set_args(args);
	_case_sensitive = !args.at("convert-domain-lowercase").toBool();
}

Inendi::PVMappingFilter::decimal_storage_type Inendi::host_mapping::process_utf8(const char* buf, size_t size, PVMappingFilter*)
{
	uint32_t ret;
	if (PVCore::Network::ipv4_aton(buf, size, ret)) {
		// That goes to the first half of the space
		ret >>= 1;
	}
	else {
		// Take the first four characters
		ret = compute_str_factor(PVCore::PVUnicodeString(buf, size));
		// That goes to the other half!
		ret = (ret >> 1) | 0x80000000;
	}

	Inendi::PVMappingFilter::decimal_storage_type ret_ds;
	ret_ds.storage_as_uint() = ret;
	return ret_ds;
}

Inendi::PVMappingFilter::decimal_storage_type Inendi::host_mapping::process_utf16(const uint16_t* buf, size_t size, PVMappingFilter*)
{
	uint32_t ret;
	if (PVCore::Network::ipv4_a16ton(buf, size, ret)) {
		// That goes to the first half of the space
		ret >>= 1;
	}
	else {
		// Take the first four characters
		ret = compute_str16_factor(PVCore::PVUnicodeString16(buf, size));
		// That goes to the other half!
		ret = (ret >> 1) | 0x80000000;
	}

	Inendi::PVMappingFilter::decimal_storage_type ret_ds;
	ret_ds.storage_as_uint() = ret;
	return ret_ds;
}

IMPL_FILTER(Inendi::PVMappingFilterHostDefault)
