/**
 * \file PVMappingFilterHostDefault.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVMappingFilterHostDefault.h"
#include <picviz/limits.h>
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


Picviz::PVMappingFilterHostDefault::PVMappingFilterHostDefault(PVCore::PVArgumentList const& args):
	PVPureMappingFilter<host_mapping>(),
	_case_sensitive(false) // This will be changed by set_args anyway
{
	INIT_FILTER(PVMappingFilterHostDefault, args);
}

DEFAULT_ARGS_FILTER(Picviz::PVMappingFilterHostDefault)
{
	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey("convert-domain-lowercase", "Convert domain strings to lower case")].setValue<bool>(true);
	return args;
}

void Picviz::PVMappingFilterHostDefault::set_args(PVCore::PVArgumentList const& args)
{
	Picviz::PVMappingFilter::set_args(args);
	_case_sensitive = !args["convert-lowercase"].toBool();
}

Picviz::PVMappingFilter::decimal_storage_type Picviz::host_mapping::process_utf8(const char* buf, size_t size, PVMappingFilter*)
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

	Picviz::PVMappingFilter::decimal_storage_type ret_ds;
	ret_ds.storage_as_uint() = ret;
	return ret_ds;
}

Picviz::PVMappingFilter::decimal_storage_type Picviz::host_mapping::process_utf16(const uint16_t* buf, size_t size, PVMappingFilter*)
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

	Picviz::PVMappingFilter::decimal_storage_type ret_ds;
	ret_ds.storage_as_uint() = ret;
	return ret_ds;
}

IMPL_FILTER(Picviz::PVMappingFilterHostDefault)
