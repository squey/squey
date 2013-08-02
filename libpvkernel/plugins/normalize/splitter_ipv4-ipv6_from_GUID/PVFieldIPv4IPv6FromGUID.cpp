/**
 * \file PVFieldIPv4IPv6FromGUID.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */

#include "PVFieldIPv4IPv6FromGUID.h"

/******************************************************************************
 *
 * PVFilter::PVFieldDuplicate::PVFieldIPv4IPv6FromGUID
 *
 *****************************************************************************/
PVFilter::PVFieldIPv4IPv6FromGUID::PVFieldIPv4IPv6FromGUID(PVCore::PVArgumentList const& args) :
	PVFieldsFilter<PVFilter::one_to_many>()
{
	INIT_FILTER(PVFilter::PVFieldIPv4IPv6FromGUID, args);
}

void PVFilter::PVFieldIPv4IPv6FromGUID::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);
	_ipv4 = args["ipv4"].toBool();
	_ipv6 = args["ipv6"].toBool();
}

DEFAULT_ARGS_FILTER(PVFilter::PVFieldIPv4IPv6FromGUID)
{
	PVCore::PVArgumentList args;
	args["ipv4"] = true;
	args["ipv6"] = false;
	return args;
}

/******************************************************************************
 *
 * PVFilter::PVFieldDuplicate::one_to_many
 *
 *****************************************************************************/
PVCore::list_fields::size_type PVFilter::PVFieldIPv4IPv6FromGUID::one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field)
{
	// GUID={7F000001-FFFF-0000-0000-000000000000}
	// IPv4=127.0.0.1
	// IPv6=FFFF-0000-0000-000000000000

	constexpr int field_utf16_max_len = 76;
	constexpr size_t ipv4_hexa_len = 8;
	constexpr size_t ipv4_dec_max_len = 15;
	constexpr size_t ipv6_hexa_len = 28;

	PVCore::list_fields::size_type ret = 0;

	PVCore::PVArgumentList args = FilterT::get_args();

	QString field_str;
	field.get_qstr(field_str);
	int start = 0;
	int end = field_str.size();
	if (field_str.size() == field_utf16_max_len) {
		start++;
		end--;
	}

	bool ipv4_ok = true;
	if (_ipv4) {

		char dec_ipv4[ipv4_dec_max_len];

		PVCore::PVField &ins_f(*l.insert(it_ins, field));

		unsigned int a, b, c, d;
		if (sscanf(field_str.mid(start+1, ipv4_hexa_len).toStdString().c_str(), "%2x%2x%2x%2x", &a, &b, &c, &d) == 4) {
			snprintf(dec_ipv4, ipv4_dec_max_len, "%u.%u.%u.%u", a, b, c, d);

			size_t ip_dec_utf16_len = strlen(dec_ipv4)*2;

			const ushort* dec_ipv4_utf16 = QString::fromAscii(dec_ipv4).utf16();
			ins_f.allocate_new(ip_dec_utf16_len);
			memcpy(ins_f.begin(), dec_ipv4_utf16, ip_dec_utf16_len);
			ins_f.set_end(ins_f.begin() + ip_dec_utf16_len);
		}
		else {
			ins_f.allocate_new(0);
			ins_f.set_end(ins_f.begin());
			ipv4_ok = false;
		}

		ret++;
	}

	if (_ipv6) {
		PVCore::PVField &ins_f(*l.insert(it_ins, field));
		if (_ipv4 && ipv4_ok) {
			ins_f.allocate_new(ipv6_hexa_len*2);
			memcpy(ins_f.begin(), field_str.utf16()+start+ipv4_hexa_len+2, ipv6_hexa_len*2);
			ins_f.set_end(ins_f.begin() + (ipv6_hexa_len-1)*2);
		}
		else {
			ins_f.allocate_new(0);
			ins_f.set_end(ins_f.begin());
		}
		ret++;
	}

	return ret;
}


IMPL_FILTER(PVFilter::PVFieldIPv4IPv6FromGUID)
