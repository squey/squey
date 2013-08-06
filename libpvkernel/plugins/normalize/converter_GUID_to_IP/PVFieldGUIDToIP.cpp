/**
 * \file PVFieldGUIDToIP.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */

#include "PVFieldGUIDToIP.h"

/******************************************************************************
 *
 * PVFilter::PVFieldGUIDToIP::PVFieldGUIDToIP
 *
 *****************************************************************************/
PVFilter::PVFieldGUIDToIP::PVFieldGUIDToIP(PVCore::PVArgumentList const& args) :
	PVFieldsConverter()
{
	INIT_FILTER(PVFilter::PVFieldGUIDToIP, args);
}

void PVFilter::PVFieldGUIDToIP::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);
	_ipv6 = args["ipv6"].toBool();
}

DEFAULT_ARGS_FILTER(PVFilter::PVFieldGUIDToIP)
{
	PVCore::PVArgumentList args;
	args["ipv6"] = false;
	return args;
}

/******************************************************************************
 *
 * PVFilter::PVFieldGUIDToIP::one_to_one
 *
 *****************************************************************************/
PVCore::PVField& PVFilter::PVFieldGUIDToIP::one_to_one(PVCore::PVField& field)
{
	// GUID={58BEFDF8-400C-0C00-0000-000000000071}
	// IPv4=88.190.253.248
	// IPv6=58be:fdf8:400c:0c00:0000:0000:0000:0071

	constexpr int field_max_len = 38;
	constexpr int field_utf16_max_len = field_max_len*2;
	constexpr size_t ipv4_hexa_len = 8;
	constexpr size_t ipv4_dec_max_len = 15;
	constexpr size_t ipv6_hexa_len = 32+7;

	PVCore::PVArgumentList args = FilterT::get_args();

	QString field_str;
	field.get_qstr(field_str);
	int start = 0;
	if (field_str.size() == field_max_len) {
		start += 2;
	}

	if (!_ipv6) { //ipv4
		char dec_ipv4[ipv4_dec_max_len];

		unsigned int a, b, c, d;
		if (sscanf(field_str.mid(start, ipv4_hexa_len).toStdString().c_str(), "%2X%2X%2X%2X", &a, &b, &c, &d) == 4) {
			snprintf(dec_ipv4, ipv4_dec_max_len, "%u.%u.%u.%u", a, b, c, d);

			size_t ip_dec_utf16_len = strlen(dec_ipv4)*2;

			const ushort* dec_ipv4_utf16 = QString::fromAscii(dec_ipv4).utf16();
			field.allocate_new(ip_dec_utf16_len);
			memcpy(field.begin(), dec_ipv4_utf16, ip_dec_utf16_len);
			field.set_end(field.begin() + ip_dec_utf16_len);
		}
		else {
			field.allocate_new(0);
			field.set_end(field.begin());
		}
	}
	else { //ipv6
		constexpr size_t ipv6_hexa_utf16_len = ipv6_hexa_len*2;
		char ipv6_hexa[ipv6_hexa_len];
		unsigned int a, b, c, d, e, f, g, h;

		if (sscanf(field_str.mid(start, ipv6_hexa_len).toStdString().c_str(), "%4X%4X-%4X-%4X-%4X-%4X%4X%4X", &a, &b, &c, &d, &e, &f, &g, &h) == 8) {
			snprintf(ipv6_hexa, ipv6_hexa_len+1, "%04x:%04x:%04x:%04x:%04x:%04x:%04x:%04x", a, b, c, d, e, f, g, h);
			const ushort* ipv6_hexa_utf16 = QString::fromAscii(ipv6_hexa).utf16();
			field.allocate_new(ipv6_hexa_utf16_len);
			memcpy(field.begin(), ipv6_hexa_utf16, ipv6_hexa_utf16_len);
			field.set_end(field.begin() + ipv6_hexa_utf16_len);
		}
		else {
			field.allocate_new(0);
			field.set_end(field.begin());
		}
	}

	return field;
}


IMPL_FILTER(PVFilter::PVFieldGUIDToIP)
