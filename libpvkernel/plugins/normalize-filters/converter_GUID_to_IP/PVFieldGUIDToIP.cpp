/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include "PVFieldGUIDToIP.h"

/******************************************************************************
 *
 * PVFilter::PVFieldGUIDToIP::PVFieldGUIDToIP
 *
 *****************************************************************************/
PVFilter::PVFieldGUIDToIP::PVFieldGUIDToIP(PVCore::PVArgumentList const& args) : PVFieldsConverter()
{
	INIT_FILTER(PVFilter::PVFieldGUIDToIP, args);
}

void PVFilter::PVFieldGUIDToIP::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);
	_ipv6 = args.at("ipv6").toBool();
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

	constexpr int field_max_len = 38; // With bracet
	constexpr size_t ipv4_dec_max_len = 15;
	constexpr size_t ipv6_hexa_len = 32 + 7;

	char* txt = field.begin();
	if (field.size() == field_max_len) {
		// Skip bracet
		txt++;
	}

	if (not _ipv6) { // ipv4
		unsigned int a, b, c, d;
		if (sscanf(txt, "%2x%2x%2x%2x", &a, &b, &c, &d) == 4) {
			field.allocate_new(ipv4_dec_max_len + 1);
			int end = snprintf(field.begin(), ipv4_dec_max_len + 1, "%u.%u.%u.%u", a, b, c, d);
			field.set_end(field.begin() + end);
		} else {
			// Invalid GUID Format
			field.set_invalid();
		}
	} else { // ipv6
		unsigned int a, b, c, d, e, f, g, h;

		if (sscanf(txt, "%4X%4X-%4X-%4X-%4X-%4X%4X%4X", &a, &b, &c, &d, &e, &f, &g, &h) == 8) {
			field.allocate_new(ipv6_hexa_len + 1);
			int end = snprintf(field.begin(), ipv6_hexa_len + 1,
			                   "%04x:%04x:%04x:%04x:%04x:%04x:%04x:%04x", a, b, c, d, e, f, g, h);
			field.set_end(field.begin() + end);
		} else {
			field.set_invalid();
		}
	}

	return field;
}
