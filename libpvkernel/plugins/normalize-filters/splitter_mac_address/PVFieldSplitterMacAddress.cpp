/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include "PVFieldSplitterMacAddress.h"

/******************************************************************************
 * PVFilter::PVFieldSplitterMacAddress::PVFieldSplitterMacAddress
 *****************************************************************************/

PVFilter::PVFieldSplitterMacAddress::PVFieldSplitterMacAddress()
    : PVFieldsFilter<PVFilter::one_to_many>()
{
	INIT_FILTER_NOPARAM(PVFilter::PVFieldSplitterMacAddress);
}

/******************************************************************************
 * PVFilter::PVFieldSplitterMacAddress::one_to_many
 *****************************************************************************/

PVCore::list_fields::size_type PVFilter::PVFieldSplitterMacAddress::one_to_many(
    PVCore::list_fields& l, PVCore::list_fields::iterator it_ins, PVCore::PVField& field)
{
	/**
	 * according to http://en.wikipedia.org/wiki/MAC_address, there are 3
	 * different formats for MAC address:
	 * - 01:23:45:67:89:ab
	 * - 01-23-45-67-89-ab
	 * - 0123.4567.89ab
	 */

	char* txt = field.begin();
	PVCore::PVField f(field);
	if (field.size() == 17) {
		char sep = txt[2];
		if ((sep != '-') and (sep != ':')) {
			// malformed: first separator is invalid (must match '[-:]')
			return 0;
		}
		// std::all on range
		for (size_t i = 5; i <= 14; i += 3) {
			if (txt[i] != sep) {
				// no consistent separator
				return 0;
			}
		}

		f.set_end(field.begin() + 8);
		l.insert(it_ins, f);

		f.set_begin(field.begin() + 9);
		f.set_end(field.begin() + 17);
		l.insert(it_ins, f);
	} else if (field.size() == 14) {
		if (txt[4] != '.' or txt[9] != '.') {
			// Invalid format
			return 0;
		}

		f.set_end(field.begin() + 7);
		l.insert(it_ins, f);

		f.set_begin(field.begin() + 7);
		f.set_end(field.begin() + 14);
		l.insert(it_ins, f);
	} else {
		// Incorrect mac adresse format.
		return 0;
	}

	return 6;
}

/******************************************************************************
 * IMPL_FILTER_NOPARAM
 *****************************************************************************/

IMPL_FILTER_NOPARAM(PVFilter::PVFieldSplitterMacAddress)
