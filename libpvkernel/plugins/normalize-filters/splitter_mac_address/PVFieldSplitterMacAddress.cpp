/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVFieldSplitterMacAddress.h"
#include <iostream>

const char* PVFilter::PVFieldSplitterMacAddress::UPPERCASE = "uppercase";

/******************************************************************************
 * PVFilter::PVFieldSplitterMacAddress::PVFieldSplitterMacAddress
 *****************************************************************************/

PVFilter::PVFieldSplitterMacAddress::PVFieldSplitterMacAddress(PVCore::PVArgumentList const& args) :
	PVFieldsFilter<PVFilter::one_to_many>()
{
	INIT_FILTER(PVFilter::PVFieldSplitterMacAddress, args);
}

/******************************************************************************
 * PVFilter::PVFieldSplitterMacAddress::set_args
 *****************************************************************************/

void PVFilter::PVFieldSplitterMacAddress::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);
	_uppercased = args.at(UPPERCASE).toBool();
}

/******************************************************************************
 * DEFAULT_ARGS_FILTER
 *****************************************************************************/

DEFAULT_ARGS_FILTER(PVFilter::PVFieldSplitterMacAddress)
{
	PVCore::PVArgumentList args;
	args[UPPERCASE] = false;
	return args;
}

/******************************************************************************
 * PVFilter::PVFieldSplitterMacAddress::one_to_many
 *****************************************************************************/

PVCore::list_fields::size_type
PVFilter::PVFieldSplitterMacAddress::one_to_many(PVCore::list_fields &l,
                                                 PVCore::list_fields::iterator it_ins,
                                                 PVCore::PVField &field)
{
	/**
	 * according to http://en.wikipedia.org/wiki/MAC_address, there are 3
	 * different formats for MAC address:
	 * - 01:23:45:67:89:ab
	 * - 01-23-45-67-89-ab
	 * - 0123.4567.89ab
	 */

	PVCore::PVField f(field);
	if(field.size() == 17) {
		char sep = field.begin()[2];
		if ((sep != '-') && (sep != ':')) {
			// malformed: first separator is invalid (must match '[-:]')
			return 0;
		}

		f.set_end(field.begin() + 2);
		l.insert(it_ins, f);

		for (char* begin = field.begin() + 2; begin < field.end(); begin += 3) {
			if (*begin != sep) {
				/* malformed: next separator has changed (must
				 * previously found separator
				 */
				return 0;
			}
			f.set_begin(begin + 1);
			f.set_end(begin + 3);
			l.insert(it_ins, f);
		}
	} else if(field.size() == 14) {
		for(char* begin=field.begin(); begin<field.end(); begin+=2) {
			f.set_begin(begin);
			f.set_end(begin + 2);
			l.insert(it_ins, f);
			size_t i = begin - field.begin() + 2;
			if(i==4 or i==9) {
				if(*(begin + 2) != '.') {
					// malformed: first separator is invalid (must match '[.]')
					return 0;
				}
				begin++;
			}
		}
	} else {
		// Incorrect mac adresse format.
		return 0;
	}

	return 6;
}

/******************************************************************************
 * IMPL_FILTER
 *****************************************************************************/

IMPL_FILTER(PVFilter::PVFieldSplitterMacAddress)
