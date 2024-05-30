//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include "PVFieldSplitterMacAddress.h"

/******************************************************************************
 * PVFilter::PVFieldSplitterMacAddress::PVFieldSplitterMacAddress
 *****************************************************************************/

PVFilter::PVFieldSplitterMacAddress::PVFieldSplitterMacAddress()
    : PVFieldsFilter<PVFilter::one_to_many>()
{
	INIT_FILTER_NOPARAM(PVFilter::PVFieldSplitterMacAddress);
	set_number_expected_fields(2);
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

	return 2;
}
