/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "helpers.h"
#include <iostream>

using PVCore::PVChunk;
using PVCore::PVElement;
using PVCore::PVField;
using PVCore::list_fields;

void dump_chunk_csv(PVChunk& c, std::ostream& out)
{
	for (PVElement* elt : c.elements()) {
		if (not elt->valid()) {
			continue;
		}
		list_fields& l = elt->fields();
		if (l.size() == 1) {
			out << std::string(l.begin()->begin(), l.begin()->size());
		} else {
			list_fields::iterator itf, itfe;
			itfe = l.end();
			itfe--;
			for (itf = l.begin(); itf != itfe; itf++) {
				PVField& f = *itf;
				out << "'" << std::string(f.begin(), f.size()) << "',";
			}
			PVField& f = *itf;
			out << "'" << std::string(f.begin(), f.size()) << "'";
		}
		out << std::endl;
	}
}

void dump_chunk_raw(PVChunk const& c)
{
	for (PVElement const* elt : c.c_elements()) {
		for (PVField const& f : elt->c_fields()) {
			std::cout << std::string(f.begin(), f.size());
		}
		std::cout << std::endl;
	}
}
