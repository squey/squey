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

#include "helpers.h"
#include <iostream>

using PVCore::list_fields;
using PVCore::PVElement;
using PVCore::PVField;
using PVCore::PVTextChunk;

void dump_chunk_csv(PVTextChunk& c, std::ostream& out)
{
	for (PVElement* elt : c.elements()) {
		if (not elt->valid() or elt->filtered()) {
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
