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

#include <pvkernel/rush/PVRawSourceBase.h> // for PVRawSourceBase

#include "pvkernel/core/PVElement.h" // for PVElement
#include <pvkernel/core/PVTextChunk.h>

#include "pvbase/types.h" // for PVCol, PVRow

#include <cstddef> // for size_t
#include <cstdint> // for uintptr_t

void PVCore::PVTextChunk::init_elements_fields()
{
	PVCol nfields_src = _source->get_number_cols_to_reserve() + PVCol(2);
	PVRow nelts = _elts.size();
	allocate_fields_buffer(nelts, nfields_src);
	void* chunk_fields = _p_chunk_fields;
	size_t buffer_size_for_elt = nfields_src * sizeof(__node_list_field);
	for (PVElement* elt : _elts) {
		elt->init_fields(chunk_fields, buffer_size_for_elt);
		chunk_fields = (void*)((uintptr_t)chunk_fields + buffer_size_for_elt);
	}
}
