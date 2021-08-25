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

#include <pvkernel/filter/PVChunkFilterByElt.h>
#include <pvkernel/core/PVTextChunk.h> // for PVChunk, list_elts
#include <pvkernel/core/PVElement.h>   // for PVElement

#include <cstddef> // for size_t

/******************************************************************************
 *
 * PVFilter::PVChunkFilterByElt::operator()
 *
 *****************************************************************************/
PVCore::PVTextChunk* PVFilter::PVChunkFilterByElt::operator()(PVCore::PVTextChunk* chunk) const
{
	size_t nelts_valid = 0;

	for (auto& elt_ : chunk->elements()) {
		PVCore::PVElement& elt = (*_elt_filter)(*elt_);
		if (elt.valid()) {
			nelts_valid++;
		}
	}

	chunk->set_elts_stat(chunk->elements().size(), nelts_valid);
	return chunk;
}
