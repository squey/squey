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

#include <pvkernel/filter/PVChunkFilterDumpElts.h>

/******************************************************************************
 *
 * PVFilter::PVChunkFilterDumpElts::PVChunkFilterDumpElts
 *
 *****************************************************************************/
PVFilter::PVChunkFilterDumpElts::PVChunkFilterDumpElts(std::map<size_t, std::string>& l)
    : PVChunkFilter(), _l(l)
{
}

/******************************************************************************
 *
 * PVFilter::PVChunkFilterDumpElts::operator()
 *
 *****************************************************************************/
PVCore::PVTextChunk* PVFilter::PVChunkFilterDumpElts::operator()(PVCore::PVTextChunk* chunk)
{
	// TODO : This should be const
	for (PVCore::PVElement* elt : chunk->elements()) {
		if (not elt->valid()) {
			_l.emplace(std::piecewise_construct, std::forward_as_tuple(elt->get_elt_agg_index()),
			           std::forward_as_tuple(elt->begin(), elt->size()));
		}
	}

	return chunk;
}
