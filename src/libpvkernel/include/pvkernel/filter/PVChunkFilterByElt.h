/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVFILTER_PVCHUNKFILTERBYELT_H
#define PVFILTER_PVCHUNKFILTERBYELT_H

#include <pvkernel/filter/PVChunkFilter.h>   // for PVChunkFilter
#include <pvkernel/filter/PVElementFilter.h> // for PVElementFilter
#include <algorithm> // for move
#include <memory>    // for unique_ptr
#include <utility>

namespace PVCore
{
class PVTextChunk;
} // namespace PVCore

namespace PVFilter
{

/**
 * Apply filter to split the line from One PVElement to multiple.
 */
class PVChunkFilterByElt : public PVChunkFilter
{
  public:
	explicit PVChunkFilterByElt(std::unique_ptr<PVElementFilter> elt_filter)
	    : _elt_filter(std::move(elt_filter))
	{
	}
	/**
	 * Apply splitting to every elements from this chunk.
	 */
	PVCore::PVTextChunk* operator()(PVCore::PVTextChunk* chunk) const;

  protected:
	std::unique_ptr<PVElementFilter> _elt_filter; // filter to apply for splitting.
};
} // namespace PVFilter

#endif
