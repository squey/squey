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

#ifndef PVFILTER_PVCHUNKFILTERREMOVEINVALIDELTS_H
#define PVFILTER_PVCHUNKFILTERREMOVEINVALIDELTS_H

#include <pvkernel/filter/PVChunkFilter.h> // for PVChunkFilter

#include <cstddef> // for size_t

namespace PVCore
{
class PVTextChunk;
} // namespace PVCore

namespace PVFilter
{

/**
 * This Filter remove invalid elements and filtered elements from elements list
 * of this chunk and update agg_index accordingly to save value in a compacted
 * way in the NRaw.
 */
class PVChunkFilterRemoveInvalidElts : public PVChunkFilter
{

  public:
	explicit PVChunkFilterRemoveInvalidElts(bool& job_done);

	PVCore::PVTextChunk* operator()(PVCore::PVTextChunk* chunk);

  protected:
	size_t _current_agg_index;
	bool& _job_done;
};
} // namespace PVFilter

#endif
