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

#ifndef PVFILTER_PVCHUNKFILTERDUMPELTS_H
#define PVFILTER_PVCHUNKFILTERDUMPELTS_H

#include <pvkernel/core/PVTextChunk.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <stddef.h>
#include <map>
#include <string>

namespace PVCore {
class PVTextChunk;
}  // namespace PVCore

namespace PVFilter
{

/**
 * This is a filter which doesn't change the PVChunk but save invalid elements
 * in the QStringList set at contruct time.
 */
class PVChunkFilterDumpElts : public PVChunkFilter
{

  public:
	explicit PVChunkFilterDumpElts(std::map<size_t, std::string>& l);

	PVCore::PVTextChunk* operator()(PVCore::PVTextChunk* chunk);

  protected:
	std::map<size_t, std::string>& _l; //!< List with invalid elements.
};
} // namespace PVFilter

#endif
