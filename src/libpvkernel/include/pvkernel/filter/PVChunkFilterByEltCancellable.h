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

#ifndef PVFILTER_PVCHUNKFILTERBYELTCANCELLABLE_H
#define PVFILTER_PVCHUNKFILTERBYELTCANCELLABLE_H

#include <pvkernel/filter/PVChunkFilter.h>   // for PVChunkFilter
#include <pvkernel/filter/PVElementFilter.h> // for PVElementFilter

#include <memory> // for unique_ptr

namespace PVCore
{
class PVTextChunk;
} // namespace PVCore

namespace PVFilter
{

class PVChunkFilterByEltCancellable : public PVChunkFilter
{
  public:
	PVChunkFilterByEltCancellable(std::unique_ptr<PVElementFilter> elt_filter,
	                              float timeout,
	                              bool* cancellation = nullptr);
	PVCore::PVTextChunk* operator()(PVCore::PVTextChunk* chunk) const;

  private:
	std::unique_ptr<PVElementFilter> _elt_filter;

	float _timeout;
	bool* _cancellation;
};
} // namespace PVFilter

#endif
