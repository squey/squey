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

#ifndef __PVCORE_PVCHUNK__
#define __PVCORE_PVCHUNK__

#include "pvbase/types.h" // for chunk_index, PVCol, PVRow

namespace PVRush
{
class PVAggregator;
}

namespace PVCore
{

class PVChunk
{
	friend class PVRush::PVAggregator;

  public:
	virtual ~PVChunk() = default;

	/**
	 * Set size from input stored in this chunk
	 */
	void set_init_size(size_t size) { _init_size = size; }
	size_t get_init_size() const { return _init_size; }

	virtual size_t rows_count() const = 0;
	virtual void set_elements_index(){};
	virtual void remove_nelts_front(size_t /*nelts_remove*/, size_t /*nstart_rem*/ = 0) {}

	virtual void free() = 0;

  protected:
	chunk_index _index = 0;
	chunk_index _agg_index = 0;
	size_t _init_size = 0; //!< Data quantity loaded from the input. (metrics depend on source kind)
};

} // namespace PVCore

#endif // __PVCORE_PVCHUNK__
