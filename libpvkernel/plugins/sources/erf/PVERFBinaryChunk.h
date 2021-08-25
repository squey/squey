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

#ifndef __PVCORE_PVERFBINARYCHUNK__
#define __PVCORE_PVERFBINARYCHUNK__

#include <pvkernel/core/PVBinaryChunk.h>
#include "../../common/erf/PVERFAPI.h"

namespace PVRush
{

template <typename T>
class PVERFBinaryChunk : public PVCore::PVBinaryChunk
{
	static constexpr const size_t MEGA = 1024 * 1024;

  public:
	PVERFBinaryChunk(size_t files_count,
	                 size_t file_index,
	                 std::vector<std::vector<PVERFAPI::int_t>>&& ids,
	                 std::vector<std::vector<T>>&& results,
	                 size_t row_count,
	                 size_t nraw_start_row)
	    : PVCore::PVBinaryChunk(
	          (files_count > 1) + ids.size() + results.size(), row_count, (PVRow)nraw_start_row)
	    , _ids(std::move(ids))
	    , _results(std::move(results))
	{
		PVCol col_count(0);

		if (files_count > 1) {
			_input_index = std::vector<pvcop::db::index_t>(row_count, (pvcop::db::index_t)file_index);
			set_raw_column_chunk(col_count++, (void*)(_input_index.data()), row_count,
			                     sizeof(pvcop::db::index_t), erf_type_traits<pvcop::db::index_t>::string);
		}
		for (const auto& id : _ids) {
			set_raw_column_chunk(col_count++, (void*)(id.data()), row_count,
			                     sizeof(PVERFAPI::int_t), erf_type_traits<PVERFAPI::int_t>::string);
		}
		for (const auto& result : _results) {
			set_raw_column_chunk(col_count++, (void*)(result.data()), row_count, sizeof(T),
			                     erf_type_traits<T>::string);
		}

		set_init_size(row_count * MEGA);
	}

  private:
	std::vector<pvcop::db::index_t> _input_index;
	std::vector<std::vector<PVERFAPI::int_t>> _ids;
	std::vector<std::vector<T>> _results;
};

} // namespace PVRush

#endif // __PVCORE_PVERFBINARYCHUNK__
