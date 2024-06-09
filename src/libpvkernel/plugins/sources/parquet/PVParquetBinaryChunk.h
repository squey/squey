/* * MIT License
 *
 * © Squey, 2024
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

#ifndef __PVCORE_PVPARQUETBINARYCHUNK__
#define __PVCORE_PVPARQUETBINARYCHUNK__

#include <pvkernel/core/PVBinaryChunk.h>
#include <arrow/record_batch.h>
#include <arrow/api.h>
#include <arrow/util/type_fwd.h>
#include <stddef.h>
#include <memory>
#include <vector>

#include "../../common/parquet/PVParquetAPI.h"
#include "pvcop/db/types.h"

namespace arrow {
class Table;
}  // namespace arrow
namespace pvcop {
namespace db {
class write_dict;
}  // namespace db
}  // namespace pvcop

namespace PVRush
{

class PVParquetBinaryChunk : public PVCore::PVBinaryChunk
{
  public:
	static constexpr const size_t MEGA = 1024 * 1024;

  public:
	PVParquetBinaryChunk(
		bool multi_inputs,
		bool is_bit_optimizable,
		size_t input_index,
		std::shared_ptr<arrow::Table>& table,
		const std::vector<size_t>& column_indexes,
		std::vector<pvcop::db::write_dict*>& dicts,
		size_t row_count,
		size_t nraw_start_row
	);

  private:
	std::vector<std::vector<void*>> _values;
	std::vector<pvcop::db::index_t> _input_index;
};

} // namespace PVRush

#endif // __PVCORE_PVPARQUETBINARYCHUNK__
