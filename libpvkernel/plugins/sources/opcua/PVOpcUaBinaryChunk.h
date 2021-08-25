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

#ifndef __PVCORE_PVOPCUABINARYCHUNK__
#define __PVCORE_PVOPCUABINARYCHUNK__

#include <pvkernel/core/PVBinaryChunk.h>
#include "../../common/opcua/PVOpcUaAPI.h"

namespace PVRush
{

class PVOpcUaBinaryChunk : public PVCore::PVBinaryChunk
{
	static constexpr const size_t MEGA = 1024 * 1024;

  public:
	PVOpcUaBinaryChunk(size_t nodes_count, size_t row_count, size_t nraw_start_row, std::vector<boost::posix_time::ptime>&& sourcetimes)
	    : PVCore::PVBinaryChunk(nodes_count + 1, row_count, (PVRow)nraw_start_row),
          _sourcetimes(std::move(sourcetimes))
	{
        set_raw_column_chunk(PVCol(0), _sourcetimes.data(), row_count,
	                         sizeof(boost::posix_time::ptime), "datetime_us");
        _values.resize(nodes_count);
	}

    void set_node_values(size_t node_index, std::vector<uint8_t>&& node_values, UA_DataType const* type)
    {
        _values[node_index] = std::move(node_values); 
        set_raw_column_chunk(PVCol(1 + node_index), _values[node_index].data(), rows_count(),
		                     type->memSize, PVRush::PVOpcUaAPI::pvcop_type(type->typeIndex));
    }

  private:
	std::vector<boost::posix_time::ptime> _sourcetimes;
	std::vector<std::vector<uint8_t>> _values;
};

} // namespace PVRush

#endif // __PVCORE_PVOPCUABINARYCHUNK__
