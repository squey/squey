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
