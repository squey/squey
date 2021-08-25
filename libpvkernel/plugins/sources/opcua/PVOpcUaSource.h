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

#ifndef PVOPCUASOURCE_FILE_H
#define PVOPCUASOURCE_FILE_H

#include <QString>

#include <pvkernel/rush/PVRawSourceBase.h>
#include <pvkernel/rush/PVInputDescription.h>

#include "../../common/opcua/PVOpcUaAPI.h"
#include "PVOpcUaBinaryChunk.h"

namespace PVRush
{

class PVOpcUaQuery;

class PVOpcUaSource : public PVRawSourceBaseType<PVCore::PVBinaryChunk>
{
  public:
	PVOpcUaSource(PVInputDescription_p input);
	virtual ~PVOpcUaSource() = default;

  public:
	QString human_name() override;
	void seek_begin() override;
	void prepare_for_nelts(chunk_index nelts) override;
	size_t get_size() const override;
	PVOpcUaBinaryChunk* operator()() override;

  protected:
	chunk_index _next_index;

	void setup_query();

	void download_interval();
	void fill_sourcetime_interval(UA_DateTime start_time, std::vector<bool> const& has_data);

	void download_full();
	auto consolidate_datetimes_full() const -> std::vector<UA_DateTime>;
	void consolidate_values_full(std::vector<UA_DateTime> const& consolidated_datetimes);
	void fill_sourcetime_full(std::vector<UA_DateTime> const& datetimes);

  private:
	PVOpcUaQuery& _query;
	PVOpcUaAPI _api;
	std::vector<std::unique_ptr<PVOpcUaBinaryChunk>> _chunks;

	struct NodeData
	{
		std::vector<uint8_t> values;
		std::vector<UA_DateTime> datetimes;
		size_t read_index = 0;
		bool has_more = true;
		UA_DataType const* type = nullptr;
		std::vector<uint8_t> chunk_values;
	};

	std::vector<NodeData> _data;
	std::vector<boost::posix_time::ptime> _sourcetimes;
	std::vector<QString> _node_ids;

	unsigned int _current_chunk = 0;
	uint64_t _nodes_count = 0;
	size_t _sourcetimes_current = 0;
	size_t _chunk_size = 1000;

	UA_DateTime _query_start = 0;
	UA_DateTime _query_end = 0;
	int64_t _query_interval = 0;
	size_t _query_nb_of_times = 0;
};
} // namespace PVRush

#endif
