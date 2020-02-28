/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
 */

#ifndef PVOPCUASOURCE_FILE_H
#define PVOPCUASOURCE_FILE_H

#include <QString>

#include "boost/date_time/posix_time/posix_time.hpp"

#include <pvkernel/rush/PVRawSourceBase.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/core/PVBinaryChunk.h>

#include "../../common/opcua/PVOpcUaAPI.h"

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
	PVCore::PVBinaryChunk* operator()() override;

  protected:
	chunk_index _next_index;

	void setup_query();

	void fill_sourcetime(UA_DateTime start_time, std::vector<bool> const& has_data);

  private:
	PVOpcUaQuery& _query;
	PVOpcUaAPI _api;
	std::vector<std::unique_ptr<PVCore::PVBinaryChunk>> _chunks;
	std::vector<std::pair<std::vector<uint8_t>, UA_DataType const*>> _data;
	std::vector<boost::posix_time::ptime> _sourcetimes;
	std::vector<QString> _node_ids;

	unsigned int _current_chunk = 0;
	uint64_t _nodes_count = 0;
	size_t _sourcetimes_current = 0;

	UA_DateTime _query_start;
	UA_DateTime _query_end;
	int64_t _query_interval;
	size_t _query_nb_of_times;
};
} // namespace PVRush

#endif
