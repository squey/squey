/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
 */

#ifndef PVOPCUASOURCE_FILE_H
#define PVOPCUASOURCE_FILE_H

#include <QString>

#include <pvkernel/rush/PVRawSourceBase.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/core/PVBinaryChunk.h>

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

  private:
	PVOpcUaQuery& _query;
	std::vector<std::unique_ptr<PVCore::PVBinaryChunk>> _chunks;
	std::vector<int> _data;

	unsigned int _current_chunk = 0;
	uint64_t _nodes_count = 0;

  private:
	void connect_to_server();
};
} // namespace PVRush

#endif
