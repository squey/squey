/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2019
 */

#ifndef __PVERFSOURCE_FILE_H__
#define __PVERFSOURCE_FILE_H__

#include <iterator>
#include <fcntl.h>
#include <memory>

#include <QString>

#include <pvkernel/core/PVTextChunk.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVInput.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/rush/PVRawSourceBase.h>
#include <pvkernel/core/serialize_numbers.h>

#include "../../common/erf/PVERFDescription.h"
#include "../../common/erf/PVERFAPI.h"
#include "PVERFSource.h"
#include "PVERFBinaryChunk.h"

#include <rapidjson/pointer.h>

namespace PVRush
{

class PVERFSource : public PVRawSourceBaseType<PVCore::PVBinaryChunk>
{
  private:
	static constexpr const size_t MEGA = 1024 * 1024;
	static constexpr const size_t CHUNK_ELEMENT_COUNT = 1000000;

  public:
	PVERFSource(PVInputDescription_p input);

	QString human_name() override { return "ERF"; }
	void seek_begin() override {}
	void prepare_for_nelts(chunk_index /*nelts*/) override {}
	size_t get_size() const override { return _source_row_count * MEGA * _files_path.size(); }
	PVCore::PVBinaryChunk* operator()() override;

  private:
	ERF_INT add_entityresults(ERF_INT state_id,
	                          size_t states_count,
	                          const rapidjson::Value* entityresults,
	                          std::vector<std::vector<PVERFAPI::int_t>>& ids,
	                          std::vector<std::vector<PVERFAPI::float_t>>& results);

	void add_inputs_dict(PVCore::PVBinaryChunk* chunk);

	size_t compute_source_row_count(const rapidjson::Value* entities);

  private:
	PVRush::PVERFDescription* _input_desc;
	QStringList _files_path;
	size_t _current_file_index = 0;

	// source
	ERF_INT _source_start_row = 0;
	ERF_INT _source_row_count = std::numeric_limits<ERF_INT>::max();

	// states
	std::vector<std::pair<size_t, size_t>> _selected_states;
	decltype(_selected_states)::iterator _current_states_range;
	ERF_INT _state_start_row = 0;
	ERF_INT _state_row_count = 0;
	size_t _range_index = 0;
	size_t _states_count = 0;
	size_t _state_id = 0;
	std::vector<ERF_INT> _state_ids;

	PVRush::PVERFAPI _erf;
	const rapidjson::Document& _selected_nodes;
	std::vector<std::vector<ERF_INT>> _ids;

	PVCore::PVBinaryChunk* _first_chunk = nullptr;
	size_t _row_count_by_chunk = 1 /*+1*/;

	std::vector<size_t> _node_list;
};

} // namespace PVRush

#endif // __PVERFSOURCE_FILE_H__
