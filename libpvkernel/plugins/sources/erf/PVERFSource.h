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

template <typename T>
void add_to_results(std::vector<std::vector<T>>& results,
                    std::vector<T>&& todemux,
                    int node_per_elem)
{
	results.emplace_back(std::move(todemux));

	if (node_per_elem > 1) { // De-interlace multi-dimentional arrays
		const std::vector<T>& orig = results.back();
		size_t row_count = orig.size() / node_per_elem;
		std::vector<std::vector<T>> demux;
		demux.resize(node_per_elem);
		for (int i = 0; i < node_per_elem; i++) {
			demux[i] = std::vector<T>(row_count);
		}
		for (size_t i = 0, c = 0; i < orig.size(); i += node_per_elem, c++) {
			for (int j = 0; j < node_per_elem; j++) {
				demux[j][c] = orig[i + j];
			}
		}
		results.pop_back();
		for (int i = 0; i < node_per_elem; i++) {
			results.emplace_back(std::move(demux[i]));
		}
	}
}

class PVERFSource : public PVRawSourceBaseType<PVCore::PVBinaryChunk>
{
  private:
	static constexpr const size_t MEGA = 1024 * 1024;
	static constexpr const size_t CHUNK_ELEMENT_COUNT = 1000000;

  public:
	PVERFSource(PVInputDescription_p input)
	    : _input_desc(dynamic_cast<PVRush::PVERFDescription*>(input.get()))
	    , _files_path(_input_desc->paths())
	    , _erf(_files_path.front().toStdString())
	    , _selected_nodes(_input_desc->selected_nodes())

	{
		// edit PVERFDescription::split_selected_nodes_by_sources
		assert(not _selected_nodes.IsNull());

		_erf.stage()->GetStateIds(_state_ids);

		auto get_node_list = [this](const std::string& json_pointer) {
			const rapidjson::Value* node_list =
			    rapidjson::Pointer(json_pointer.c_str()).Get(_selected_nodes);
			if (node_list) {
				const std::string& node_list_str = node_list->GetString();
				if (not node_list_str.empty())
					try {
						_node_list = PVCore::deserialize_numbers_as_values(node_list_str);
					} catch (...) {
					}
				_state_row_count = _node_list.size();
			}
		};
		get_node_list("/post/constant/entityresults/NODE/list");
		get_node_list("/post/singlestate/entityresults/NODE/list");

		_first_chunk = operator()(); // extract first chunk to have stats
		if (_first_chunk) {
			_row_count_by_chunk = CHUNK_ELEMENT_COUNT / _first_chunk->columns_count();
		}
	}

	QString human_name() override { return "ERF"; }
	void seek_begin() override {}
	void prepare_for_nelts(chunk_index nelts) override {}
	size_t get_size() const override { return _source_row_count * MEGA * _files_path.size(); }
	PVCore::PVBinaryChunk* operator()() override
	{
		if (_first_chunk != nullptr) {
			PVCore::PVBinaryChunk* first_chunk = _first_chunk;
			_first_chunk = nullptr;
			return first_chunk;
		}

		if (_source_start_row >= _source_row_count) {
			if (_current_file_index == _files_path.size() - 1) {
				return nullptr;
			} else { // load next file
				_erf.set_path(_files_path[++_current_file_index].toStdString());
				_source_start_row = _state_start_row = _range_index = 0;
				_current_states_range = _selected_states.begin();
				_state_id = _current_states_range->first;
			}
		}

		size_t nraw_start_row = (_current_file_index * _source_row_count) + _source_start_row;
		ERF_INT row_count = 0;

		PVCore::PVBinaryChunk* chunk = nullptr;

		const rapidjson::Value* constant_connectivities =
		    rapidjson::Pointer("/post/constant/connectivities").Get(_selected_nodes);
		if (constant_connectivities) {
			std::vector<std::vector<PVERFAPI::int_t>> results;
			_ids.clear();

			EString entity_type;

			ErfElementIList element_list;
			ErfErrorCode Status = _erf.stage()->GetElementList(0, element_list);
			for (int i = 0; i < element_list.size(); i++) {
				ErfElementI* element = element_list[i];

				// idele
				std::vector<PVERFAPI::int_t> idele;
				Status = element->ReadIds(idele);
				_ids.emplace_back(std::move(idele));

				// pid
				std::vector<PVERFAPI::int_t> pid;
				Status = element->ReadPartIds(pid);
				_ids.emplace_back(std::move(pid));

				// ic
				ERF_INT node_per_elem;
				ERF_INT dim_count;
				element->ReadHeader(row_count, node_per_elem, dim_count);
				row_count *= node_per_elem;
				_source_row_count = row_count;

				expand(_ids, node_per_elem);

				std::vector<PVERFAPI::int_t> values;
				Status = element->ReadConnectivities(values);

				results.emplace_back(std::move(values));
			}

			chunk = new PVRush::PVERFBinaryChunk<PVERFAPI::int_t>(
			    _files_path.size(), _current_file_index, std::move(_ids), std::move(results),
			    row_count, nraw_start_row);
		}

		const rapidjson::Value* constant_entityresults =
		    rapidjson::Pointer("/post/constant/entityresults").Get(_selected_nodes);
		if (constant_entityresults) {
			std::vector<std::vector<PVERFAPI::float_t>> results;
			row_count = add_entityresults(0, 1, constant_entityresults, _ids, results);

			std::vector<std::vector<PVERFAPI::int_t>> ids;
			ids.emplace_back(_ids[0].begin() + _state_start_row,
			                 _ids[0].begin() + _state_start_row + row_count);

			chunk = new PVRush::PVERFBinaryChunk<PVERFAPI::float_t>(
			    _files_path.size(), _current_file_index, std::move(ids), std::move(results),
			    row_count, nraw_start_row);

			_state_start_row += row_count;
		}

		const rapidjson::Value* singlestate_entityresults =
		    rapidjson::Pointer("/post/singlestate/entityresults").Get(_selected_nodes);
		if (singlestate_entityresults) {
			if (_selected_states.empty()) {
				const rapidjson::Value* singlestate_states =
				    rapidjson::Pointer("/post/singlestate/states").Get(_selected_nodes);
				if (singlestate_states) {
					const std::string& states = singlestate_states->GetString();
					_selected_states = PVCore::deserialize_numbers_as_ranges(states);
					_states_count = PVCore::get_count_from_ranges(_selected_states);
					_current_states_range = _selected_states.begin();
					_state_id = _current_states_range->first + _range_index;
				} else {
					return nullptr;
				}
			}

			std::vector<std::vector<PVERFAPI::float_t>> results;
			row_count = add_entityresults(_state_ids[_state_id], _states_count,
			                              singlestate_entityresults, _ids, results);

			std::vector<std::vector<PVERFAPI::int_t>> ids;
			ids.emplace_back(row_count, _state_ids[_state_id]);
			ids.emplace_back(_ids[0].begin() + _state_start_row,
			                 _ids[0].begin() + _state_start_row + row_count);

			chunk = new PVRush::PVERFBinaryChunk<PVERFAPI::float_t>(
			    _files_path.size(), _current_file_index, std::move(ids), std::move(results),
			    row_count, nraw_start_row);

			if (_state_start_row >= _state_row_count) {
				if (_state_id == _current_states_range->second) {
					_current_states_range++;
					_range_index = 0;
				} else {
					_range_index++;
				}
				_state_start_row = 0;
				_state_id = _current_states_range->first + _range_index;
			} else {
				_state_start_row += row_count;
			}
		}

		_source_start_row += row_count;

		return chunk;
	}

  private:
	ERF_INT add_entityresults(ERF_INT state_id,
	                          size_t states_count,
	                          const rapidjson::Value* entityresults,
	                          std::vector<std::vector<PVERFAPI::int_t>>& ids,
	                          std::vector<std::vector<PVERFAPI::float_t>>& results)
	{
		EString entity_type;
		size_t row_count = 0;
		bool entid_init = false;

		for (const auto& entity_type : entityresults->GetObject()) {
			const std::string& entity_type_name = entity_type.name.GetString();

			const auto& entity_groups =
			    entity_type_name == "NODE" ? entity_type.value["groups"] : entity_type.value;

			for (const auto& entity_group : entity_groups.GetArray()) {
				const std::string& entity_group_name = entity_group.GetString();

				std::vector<EString> zones;
				_erf.stage()->GetContourZones(state_id, ENTITY_RESULT, entity_type_name,
				                              entity_group_name, zones);
				for (const std::string& zone : zones) {

					ErfResultIPtr result = nullptr;
					ErfErrorCode Status = _erf.stage()->GetContourResult(
					    state_id, ENTITY_RESULT, entity_type_name, entity_group_name, zone, result);

					EString entity_type;
					ERF_INT node_per_elem;
					ERF_INT state_row_count;
					result->ReadHeader(entity_type, state_row_count, node_per_elem);
					if (_node_list.empty()) {
						_state_row_count = state_row_count;
					}
					// FIXME disable if count or max is invalid
					if (_source_row_count == std::numeric_limits<ERF_INT>::max()) {
						_source_row_count = _state_row_count * states_count;
					}
					row_count = std::min(_row_count_by_chunk,
					                     (size_t)(_state_row_count - _state_start_row));

					if (_state_start_row == 0 and not entid_init) {
						std::vector<PVERFAPI::int_t> entid;
						Status = result->ReadIds(entid);

						ids.clear();
						ids.emplace_back(std::move(entid));
						entid_init = true;
					}

					std::vector<PVERFAPI::float_t> values;
					if (_node_list.empty()) {
						ERF_LLONG row_counts[] = {row_count, node_per_elem};
						ERF_LLONG start_rows[] = {_state_start_row, 0};

						Status = result->ReadResultSelectiveValues(
						    ERF_SEL_TYPE_HYPERSLAB, row_counts, start_rows, 0, nullptr, values);
					} else {
						std::vector<long long int> pl(row_count * 2 * node_per_elem);
						for (size_t i = 0; i < row_count; i++) {
							for (size_t j = 0; j < node_per_elem; j++) {
								pl[(i * node_per_elem + j) * 2] =
								    (_node_list[_state_start_row + i] - 1) + 0;
								pl[(i * node_per_elem + j) * 2 + 1] = j; // node_per_elem indice
							}
						}

						Status = result->ReadResultSelectiveValues(ERF_SEL_TYPE_POINTLIST, 0, 0,
						                                           row_count * node_per_elem,
						                                           pl.data(), values);
					}

					// De-interlace multi-dimentional arrays if needed
					add_to_results(results, std::move(values), node_per_elem);
				}
			}
		}

		return row_count;
	}

	void expand(std::vector<std::vector<PVERFAPI::int_t>>& ids, size_t node_per_elem)
	{
		for (std::vector<PVERFAPI::int_t>& id : ids) {
			std::vector<PVERFAPI::int_t> new_id(id.size() * node_per_elem);
#pragma omp parallel for
			for (size_t i = 0; i < id.size(); i++) {
				for (size_t j = 0; j < node_per_elem; j++) {
					new_id[(i * node_per_elem) + j] = id[i];
				}
			}
			std::swap(id, new_id);
		}
	}

  private:
	PVRush::PVERFDescription* _input_desc;
	QStringList _files_path;
	size_t _current_file_index = 0;

	// source
	size_t _source_start_row = 0;
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
