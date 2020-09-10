/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2019
 */

#include "PVERFSource.h"

PVRush::PVERFSource::PVERFSource(PVInputDescription_p input)
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

static void expand(std::vector<PVRush::PVERFAPI::int_t>& values, size_t node_per_elem)
{
	std::vector<PVRush::PVERFAPI::int_t> new_values(values.size() * node_per_elem);
#pragma omp parallel for
	for (size_t i = 0; i < values.size(); i++) {
		for (size_t j = 0; j < node_per_elem; j++) {
			new_values[(i * node_per_elem) + j] = values[i];
		}
	}
	std::swap(values, new_values);
}

PVCore::PVBinaryChunk* PVRush::PVERFSource::operator()()
{
	if (_first_chunk != nullptr) {
		PVCore::PVBinaryChunk* first_chunk = _first_chunk;
		_first_chunk = nullptr;
		return first_chunk;
	}

	if (_source_start_row >= _source_row_count) {
		if (_current_file_index == (size_t)_files_path.size() - 1) {
			return nullptr;
		} else { // load next file
			_erf.set_path(_files_path[++_current_file_index].toStdString());
			_source_start_row = _state_start_row = _range_index = 0;
			if (not _selected_states.empty()) {
				_current_states_range = _selected_states.begin();
				_state_id = _current_states_range->first;
			}
		}
	}

	size_t nraw_start_row = (_current_file_index * _source_row_count) + _source_start_row;
	ERF_INT row_count = 0;

	PVCore::PVBinaryChunk* chunk = nullptr;

	const rapidjson::Value* constant_connectivities =
	    rapidjson::Pointer("/post/constant/connectivities").Get(_selected_nodes);
	if (constant_connectivities) {
		std::vector<std::vector<PVERFAPI::int_t>> results;
		results.resize(1);
		_ids.clear();
		_ids.resize(3);
		std::unique_ptr<pvcop::db::write_dict> types_dict(new pvcop::db::write_dict);

		size_t connectivity_type_id = 0;
		ErfElementIList element_list;
		ErfErrorCode status = _erf.stage()->GetElementList(0, element_list);
		std::vector<std::string> connectivity_types;
		_erf.stage()->GetElementTypes(0, connectivity_types);

		for (const auto& connectivity_type : constant_connectivities->GetArray()) {
			const std::string& connectivity_type_name = connectivity_type.GetString();

			auto it = std::find(connectivity_types.begin(), connectivity_types.end(), connectivity_type_name);
			assert(it != connectivity_types.end());
			size_t connectivity_type_index = std::distance(connectivity_types.begin(), it);
			ErfElementI* element = element_list[connectivity_type_index];

			types_dict->insert(connectivity_type_name.c_str());

			// idele
			std::vector<PVERFAPI::int_t> idele;
			status = element->ReadIds(idele);

			// pid
			std::vector<PVERFAPI::int_t> pid;
			status = element->ReadPartIds(pid);

			// nodeid
			ERF_INT node_per_elem;
			ERF_INT dim_count;
			ERF_INT local_row_count;
			element->ReadHeader(local_row_count, node_per_elem, dim_count);
			local_row_count *= node_per_elem;
			row_count += local_row_count;

			// type
			std::vector<PVERFAPI::int_t> type(local_row_count, (PVERFAPI::int_t)connectivity_type_id);

			std::vector<PVERFAPI::int_t> nodeids;
			status = element->ReadConnectivities(nodeids);

			expand(idele, node_per_elem);
			expand(pid, node_per_elem);

			_ids[0].insert(_ids[0].end(), idele.begin(), idele.end());
			_ids[1].insert(_ids[1].end(), type.begin(), type.end());
			_ids[2].insert(_ids[2].end(), pid.begin(), pid.end());
			results[0].insert(results[0].end(), nodeids.begin(), nodeids.end());

			(void)status;
			connectivity_type_id++;
		}

		_source_row_count = row_count;

		chunk = new PVRush::PVERFBinaryChunk<PVERFAPI::int_t>(
			_files_path.size(), _current_file_index, std::move(_ids), std::move(results), row_count,
			nraw_start_row);


		if (_current_file_index == 0) {
			chunk->set_column_dict(PVCol(1 + (_files_path.size() > 1)), std::move(types_dict));

			add_inputs_dict(chunk);
		}
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
		    _files_path.size(), _current_file_index, std::move(ids), std::move(results), row_count,
		    nraw_start_row);

		if (_current_file_index == 0) {
			add_inputs_dict(chunk);
		}

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
		    _files_path.size(), _current_file_index, std::move(ids), std::move(results), row_count,
		    nraw_start_row);

		if (_current_file_index == 0) {
			add_inputs_dict(chunk);
		}

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

template <typename T>
static void
add_to_results(std::vector<std::vector<T>>& results, std::vector<T>&& todemux, int node_per_elem)
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

size_t PVRush::PVERFSource::compute_source_row_count(const rapidjson::Value* entities)
{
	auto state_row_count = [&](size_t state_id, const rapidjson::Value* entities) {
		EString entity_type;
		ERF_INT row_count = 0;
		for (const auto& entity_type : entities->GetObject()) {
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
					ErfErrorCode status = _erf.stage()->GetContourResult(
						state_id, ENTITY_RESULT, entity_type_name, entity_group_name, zone, result);

					EString entity_type;
					ERF_INT node_per_elem;
					ERF_INT local_row_count;
					result->ReadHeader(entity_type, local_row_count, node_per_elem);
					if (row_count == 0) {
						row_count = local_row_count;
					}
					else if (local_row_count != row_count) {
						throw std::runtime_error(std::string("entid counts differs between '") + entity_type_name + "' entities");
					}
				}
			}
		}

		return row_count;
	};
	

	size_t source_row_count = 0;
	for (const auto& range : _selected_states) {
		size_t begin = range.first;
		size_t end = range.second;
		for (size_t state_id = begin; state_id <= end; state_id++) {
			source_row_count += state_row_count(_state_ids[state_id], entities);
		}
	}

	return source_row_count;
}

ERF_INT PVRush::PVERFSource::add_entityresults(ERF_INT state_id,
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
				ErfErrorCode status = _erf.stage()->GetContourResult(
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
					if (state_id == 0) {
						_source_row_count = _state_row_count * states_count;
					}
					else {
						_source_row_count = compute_source_row_count(entityresults);
					}
				}
				row_count =
				    std::min(_row_count_by_chunk, (size_t)(_state_row_count - _state_start_row));

				if (_state_start_row == 0 and not entid_init) {
					std::vector<PVERFAPI::int_t> entid;
					status = result->ReadIds(entid);

					ids.clear();
					ids.emplace_back(std::move(entid));
					entid_init = true;
				}

				std::vector<PVERFAPI::float_t> values;
				if (_node_list.empty()) {
					ERF_LLONG row_counts[] = {(ERF_LLONG)row_count, (ERF_LLONG)node_per_elem};
					ERF_LLONG start_rows[] = {(ERF_LLONG)_state_start_row, 0};

					status = result->ReadResultSelectiveValues(ERF_SEL_TYPE_HYPERSLAB, row_counts,
					                                           start_rows, 0, nullptr, values);
				} else {
					std::vector<long long int> pl(row_count * 2 * node_per_elem);
					for (size_t i = 0; i < row_count; i++) {
						for (ERF_INT j = 0; j < node_per_elem; j++) {
							pl[(i * node_per_elem + j) * 2] =
							    (_node_list[_state_start_row + i] - 1) + 0;
							pl[(i * node_per_elem + j) * 2 + 1] = j; // node_per_elem indice
						}
					}

					status = result->ReadResultSelectiveValues(
					    ERF_SEL_TYPE_POINTLIST, 0, 0, row_count * node_per_elem, pl.data(), values);
				}

				(void)status;

				// De-interlace multi-dimentional arrays if needed
				add_to_results(results, std::move(values), node_per_elem);
			}
		}
	}

	return row_count;
}

void PVRush::PVERFSource::add_inputs_dict(PVCore::PVBinaryChunk* chunk)
{
	// TODO : remove common path part
	if (_files_path.size() > 1) {

		//std::vector<std::string> col_name_hierarchy;
		//boost::split(col_name_hierarchy, col, boost::is_any_of("/"));

		std::unique_ptr<pvcop::db::write_dict> inputs_dict(new pvcop::db::write_dict);
		for (const QString& input_name : _files_path) {
			inputs_dict->insert(input_name.toStdString().c_str());
		}
		chunk->set_column_dict(PVCol(0), std::move(inputs_dict));
	}
}
