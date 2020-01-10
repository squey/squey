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

#include "../../common/erf/PVERFDescription.h"
#include "../../common/erf/PVERFAPI.h"
#include "PVERFSource.h"
#include "PVERFBinaryChunk.h"

///
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
///

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
	static constexpr const size_t CHUNK_ROW_COUNT = 100000; // TODO : CHUNK_ELEM_COUNT
	static constexpr const size_t MEGA = 1024 * 1024;

  public:
	PVERFSource(PVInputDescription_p input)
	    : _input_desc(dynamic_cast<PVRush::PVERFDescription*>(input.get()))
	    , _erf(_input_desc->path().toStdString())
	    , _selected_nodes(_input_desc->current_source_selected_nodes())

	{
		rapidjson::StringBuffer buffer;
		buffer.Clear();
		rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
		_selected_nodes.Accept(writer);
		pvlogger::fatal() << buffer.GetString() << std::endl;

		assert(
		    not _selected_nodes.IsNull()); // edit PVERFDescription::split_selected_nodes_by_sources
	}
	virtual ~PVERFSource() {}

	QString human_name() override { return "ERF"; }
	void seek_begin() override {}
	void prepare_for_nelts(chunk_index nelts) override {}       // FIXME
	size_t get_size() const override { return 1102236 * MEGA; } // FIXME
	PVCore::PVBinaryChunk* operator()() override
	{
		// TODO : DRY
		if (_starting_row >= _total_row_count) {
			return nullptr;
		}

		ERF_INT row_count = 0;

		PVCore::PVBinaryChunk* chunk = nullptr;
		std::vector<std::vector<PVERFAPI::int_t>> ids;

		const rapidjson::Value* constant_connectivities =
		    rapidjson::Pointer("/post/constant/connectivities").Get(_selected_nodes);
		if (constant_connectivities) {
			std::vector<std::vector<PVERFAPI::int_t>> results;

			EString entity_type;

			ErfElementIList element_list;
			ErfErrorCode Status = _erf.stage()->GetElementList(0, element_list);
			for (int i = 0; i < element_list.size(); i++) {
				ErfElementI* element = element_list[i];

				// idele
				std::vector<PVERFAPI::int_t> idele;
				Status = element->ReadIds(idele);
				ids.emplace_back(std::move(idele));

				// pid
				std::vector<PVERFAPI::int_t> pid;
				Status = element->ReadPartIds(pid);
				ids.emplace_back(std::move(pid));

				// ic
				ERF_INT node_per_elem;
				ERF_INT dim_count;
				element->ReadHeader(row_count, node_per_elem, dim_count);
				row_count *= node_per_elem;
				_total_row_count = row_count;

				expand(ids, node_per_elem);

				std::vector<PVERFAPI::int_t> values;
				Status = element->ReadConnectivities(values);

				results.emplace_back(std::move(values));
			}

			chunk = new PVRush::PVERFBinaryChunk<PVERFAPI::int_t>(
			    std::move(ids), std::move(results), row_count, _starting_row);
		}

		const rapidjson::Value* constant_entityresults =
		    rapidjson::Pointer("/post/constant/entityresults").Get(_selected_nodes);
		if (constant_entityresults) {
			std::vector<std::vector<PVERFAPI::float_t>> results;
			row_count = add_entityresults(0, constant_entityresults, ids, results);

			chunk = new PVRush::PVERFBinaryChunk<PVERFAPI::float_t>(
			    std::move(ids), std::move(results), row_count, _starting_row);
		}

		const rapidjson::Value* singlestate_entityresults =
		    rapidjson::Pointer("/post/singlestate/entityresults").Get(_selected_nodes);
		if (singlestate_entityresults) {
			std::vector<std::vector<PVERFAPI::float_t>> results;
			row_count = add_entityresults(1, singlestate_entityresults, ids, results);

			pvlogger::info() << "OK" << std::endl;

			chunk = new PVRush::PVERFBinaryChunk<PVERFAPI::float_t>(
			    std::move(ids), std::move(results), row_count, _starting_row);
		}

		_starting_row += row_count;

		return chunk;
	}

  private:
	ERF_INT add_entityresults(ERF_INT StateId,
	                          const rapidjson::Value* entityresults,
	                          std::vector<std::vector<PVERFAPI::int_t>>& ids,
	                          std::vector<std::vector<PVERFAPI::float_t>>& results)
	{
		// bool entid = false;
		EString entity_type;
		ERF_INT row_count = 0;

		for (const auto& entity_type : entityresults->GetObject()) {
			const std::string& entity_type_name = entity_type.name.GetString();

			for (const auto& entity_group : entity_type.value.GetArray()) {
				const std::string& entity_group_name = entity_group.GetString();

				std::vector<EString> zones;
				_erf.stage()->GetContourZones(StateId, ENTITY_RESULT, entity_type_name,
				                              entity_group_name, zones);
				for (const std::string& zone : zones) {

					ErfResultIPtr result = nullptr;
					ErfErrorCode Status = _erf.stage()->GetContourResult(
					    StateId, ENTITY_RESULT, entity_type_name, entity_group_name, zone, result);

					EString entity_type;
					ERF_INT node_per_elem;
					result->ReadHeader(entity_type, row_count, node_per_elem);
					_total_row_count = row_count;

					ERF_LLONG row_counts[] = {row_count, node_per_elem};
					ERF_LLONG starting_rows[] = {_starting_row, 0};
					std::vector<PVERFAPI::float_t> values;

					Status = result->ReadResultSelectiveValues(ERF_SEL_TYPE_HYPERSLAB, row_counts,
					                                           starting_rows, 0, nullptr, values);

					add_to_results(
					    results, std::move(values),
					    node_per_elem); // De-interlace multi-dimentional arrays if needed

					/*if (not entid) {
					    _ids.emplace_back(std::move(ids));
					    chunk->set_raw_column_chunk(
					        _col_count++,
					        (void*)(_ids.back().data()),
					        row_count, sizeof(ERF_INT), PVERFAPI::int_type);
					    entid = true;
					}*/
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
	size_t _starting_row = 0;
	ERF_INT _total_row_count = std::numeric_limits<ERF_INT>::max();
	PVRush::PVERFAPI _erf;
	const rapidjson::Document& _selected_nodes;
	// std::vector<std::vector<ERF_INT>> _ids;
};

} // namespace PVRush

#endif // __PVERFSOURCE_FILE_H__
