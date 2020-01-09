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

		const rapidjson::Value* constant_connectivities =
		    rapidjson::Pointer("/post/constant/connectivities").Get(_selected_nodes);
		if (constant_connectivities) {
			std::vector<std::vector<PVERFAPI::int_t>> results;

			EString entity_type;

			ErfElementIList element_list;
			ErfErrorCode Status = _erf.stage()->GetElementList(0, element_list);
			for (int i = 0; i < element_list.size(); i++) {
				ErfElementI* element = element_list[i];

				ERF_INT node_per_elem;
				ERF_INT dim_count;
				element->ReadHeader(row_count, node_per_elem, dim_count);
				_total_row_count = row_count;

				std::vector<PVERFAPI::int_t> res;
				Status = element->ReadConnectivities(res);

				results.emplace_back(std::move(res));

				if (node_per_elem > 1) { // De-interlace multi-dimentional arrays
					const std::vector<PVERFAPI::int_t>& orig = results.back();
					std::vector<std::vector<PVERFAPI::int_t>> demux;
					demux.resize(node_per_elem);
					for (int i = 0; i < node_per_elem; i++) {
						demux[i] = std::vector<PVERFAPI::int_t>(row_count);
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

			chunk = new PVRush::PVERFBinaryChunk<PVERFAPI::int_t>(std::move(results), row_count,
			                                                      _starting_row);
		}

		const rapidjson::Value* constant_entityresults =
		    rapidjson::Pointer("/post/constant/entityresults").Get(_selected_nodes);
		if (constant_entityresults) {
			std::vector<std::vector<PVERFAPI::float_t>> results;
			// bool entid = false;
			EString entity_type;

			std::vector<ErfNodeIPtr> vListOfNodeIPtr;
			ErfErrorCode Status = _erf.stage()->GetNodeList(0, vListOfNodeIPtr);
			for (int i = 0; i < vListOfNodeIPtr.size(); i++) {
				ErfNodeI* node_group = vListOfNodeIPtr[i];

				EString entity_type;
				ERF_INT node_per_elem;
				node_group->ReadHeader(entity_type, _total_row_count, node_per_elem);
				row_count = std::min(CHUNK_ROW_COUNT, _total_row_count - _starting_row);
			}

			for (const auto& entity_type : constant_entityresults->GetObject()) {
				const std::string& entity_type_name = entity_type.name.GetString();

				for (const auto& entity_group : entity_type.value.GetArray()) {
					const std::string& entity_group_name = entity_group.GetString();

					std::vector<EString> zones;
					_erf.stage()->GetContourZones(0, ENTITY_RESULT, entity_type_name,
					                              entity_group_name, zones);
					for (const std::string& zone : zones) {

						ErfResultIPtr result = nullptr;
						Status = _erf.stage()->GetContourResult(0, ENTITY_RESULT, entity_type_name,
						                                        entity_group_name, zone, result);

						EString entity_type;
						ERF_INT total_row_count;
						ERF_INT node_per_elem;
						result->ReadHeader(entity_type, total_row_count, node_per_elem);

						ERF_LLONG row_counts[] = {row_count, node_per_elem};
						ERF_LLONG starting_rows[] = {_starting_row, 0};
						std::vector<PVERFAPI::float_t> res;

						Status = result->ReadResultSelectiveValues(
						    ERF_SEL_TYPE_HYPERSLAB, row_counts, starting_rows, 0, nullptr, res);

						results.emplace_back(std::move(res));

						if (node_per_elem > 1) { // De-interlace multi-dimentional arrays
							const std::vector<PVERFAPI::float_t>& orig = results.back();
							std::vector<std::vector<PVERFAPI::float_t>> demux;
							demux.resize(node_per_elem);
							for (int i = 0; i < node_per_elem; i++) {
								demux[i] = std::vector<PVERFAPI::float_t>(row_count);
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

			chunk = new PVRush::PVERFBinaryChunk<PVERFAPI::float_t>(std::move(results), row_count,
			                                                        _starting_row);
		}

		_starting_row += row_count;

		return chunk;
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
