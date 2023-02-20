//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include "PVERFAPI.h"

constexpr const char input_column_name[] = "filename";

PVRush::PVERFAPI::PVERFAPI(const std::string& erf_path)
{
	// Initialize the ERF library once and for all
	static ErfErrorCode Status = []() { return ErfLibManager::LibInitialize(); }();

	if (Status == ERF_SUCCESS) {
		_lib_Initialized = true;
		set_path(erf_path);
	}
}

bool PVRush::PVERFAPI::set_path(const std::string& erf_path)
{
	if (_lib_Initialized) {
		// Create a filer interface
		ErfErrorCode Status;
		_filer.reset(ErfFactory::CreateFiler(erf_path, ERF_READ, Status));
		if (Status == ERF_SUCCESS and _filer) {
			_stage = _filer->GetStage(_stage_name);
		}
		return true;
	}
	return false;
}

std::vector<std::tuple<rapidjson::Document, std::string, PVRush::PVFormat>>
PVRush::PVERFAPI::get_sources_info(const rapidjson::Document& selected_nodes,
                                   bool multi_inputs) const
{
	auto descs = get_selected_nodes_by_source(selected_nodes);
	auto formats = get_formats_from_selected_nodes(selected_nodes, multi_inputs);

	assert(descs.size() == formats.size());

	std::vector<std::tuple<rapidjson::Document, std::string, PVRush::PVFormat>> infos;
	for (size_t i = 0; i < descs.size(); i++) {
		infos.emplace_back(std::make_tuple(std::move(descs[i].first), std::move(descs[i].second),
		                                   PVRush::PVFormat(formats[i].documentElement())));
	}

	return infos;
}

size_t PVRush::PVERFAPI::memory_size(const rapidjson::Document& selected_nodes,
                                     size_t files_count) const
{
	size_t input_size = files_count > 1 ? sizeof(uint16_t) : 0;

	size_t connectivity_size = 0;
	const rapidjson::Value* connectivities =
	    rapidjson::Pointer("/post/constant/connectivities").Get(selected_nodes);
	if (connectivities) {
		ErfElementIList element_list;
		ErfErrorCode Status = _stage->GetElementList(0, element_list);
		(void) Status;
		for (size_t i = 0; i < element_list.size(); i++) {
			ErfElementI* element = element_list[i];
			ERF_INT row_count;
			ERF_INT node_per_elem;
			ERF_INT dim_count;
			element->ReadHeader(row_count, node_per_elem, dim_count);

			// (input) + "idele", "pid", "entity" + plotted
			connectivity_size +=
			    (row_count * node_per_elem) *
			    ((input_size + 3 * sizeof(int_t)) + (4 + (files_count > 1)) * sizeof(uint32_t));
		}
	}

	auto entityresults_size = [&](const std::string& pointer, size_t state_id) {
		size_t entityresults_size = 0;
		const rapidjson::Value* entityresults =
		    rapidjson::Pointer(pointer.c_str()).Get(selected_nodes);
		if (entityresults) {
			for (const auto& entity_type : entityresults->GetObject()) {
				const std::string& entity_type_name = entity_type.name.GetString();
				const auto& entity_groups =
				    entity_type_name == "NODE" ? entity_type.value["groups"] : entity_type.value;

				size_t nodes_count = 0;
				if (entity_type_name == "NODE") {
					const std::string& list = entity_type.value["list"].GetString();
					try {
						const auto& ranges = PVCore::deserialize_numbers_as_ranges(list);
						nodes_count = PVCore::get_count_from_ranges(ranges);
					} catch (...) {
					}
				}

				ERF_INT state_row_count;
				for (const auto& entity_group : entity_groups.GetArray()) {
					const std::string& entity_group_name = entity_group.GetString();
					std::vector<EString> zones;
					_stage->GetContourZones(state_id, ENTITY_RESULT, entity_type_name,
					                        entity_group_name, zones);
					for (const std::string& zone : zones) {
						ErfResultIPtr result = nullptr;
						ErfErrorCode Status =
						    _stage->GetContourResult(state_id, ENTITY_RESULT, entity_type_name,
						                             entity_group_name, zone, result);
						(void) Status;

						EString entity_type;
						ERF_INT node_per_elem;
						result->ReadHeader(entity_type, state_row_count, node_per_elem);

						if (nodes_count == 0) {
							nodes_count = state_row_count;
						}
						// FIXME disable if count or max is invalid

						entityresults_size +=
						    (nodes_count * node_per_elem) * ((sizeof(float_t) + sizeof(uint32_t)));
					}
				}
				entityresults_size += nodes_count * (input_size + 2 * sizeof(int_t) +
				                                     (2 + files_count > 1) * sizeof(uint32_t));
			}
		}

		return entityresults_size;
	};

	size_t constant_entityresults_size = entityresults_size("/post/constant/entityresults", 0);
	size_t singlestate_entityresults_size =
	    entityresults_size("/post/singlestate/entityresults", 1);

	size_t states_count = 0;
	const rapidjson::Value* states =
	    rapidjson::Pointer("/post/singlestate/states").Get(selected_nodes);
	if (states) {
		const std::string& states_list = states->GetString();
		try {
			const auto& ranges = PVCore::deserialize_numbers_as_ranges(states_list);
			states_count = PVCore::get_count_from_ranges(ranges);
		} catch (...) {
		}
	}

	return files_count * (connectivity_size + constant_entityresults_size +
	                      (singlestate_entityresults_size * states_count));
}

std::vector<QDomDocument>
PVRush::PVERFAPI::get_formats_from_selected_nodes(const rapidjson::Document& selected_nodes,
                                                  bool multi_inputs) const
{
	std::vector<QDomDocument> formats;
	std::vector<ERF_INT> state_ids;
	_stage->GetStateIds(state_ids);

	const rapidjson::Value* constant_connectivities =
	    rapidjson::Pointer("/post/constant/connectivities").Get(selected_nodes);
	if (constant_connectivities) {
		add_connectivities(formats, multi_inputs);
	}

	const rapidjson::Value* constant_entityresults =
	    rapidjson::Pointer("/post/constant/entityresults").Get(selected_nodes);
	if (constant_entityresults) {
		add_entityresults(0, formats, constant_entityresults, multi_inputs);
	}

	const rapidjson::Value* singlestate_entityresults =
	    rapidjson::Pointer("/post/singlestate/entityresults").Get(selected_nodes);
	if (singlestate_entityresults) {
		add_entityresults(state_ids[1], formats, singlestate_entityresults, multi_inputs);
	}

#if 0
    for (const QDomDocument& doc : formats) {
        QString xml_str;
        QTextStream stream(&xml_str);
        QDomNode node = doc.firstChildElement("param");
        node.save(stream, 4);
        pvlogger::warn() << qPrintable(xml_str) << std::endl;
    }
#endif

	return formats;
}

std::vector<std::pair<rapidjson::Document, std::string>>
PVRush::PVERFAPI::get_selected_nodes_by_source(const rapidjson::Document& selected_nodes) const
{

#if 0
    rapidjson::StringBuffer buffer;
    buffer.Clear();
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    selected_nodes.Accept(writer);
    pvlogger::warn() << "selected_nodes=" << buffer.GetString() << std::endl;
#endif

	std::vector<std::pair<rapidjson::Document, std::string>> selected_nodes_by_source;

	std::vector<std::tuple<std::string /* pointer */, std::string /* pointer_base */,
	                       std::string /* source_name */>>
	    pointers_source = {
	        {"/post/constant/connectivities", "", "connectivities"},
	        {"/post/constant/entityresults", "", "constant"},
	        {"/post/singlestate/entityresults", "/post/singlestate/states", "results"}};

	for (auto& [pointer_source, pointer_base, source_name] : pointers_source) {
		const rapidjson::Value* source_node =
		    rapidjson::Pointer(pointer_source.c_str()).Get(selected_nodes);
		if (source_node) {
			std::vector<std::string> entity_types;
			if (source_node->IsObject()) {
				for (const auto& entity_type : source_node->GetObject()) {
					rapidjson::Document doc;
					if (not pointer_base.empty()) {
						const rapidjson::Value* base_node =
						    rapidjson::Pointer(pointer_base.c_str()).Get(selected_nodes);
						assert(base_node);
						rapidjson::Pointer(pointer_base.c_str()).Set(doc, *base_node);
					}
					const std::string& subpointer_source =
					    pointer_source + "/" + entity_type.name.GetString();
					rapidjson::Pointer(subpointer_source.c_str()).Set(doc, entity_type.value);
					selected_nodes_by_source.emplace_back(
					    std::move(doc), source_name + "/" + entity_type.name.GetString());
				}
			} else {
				rapidjson::Document doc;
				rapidjson::Pointer(pointer_source.c_str()).Set(doc, *source_node);
				selected_nodes_by_source.emplace_back(std::move(doc), source_name);
			}
		}
	}

#if 0
    for (const auto& [doc, source_name] : selected_nodes_by_source) {
        rapidjson::StringBuffer buffer;
        buffer.Clear();
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        doc.Accept(writer);
        pvlogger::fatal() << source_name << "=" << buffer.GetString() << std::endl;
    }
#endif

	return selected_nodes_by_source;
}

void PVRush::PVERFAPI::add_connectivities(std::vector<QDomDocument>& formats,
                                          bool multi_inputs) const
{
	formats.emplace_back(QDomDocument());
	std::unique_ptr<PVXmlTreeNodeDom> format_root(
	    PVRush::PVXmlTreeNodeDom::new_format(formats.back()));

	if (multi_inputs) {
		format_root->addOneField(
		   	input_column_name,
		    "string");
	}

	format_root->addOneField(QString::fromStdString("idele"), erf_type_traits<int_t>::string);
	format_root->addOneField(QString::fromStdString("type"), "string");
	format_root->addOneField(QString::fromStdString("pid"), erf_type_traits<int_t>::string);
	format_root->addOneField(QString::fromStdString("nodeid"), erf_type_traits<int_t>::string);
}

void PVRush::PVERFAPI::add_entityresults(ERF_INT state_id,
                                         std::vector<QDomDocument>& formats,
                                         const rapidjson::Value* entityresults,
                                         bool multi_inputs) const
{
	for (const auto& entity_type : entityresults->GetObject()) {
		const std::string& entity_type_name = entity_type.name.GetString();

		formats.emplace_back(QDomDocument());
		std::unique_ptr<PVXmlTreeNodeDom> format_root(
		    PVRush::PVXmlTreeNodeDom::new_format(formats.back()));

		if (multi_inputs) {
			format_root->addOneField(
			    input_column_name,
			    "string");
		}

		if (state_id > 0) {
			format_root->addOneField(QString::fromStdString("state"),
			                         erf_type_traits<int_t>::string);
		}

		format_root->addOneField(QString::fromStdString("entid"), erf_type_traits<int_t>::string);

		const auto& entity_groups =
		    entity_type_name == "NODE" ? entity_type.value["groups"] : entity_type.value;

		for (const auto& entity_group : entity_groups.GetArray()) {

			const std::string& entity_group_name = entity_group.GetString();

			std::vector<EString> zones;
			_stage->GetContourZones(state_id, ENTITY_RESULT, entity_type_name, entity_group_name,
			                        zones);
			for (const std::string& zone : zones) {
				ErfResultIPtr result;
				_stage->GetContourResult(state_id, ENTITY_RESULT, entity_type_name,
				                         entity_group_name, zone, result);

				EString entity_type;
				ERF_INT row_count;
				ERF_INT dim_count;
				result->ReadHeader(entity_type, row_count, dim_count);

				for (ERF_INT i = 0; i < dim_count; i++) {
					format_root->addOneField(
					    QString::fromStdString(
					        entity_group_name +
					        (dim_count > 1 ? ("/" + std::to_string(i + 1)) : "")),
					    erf_type_traits<float_t>::string);
				}
			}
		}
	}
}
