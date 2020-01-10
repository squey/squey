/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2019
 */

#ifndef __RUSH_PVERFAPI_H__
#define __RUSH_PVERFAPI_H__

//#define DOUBLE_PRECISION 1 // ERF Double precision

#include <ErfPublic.h>
#include <ErfSpecType.h>

#include <QDomDocument>
#include <QTextStream>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVXmlTreeNodeDom.h>

#include <rapidjson/pointer.h>

namespace PVRush
{

/******************************************************************************
 *
 * PVRush::PVERFAPI
 *
 *****************************************************************************/

template <typename T>
struct erf_type_traits {
};
template <>
struct erf_type_traits<float> {
	static constexpr const char* string = "number_float";
};
template <>
struct erf_type_traits<double> {
	static constexpr const char* string = "number_double";
};
template <>
struct erf_type_traits<int32_t> {
	static constexpr const char* string = "number_int32";
};
template <>
struct erf_type_traits<int64_t> {
	static constexpr const char* string = "number_int64";
};

class PVERFAPI
{
  public:
  public:
	using float_t = float;
	using int_t = ERF_INT;

  public:
	template <typename NodeType>
	using visit_node_f =
	    std::function<NodeType*(const std::string& name, bool is_leaf, NodeType* parent)>;

  private:
	template <typename NodeType>
	using parents_nodes_stack_t = std::vector<std::pair<NodeType*, bool>>;
	template <typename NodeType>
	using create_node_f =
	    std::function<NodeType*(const std::string& name, bool is_leaf, bool is_last_child)>;

  public:
	PVERFAPI(const std::string& erf_path)
	{
		_filer = 0;
		_stage = 0;
		_lib_Initialized = false;
		_stage_name = "post";

		// Initialize the ERF library once and for all
		static ErfErrorCode Status = []() { return ErfLibManager::LibInitialize(); }();

		if (Status == ERF_SUCCESS) {
			_lib_Initialized = true;
			// Create a filer interface
			ErfOpenMode Mode = ERF_READ;
			_filer = ErfFactory::CreateFiler(erf_path, ERF_READ, Status);
			if (Status == 0 && _filer) {
				_stage = _filer->GetStage(_stage_name);
			}
		}
	}

	~PVERFAPI() { delete _filer; }

  public:
	ErfStageIPtr stage() { return _stage; }

	template <typename NodeType>
	void visit_nodes(NodeType* root, const visit_node_f<NodeType>& f)
	{
		parents_nodes_stack_t<NodeType> parents({std::make_pair(root, true)});

		auto create_node = [&](const std::string& name, bool is_leaf, bool is_last_child) {
			NodeType* item = f(name, is_leaf, parents.back().first);

			if (not is_leaf) { // node
				parents.emplace_back(item, is_last_child);
			}

			if (is_last_child) {
				if (is_leaf) {
					while (parents.back().second) {
						parents.pop_back();
					}
					parents.pop_back();
				}
			}

			return item;
		};

		visit_nodes_impl<NodeType>(create_node, parents);
	}

  private:
	template <typename NodeType>
	void visit_nodes_impl(const create_node_f<NodeType>& f,
	                      parents_nodes_stack_t<NodeType>& parents)
	{

		f(_stage_name, false, true);

		f("constant", false, false);

		// Get connectivities names
		std::vector<std::string> connectivity_types;
		_stage->GetElementTypes(0, connectivity_types);
		f("connectivities", false, false);
		for (size_t i = 0; i < connectivity_types.size(); i++) {
			f(connectivity_types[i], true, i == connectivity_types.size() - 1);
		}

		// Get entityresults names
		f("entityresults", false, true);
		std::vector<EString> entity_types;
		std::vector<EString> element_types;
		ErfErrorCode status = _stage->GetContourTypes(0, entity_types, element_types);
		for (size_t i = 0; i < entity_types.size(); i++) {
			f(entity_types[i], false, i == entity_types.size() - 1);
			std::vector<EString> entity_groups;
			status = _stage->GetContourGroups(0, ENTITY_RESULT, entity_types[i], entity_groups);
			for (size_t j = 0; j < entity_groups.size(); j++) {
				f(entity_groups[j], true, j == entity_groups.size() - 1);
			}
		}

		// singlestate
		status = _stage->GetContourTypes(1, entity_types, element_types);
		const std::string& node_type = "NODE";
		if (std::find(entity_types.begin(), entity_types.end(), node_type) != entity_types.end()) {
			pvlogger::info() << "OK" << std::endl;

			std::vector<EString> state_names;
			_stage->GetStateNames(state_names);
			if (state_names.size() > 0) {
				f("singlestate", false, false);

				f("entityresults", false, false);
				std::vector<std::string> entity_groups;
				_stage->GetContourGroups(1, ENTITY_RESULT, node_type, entity_groups);

				f(node_type, false, true);

				// Get entity groups names
				for (size_t j = 0; j < entity_groups.size(); j++) {
					f(entity_groups[j], true, j == entity_groups.size() - 1);
				}

				// Get state names
				f("states", false, true);
				for (size_t state_index = 0; state_index < state_names.size(); state_index++) {
					const std::string& state_name = state_names[state_index];
					f(state_name, true, state_index == state_names.size() - 1);
				}
			}
		}
	}

  public:
	std::vector<QDomDocument>
	get_formats_from_selected_nodes(const rapidjson::Document& selected_nodes) const
	{
		std::vector<QDomDocument> formats;

		const rapidjson::Value* constant_connectivities =
		    rapidjson::Pointer("/post/constant/connectivities").Get(selected_nodes);
		if (constant_connectivities) {
			add_connectivities(formats, constant_connectivities);
		}

		const rapidjson::Value* constant_entityresults =
		    rapidjson::Pointer("/post/constant/entityresults").Get(selected_nodes);
		if (constant_entityresults) {
			add_entityresults(0, formats, constant_entityresults);
		}

		const rapidjson::Value* singlestate_entityresults =
		    rapidjson::Pointer("/post/singlestate/entityresults").Get(selected_nodes);
		if (singlestate_entityresults) {
			add_entityresults(1, formats, singlestate_entityresults);
		}

		for (const QDomDocument& doc : formats) {
			QString xml_str;
			QTextStream stream(&xml_str);
			QDomNode node = doc.firstChildElement("param");
			node.save(stream, 4);
			pvlogger::warn() << qPrintable(xml_str) << std::endl;
		}

		return formats;
	}

  private:
	void add_connectivities(std::vector<QDomDocument>& formats,
	                        const rapidjson::Value* connectivities) const
	{
		formats.emplace_back(QDomDocument());
		std::unique_ptr<PVXmlTreeNodeDom> format_root(
		    PVRush::PVXmlTreeNodeDom::new_format(formats.back()));

		for (const auto& entity_type : connectivities->GetArray()) {
			std::string entity_type_name = entity_type.GetString();

			ErfErrorCode status;
			ErfElementI* elem = _stage->GetElement(0, entity_type_name, status);

			format_root->addOneField(QString::fromStdString("idele"),
			                         erf_type_traits<int_t>::string);

			format_root->addOneField(QString::fromStdString("pid"), erf_type_traits<int_t>::string);

			format_root->addOneField(QString::fromStdString(entity_type_name),
			                         erf_type_traits<int_t>::string);
		}
	}

	void add_entityresults(ERF_INT StateId,
	                       std::vector<QDomDocument>& formats,
	                       const rapidjson::Value* entityresults) const
	{
		formats.emplace_back(QDomDocument());
		std::unique_ptr<PVXmlTreeNodeDom> format_root(
		    PVRush::PVXmlTreeNodeDom::new_format(formats.back()));

		for (const auto& entity_type : entityresults->GetObject()) {
			const std::string& entity_type_name = entity_type.name.GetString();

			/*format_root->addOneField(
			        QString::fromStdString("entid"), int_type);*/

			for (const auto& entity_group : entity_type.value.GetArray()) {

				const std::string& entity_group_name = entity_group.GetString();
				pvlogger::info() << entity_group_name << std::endl;

				std::vector<EString> zones;
				_stage->GetContourZones(StateId, ENTITY_RESULT, entity_type_name, entity_group_name,
				                        zones);
				for (const std::string& zone : zones) {
					ErfResultIPtr result;
					_stage->GetContourResult(StateId, ENTITY_RESULT, entity_type_name,
					                         entity_group_name, zone, result);

					std::vector<EString> var_names;
					std::vector<EString> ovVariablesClass;
					std::vector<EString> VariablesKeys;
					result->GetVariables(var_names, ovVariablesClass, VariablesKeys);

					for (const std::string& var_name : var_names) {
						EString entity_type;
						ERF_INT row_count;
						ERF_INT dim_count;
						result->ReadHeader(entity_type, row_count, dim_count);

						for (size_t i = 0; i < dim_count; i++) {
							format_root->addOneField(
							    QString::fromStdString(
							        var_name +
							        (dim_count > 1 ? ("/" + std::to_string(i + 1)) : "")),
							    erf_type_traits<float_t>::string);
						}
					}
				}
			}
		}
	}

  private:
	std::string _stage_name;
	bool _lib_Initialized;
	ErfFilerIPtr _filer;
	ErfStageIPtr _stage;
};

} // namespace PVRush

#endif // __RUSH_PVERFAPI_H__