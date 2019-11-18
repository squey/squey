/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2019
 */

#ifndef __RUSH_PVERFAPI_H__
#define __RUSH_PVERFAPI_H__

#define DOUBLE_PRECISION 1 // ERF Double precision

#include "inendi_erfio/inendi_erfio.h"

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

class PVERFAPI : public inendi_erf::inendi_erfio
{
  public:
	static constexpr const char* float_type =
	    std::is_same<ERF_FLOAT, double>::value ? "number_double" : "number_float";

	static constexpr const char* int_type =
	    std::is_same<ERF_INT, int>::value ? "number_int32" : "number_int64";

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
	PVERFAPI(const std::string& erf_path) : inendi_erf::inendi_erfio(erf_path) {}

  public:
	int open_stage(std::string stagename)
	{
		int errCode = inendi_erf::inendi_erfio::open_stage(stagename);
		if (not errCode) {
			_parent_stages.emplace_back(m_pstage);
		}
		return errCode;
	}

	int close_stage()
	{
		int errCode = -1;
		if (_parent_stages.size() > 0) {
			m_pstage = _parent_stages.back();
			_parent_stages.pop_back();
			errCode = 0;
		}
		return errCode;
	}

	ErfStageIPtr stage() { return m_pstage; }

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
		// Get stages names
		std::vector<std::string> stage_names;
		get_stage_names(stage_names);
		for (size_t stage_index = 0; stage_index < stage_names.size(); stage_index++) {
			// Get stage names
			const std::string& stage_name = stage_names[stage_index];
			f(stage_name, false, stage_index == stage_names.size() - 1);

			f("constant", false, false);

			// Get connectivities names
			std::vector<std::string> connectivity_types;
			m_pstage->GetElementTypes(0, connectivity_types);
			f("connectivities", false, false);
			for (size_t i = 0; i < connectivity_types.size(); i++) {
				f(connectivity_types[i], true, i == connectivity_types.size() - 1);
			}

			// Get entityresults names
			f("entityresults", false, true);
			std::vector<EString> entity_types;
			std::vector<EString> element_types;
			ErfErrorCode status = m_pstage->GetContourTypes(0, entity_types, element_types);
			for (size_t i = 0; i < entity_types.size(); i++) {
				f(entity_types[i], false, i == entity_types.size() - 1);
				std::vector<EString> entity_groups;
				status =
				    m_pstage->GetContourGroups(0, ENTITY_RESULT, entity_types[i], entity_groups);
				for (size_t j = 0; j < entity_groups.size(); j++) {
					f(entity_groups[j], true, j == entity_groups.size() - 1);
				}
			}

			// Get state names
			std::vector<EString> state_names;
			m_pstage->GetStateNames(state_names);
			if (state_names.size() > 0) {
				f("singlestate", false, true);
				// f("entityresults", false, true);

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

		/*const rapidjson::Value* constant_connectivities =
		    rapidjson::Pointer("/post/constant/connectivities").Get(selected_nodes);
		if (constant_connectivities) {
		    formats.emplace_back(QDomDocument());
		    std::unique_ptr<PVXmlTreeNodeDom> format_root(
		        PVRush::PVXmlTreeNodeDom::new_format(formats.back()));

		    for (const auto& entity_type : constant_connectivities->GetArray()) {
		        std::string entity_type_name = entity_type.GetString();

		        ErfErrorCode status;
		        ErfElementI* elem = m_pstage->GetElement(0, entity_type_name, status);

		        ERF_INT row_count;
		        ERF_INT node_per_elem;
		        ERF_INT dim_count;
		        elem->ReadHeader(row_count, node_per_elem, dim_count);

		        for (size_t i = 0; i < node_per_elem; i++) {
		            PVRush::PVXmlTreeNodeDom* node = format_root->addOneField(
		                QString::fromStdString(
		                    entity_type_name +
		                    (dim_count > 1 ? ("/" + std::to_string(i + 1)) : "")),
		                int_type);
		        }
		    }

		    QString xml_str;
		    QTextStream stream(&xml_str);
		    QDomNode node = formats.back().firstChildElement("param");
		    node.save(stream, 4);
		    pvlogger::warn() << qPrintable(xml_str) << std::endl;

		}*/

		const rapidjson::Value* constant_entityresults =
		    rapidjson::Pointer("/post/constant/entityresults").Get(selected_nodes);
		if (constant_entityresults) {

			formats.emplace_back(QDomDocument());
			std::unique_ptr<PVXmlTreeNodeDom> format_root(
			    PVRush::PVXmlTreeNodeDom::new_format(formats.back()));

			for (const auto& entity_type : constant_entityresults->GetObject()) {
				const std::string& entity_type_name = entity_type.name.GetString();

				/*PVRush::PVXmlTreeNodeDom* node = format_root->addOneField(
				        QString::fromStdString("entid"), int_type);*/

				for (const auto& entity_group : entity_type.value.GetArray()) {

					const std::string& entity_group_name = entity_group.GetString();
					pvlogger::info() << entity_group_name << std::endl;

					std::vector<EString> zones;
					m_pstage->GetContourZones(0, ENTITY_RESULT, entity_type_name, entity_group_name,
					                          zones);
					for (const std::string& zone : zones) {
						ErfResultIPtr result;
						m_pstage->GetContourResult(0, ENTITY_RESULT, entity_type_name,
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
								PVRush::PVXmlTreeNodeDom* node = format_root->addOneField(
								    QString::fromStdString(
								        var_name +
								        (dim_count > 1 ? ("/" + std::to_string(i + 1)) : "")),
								    float_type);
							}
						}
					}
				}
			}

			QString xml_str;
			QTextStream stream(&xml_str);
			QDomNode node = formats.back().firstChildElement("param");
			node.save(stream, 4);
			pvlogger::warn() << qPrintable(xml_str) << std::endl;
		}

		return formats;
	}

  protected:
	std::vector<ErfStageIPtr> _parent_stages;
};

} // namespace PVRush

#endif // __RUSH_PVERFAPI_H__