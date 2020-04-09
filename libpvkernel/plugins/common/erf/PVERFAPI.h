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
#include <pvkernel/core/serialize_numbers.h>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
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
template <>
struct erf_type_traits<uint16_t> {
	static constexpr const char* string = "number_uint16";
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
	PVERFAPI(const std::string& erf_path);

	bool set_path(const std::string& erf_path);

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

		visit_nodes_impl<NodeType>(create_node);
	}

  private:
	template <typename NodeType>
	void visit_nodes_impl(const create_node_f<NodeType>& f)
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
		std::vector<ERF_INT> state_ids;
		_stage->GetStateIds(state_ids);
		if (state_ids.size() > 0) {
			status = _stage->GetContourTypes(state_ids[1], entity_types, element_types);
			std::vector<EString> state_names;
			_stage->GetStateNames(state_names);

			f("singlestate", false, false);

			f("entityresults", false, false);

			for (size_t i = 0; i < entity_types.size(); i++) {
				std::vector<std::string> entity_groups;
				_stage->GetContourGroups(state_ids[1], ENTITY_RESULT, entity_types[i],
				                         entity_groups);

				f(entity_types[i], false, i == entity_types.size() - 1);

				// Get entity groups names
				for (size_t j = 0; j < entity_groups.size(); j++) {
					f(entity_groups[j], true, j == entity_groups.size() - 1);
				}
			}

			// Get state names
			f("states", false, true);
			for (size_t state_index = 0; state_index < state_names.size(); state_index++) {
				const std::string& state_name = state_names[state_index];
				f(state_name, true, state_index == state_names.size() - 1);
			}
		}

		(void)status;
	}

  public:
	std::vector<std::tuple<rapidjson::Document, std::string, PVRush::PVFormat>>
	get_sources_info(const rapidjson::Document& selected_nodes, bool multi_inputs) const;

	size_t memory_size(const rapidjson::Document& selected_nodes, size_t files_count) const;

  private:
	std::vector<QDomDocument>
	get_formats_from_selected_nodes(const rapidjson::Document& selected_nodes,
	                                bool multi_inputs) const;

	std::vector<std::pair<rapidjson::Document, std::string>>
	get_selected_nodes_by_source(const rapidjson::Document& selected_nodes) const;

  private:
	void add_connectivities(std::vector<QDomDocument>& formats,
	                        const rapidjson::Value* connectivities,
	                        bool multi_inputs) const;

	void add_entityresults(ERF_INT state_id,
	                       std::vector<QDomDocument>& formats,
	                       const rapidjson::Value* entityresults,
	                       bool multi_inputs) const;

  private:
	std::string _stage_name;
	bool _lib_Initialized = false;
	std::unique_ptr<ErfFilerI> _filer;
	ErfStageIPtr _stage;
};

} // namespace PVRush

#endif // __RUSH_PVERFAPI_H__