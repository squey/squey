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

#include <libpvpcap.h>
#include <libpvpcap/extraction.h>
#include <libpvpcap/shell.h>
#include "include/PcapTreeModel.h"
#include "iostream"

#include <pvkernel/rush/PVNrawCacheManager.h>

#include <libpvpcap/ws.h>

#include <pvlogger.h>

#include <charconv>
#include <unordered_map>

namespace pvpcap
{

/**************************************************************************
 *
 * JsonTreeItem::JsonTreeItem
 *
 *************************************************************************/
JsonTreeItem::JsonTreeItem(rapidjson::Value* value, JsonTreeItem* parent)
{
	_parent = parent;
	_value = value;
}

JsonTreeItem::JsonTreeItem(JsonTreeItem* parent)
{
	_parent = parent;
	_value = nullptr;
}

JsonTreeItem::~JsonTreeItem()
{
	qDeleteAll(_children);
}

void JsonTreeItem::append_child(JsonTreeItem* child)
{
	_children.append(child);
}

JsonTreeItem* JsonTreeItem::child(int row)
{
	return _children.value(row);
}

int JsonTreeItem::child_count() const
{
	return _children.count();
}

int JsonTreeItem::row() const
{
	if (_parent)
		return _parent->_children.indexOf(const_cast<JsonTreeItem*>(this));

	return 0;
}

JsonTreeItem* JsonTreeItem::parent()
{
	return _parent;
}

rapidjson::Value& JsonTreeItem::value() const
{
	return *_value;
}

JsonTreeItem::CHILDREN_SELECTION_STATE JsonTreeItem::selection_state() const
{
	CHILDREN_SELECTION_STATE sel_state = CHILDREN_SELECTION_STATE::UNKOWN;

	for (const auto& child : (*_value)["fields"].GetArray()) {
		const std::string& filter_name = child["filter_name"].GetString();
		if (pvpcap::ws_disabled_fields.find(filter_name) == pvpcap::ws_disabled_fields.end()) {
			if (child["select"].GetBool()) { // child selected
				if (sel_state == CHILDREN_SELECTION_STATE::UNSELECTED) {
					return CHILDREN_SELECTION_STATE::PARTIALY_SELECTED;
				} else {
					sel_state = CHILDREN_SELECTION_STATE::TOTALY_SELECTED;
				}
			} else { // child unselected
				if (sel_state == CHILDREN_SELECTION_STATE::TOTALY_SELECTED) {
					return CHILDREN_SELECTION_STATE::PARTIALY_SELECTED;
				} else {
					sel_state = CHILDREN_SELECTION_STATE::UNSELECTED;
				}
			}
		}
	}
	return sel_state;
}

static void change_children_selection(rapidjson::Value* value, bool select_all)
{
	for (auto& child : (*value)["fields"].GetArray()) {
		const std::string& filter_name = child["filter_name"].GetString();
		if (pvpcap::ws_disabled_fields.find(filter_name) == pvpcap::ws_disabled_fields.end()) {
			child["select"].SetBool(select_all);
		}
	}
}

void JsonTreeItem::select_all_children()
{
	change_children_selection(_value, true);
}

void JsonTreeItem::unselect_children()
{
	change_children_selection(_value, false);
}

JsonTreeItem* JsonTreeItem::selected_child(int row)
{
	JsonTreeItem* last_child = nullptr;

	for (const auto& child : _children) {
		if (child->selection_state() == CHILDREN_SELECTION_STATE::PARTIALY_SELECTED) {
			if (row <= 0)
				return child;
			row--;
		}
		last_child = child;
	}
	return last_child;
}

int JsonTreeItem::selected_child_count() const
{
	int count = 0;
	for (const auto& child : _children) {
		if (child->selection_state() == CHILDREN_SELECTION_STATE::PARTIALY_SELECTED) {
			count++;
		}
	}
	return count;
}

int JsonTreeItem::selected_row() const
{
	int row = 0;

	if (_parent) {
		for (const auto& child : _parent->_children) {
			if (child->selection_state() == CHILDREN_SELECTION_STATE::PARTIALY_SELECTED) {
				if (child == this)
					return row;
				row++;
			}
		}
	}

	return row;
}

JsonTreeItem* JsonTreeItem::load(rapidjson::Value* value, JsonTreeItem* parent)
{
	assert(value->IsObject() && "In JsonTreeItem::load, value node is not a rapidjson object!");

	auto* root_item = new JsonTreeItem(value, parent);

	// children
	auto& children = (*value)["children"];
	assert(children.IsArray() && "In JsonTreeItem::load, children list is not a rapidjson array!");

	for (auto& child : children.GetArray()) {
		assert(child.IsObject() && "In JsonTreeItem::load, child node is not a rapidjson object!");

		JsonTreeItem* child_item = load(&child, root_item);
		root_item->append_child(child_item);
	}

	return root_item;
}

/**************************************************************************
 *
 * PcapTreeModel::index
 *
 *************************************************************************/

QModelIndex PcapTreeModel::index(int row, int column, const QModelIndex& parent) const
{
	if (not hasIndex(row, column, parent))
		return {};

	JsonTreeItem* parent_item;

	if (!parent.isValid())
		parent_item = _root_item;
	else
		parent_item = static_cast<JsonTreeItem*>(parent.internalPointer());

	JsonTreeItem* child_item;
	if (_with_selection)
		child_item = parent_item->selected_child(row);
	else
		child_item = parent_item->child(row);

	if (child_item)
		return createIndex(row, column, child_item);
	else
		return {};
}

/**************************************************************************
 *
 * PcapTreeModel::parent
 *
 *************************************************************************/
QModelIndex PcapTreeModel::parent(const QModelIndex& index) const
{
	if (not index.isValid()) {
		return {};
	}

	auto* child_item = static_cast<JsonTreeItem*>(index.internalPointer());
	JsonTreeItem* parent_item = child_item->parent();

	if (parent_item == _root_item)
		return {};

	int position = (_with_selection) ? parent_item->selected_row() : parent_item->row();
	return createIndex(position, 0, parent_item);
}

/**************************************************************************
 *
 * PcapTreeModel::columnCount
 *
 *************************************************************************/
int PcapTreeModel::columnCount(const QModelIndex& parent) const
{
	Q_UNUSED(parent);

	return 7; /* Name, Short name, Filter, Packets, % Packets, Bytes, % Bytes */
}

/**************************************************************************
 *
 * PcapTreeModel::rowCount
 *
 *************************************************************************/
int PcapTreeModel::rowCount(const QModelIndex& parent) const
{
	JsonTreeItem* parent_item;
	if (parent.column() > 0)
		return 0;

	if (!parent.isValid())
		parent_item = _root_item;
	else
		parent_item = static_cast<JsonTreeItem*>(parent.internalPointer());

	if (_with_selection) {
		return parent_item->selected_child_count();
	} else {
		return parent_item->child_count();
	}
}

/**************************************************************************
 *
 * PcapTreeModel::data
 *
 *************************************************************************/
bool PcapTreeModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
	if (not index.isValid()) {
		return false;
	}

	if (role == Qt::CheckStateRole) {

		auto* item = static_cast<JsonTreeItem*>(index.internalPointer());

		switch ((Qt::CheckState)value.toInt()) {
		case Qt::PartiallyChecked:
		case Qt::Unchecked:
			item->unselect_children();
			break;
		case Qt::Checked:
			item->select_all_children();
			break;
		default:
			assert(false);
		}

		return true;
	}

	return true;
}

/**************************************************************************
 *
 * PcapTreeModel::data
 *
 *************************************************************************/
QVariant PcapTreeModel::data(const QModelIndex& index, int role) const
{
	if (not index.isValid())
		return {};

	// Display protocol data.
	switch (role) {
	case Qt::DisplayRole: {
		auto* item = static_cast<JsonTreeItem*>(index.internalPointer());
		rapidjson::Value& value = item->value();
		qreal ratio;

		switch (index.column()) {
		case 0: /* Name */
			return value["name"].GetString();
		case 1: /* Short name */
			return value["short_name"].GetString();
		case 2: /* Filter name */
			return value["filter_name"].GetString();
		case 3: /* Packets */
			return (qulonglong)value["packets"].GetUint64();
		case 4: /* % Packets */
			ratio =
			    (qreal)value["packets"].GetUint64() / _root_item->value()["packets"].GetUint64();
			return QString::number(ratio * 100, 'f', 2);
		case 5: /* Bytes */
			return (qulonglong)value["bytes"].GetUint64();
		case 6: /* % Bytes */
			ratio = (qreal)value["bytes"].GetUint64() / _root_item->value()["bytes"].GetUint64();
			return QString::number(ratio * 100, 'f', 2);
		default:
			assert(false && "We only have 7 columns");
			return {};
		}
		break;
	}

	case Qt::TextAlignmentRole:
		switch (index.column()) {
		case 0: /* Name */
			return Qt::AlignLeft;
		case 1: /* Short name */
			return Qt::AlignLeft;
		case 2: /* Filter name */
			return Qt::AlignLeft;
		case 3: /* Packets */
			return Qt::AlignRight;
		case 4: /* % Packets */
			return Qt::AlignRight;
		case 5: /* Bytes */
			return Qt::AlignRight;
		case 6: /* % Bytes */
			return Qt::AlignRight;
		default:
			return {};
		}
		break;

	case Qt::CheckStateRole:
		if (index.column() == 0) {
			// Show checkbox only for the first column.
			auto* item = static_cast<JsonTreeItem*>(index.internalPointer());

			switch (item->selection_state()) {
			case JsonTreeItem::CHILDREN_SELECTION_STATE::UNSELECTED:
				return Qt::Unchecked;
			case JsonTreeItem::CHILDREN_SELECTION_STATE::PARTIALY_SELECTED:
				return Qt::PartiallyChecked;
			case JsonTreeItem::CHILDREN_SELECTION_STATE::TOTALY_SELECTED:
				return Qt::Checked;
			default:
				return Qt::Unchecked;
			}
		}
	}

	return {};
}

/**************************************************************************
 *
 * PcapTreeModel::headerData
 *
 *************************************************************************/
QVariant PcapTreeModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	if (orientation == Qt::Horizontal && role == Qt::DisplayRole) {
		switch (section) {
		case 0: /* Name */
			return "Name";
		case 1: /* Short name */
			return "Short name";
		case 2: /* Filter name */
			return "Filter";
		case 3: /* Packets */
			return "Packets";
		case 4: /* % Packets */
			return "% Packets";
		case 5: /* Bytes */
			return "Bytes";
		case 6: /* % Bytes */
			return "% Bytes";
		default:
			assert(false && "We only have 7 columns");
			return {};
		}
	}
	return {};
}

/**************************************************************************
 *
 * PcapTreeModel::flags
 *
 *************************************************************************/
Qt::ItemFlags PcapTreeModel::flags(const QModelIndex& index) const
{
	if (not index.isValid()) {
		return Qt::NoItemFlags;
	}

	if (index.column() == 0) {
		return QAbstractItemModel::flags(index) | Qt::ItemIsUserCheckable;
	}

	return QAbstractItemModel::flags(index);
}

/**************************************************************************
 *
 * PcapTreeModel::read_protocols
 *
 *************************************************************************/
namespace
{

//! The protocols and the fields tshark knows, kept in a file as long as tshark is the same.
const rapidjson::Document& protocols_dictionary()
{
	static const rapidjson::Document dictionary = []() {
		const std::string& file = protocols_dict_path();
		if (not file_exists(file)) {
			create_protocols_dict(file);
		}
		rapidjson::Document doc = parse_protocol_dict(file);
		if (not doc.HasMember("tshark_version") or not doc["tshark_version"].IsString() or
		    doc["tshark_version"].GetString() != ws_get_version()) {
			create_protocols_dict(file);
			doc = parse_protocol_dict(file);
		}
		return doc;
	}();
	return dictionary;
}

struct protocol_node {
	std::string name;
	uint64_t packets = 0;
	uint64_t bytes = 0;
	std::vector<size_t> children;
	std::unordered_map<std::string, size_t> by_name;
};

void to_json(const std::vector<protocol_node>& nodes,
             size_t index,
             rapidjson::Value& children,
             rapidjson::Document::AllocatorType& alloc)
{
	for (size_t child : nodes[index].children) {
		const protocol_node& node = nodes[child];
		rapidjson::Value protocol(rapidjson::kObjectType);
		protocol.AddMember("filter_name", rapidjson::Value(node.name.c_str(), alloc), alloc);
		protocol.AddMember("bytes", rapidjson::Value(node.bytes), alloc);
		protocol.AddMember("packets", rapidjson::Value(node.packets), alloc);
		rapidjson::Value grandchildren(rapidjson::kArrayType);
		to_json(nodes, child, grandchildren, alloc);
		protocol.AddMember("children", grandchildren, alloc);
		children.PushBack(protocol, alloc);
	}
}

} // namespace

rapidjson::Document PcapTreeModel::read_protocols(const QString& capture,
                                                  const std::atomic<bool>& canceled)
{
	const rapidjson::Document& dictionary = protocols_dictionary();

	extraction::options opts;
	opts.fields = {{"frame.len", extraction::field::kind::text},
	               {"frame.protocols", extraction::field::kind::text}};
	opts.environment = tshark_environment();
	opts.work_directory = PVRush::PVNrawCacheManager::nraw_dir().toStdString();

	// The hierarchy of the protocols, as "tshark -z io,phs" tells it, under "frame".
	std::vector<protocol_node> nodes(2);
	nodes[1].name = "frame";
	nodes[0].children.push_back(1);

	extraction reading(capture.toStdString(), opts);
	extraction::line line;
	while (reading.next(line)) {
		if (canceled) {
			reading.cancel();
			return {};
		}
		const size_t separator = line.fields.find(SEPARATOR[0]);
		if (separator == std::string_view::npos) {
			continue;
		}
		uint64_t length = 0;
		std::from_chars(line.fields.data(), line.fields.data() + separator, length);
		std::string_view protocols = line.fields.substr(separator + 1);

		size_t current = 1;
		for (size_t n : {size_t(0), size_t(1)}) {
			nodes[n].packets++;
			nodes[n].bytes += length;
		}
		while (not protocols.empty()) {
			const size_t end = std::min(protocols.find(':'), protocols.size());
			const std::string name(protocols.substr(0, end));
			protocols.remove_prefix(std::min(end + 1, protocols.size()));
			// What tshark names but does not list as a protocol has no field to offer, and
			// the ether type is a layer the hierarchy of tshark does not show.
			if (name == "ethertype" or not dictionary.HasMember(name.c_str())) {
				continue;
			}
			auto [it, added] = nodes[current].by_name.try_emplace(name, nodes.size());
			if (added) {
				nodes[current].children.push_back(nodes.size());
				nodes.push_back({});
				nodes.back().name = name;
			}
			current = it->second;
			nodes[current].packets++;
			nodes[current].bytes += length;
		}
	}

	rapidjson::Document tree;
	tree.SetObject();
	rapidjson::Document::AllocatorType& alloc = tree.GetAllocator();
	rapidjson::Value children(rapidjson::kArrayType);
	to_json(nodes, 0, children, alloc);
	tree.AddMember("children", children, alloc);
	tree.AddMember("bytes", rapidjson::Value(nodes[0].bytes), alloc);
	tree.AddMember("packets", rapidjson::Value(nodes[0].packets), alloc);
	return tree;
}

void PcapTreeModel::set_protocols(const rapidjson::Document& protocols)
{
	enrich_protocols_tree(*_json_data, protocols, protocols_dictionary());
	reset();
}

void PcapTreeModel::reset()
{
	beginResetModel();
	_root_item = JsonTreeItem::load(_json_data);

	endResetModel();
}

} // namespace pvpcap
