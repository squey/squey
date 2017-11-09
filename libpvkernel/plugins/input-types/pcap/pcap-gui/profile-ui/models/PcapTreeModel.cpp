/*!
 * \file
 * \brief Manage the pcap protocols tree model.
 *
 * Manage the model used to bind the PcapTree structure to a Qt Model.
 *
 * \copyright (C) Philippe Saade - Justin Teliet 2014
 * \copyright (C) ESI Group INENDI 2017
 */

#include <libpvpcap.h>
#include <libpvpcap/pcap_splitter.h>
#include <libpvpcap/shell.h>
#include "include/PcapTreeModel.h"
#include "iostream"

#include <pvkernel/rush/PVNrawCacheManager.h>

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
	_value = 0;
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

bool JsonTreeItem::is_selected() const
{
	for (auto& v : (*_value)["fields"].GetArray()) {
		if (v["select"].GetBool())
			return true;
	}
	return false;
}

bool JsonTreeItem::is_any_child_selected() const
{
	if (is_selected())
		return true;

	for (auto child : _children) {
		return child->is_any_child_selected();
	}

	return false;
}

JsonTreeItem* JsonTreeItem::selected_child(int row)
{
	JsonTreeItem* last_child = nullptr;

	for (auto child : _children) {
		if (child->is_any_child_selected()) {
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
	for (auto child : _children) {
		if (child->is_any_child_selected()) {
			count++;
		}
	}
	return count;
}

int JsonTreeItem::selected_row() const
{
	int row = 0;

	if (_parent) {
		for (auto child : _parent->_children) {
			if (child->is_any_child_selected()) {
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

	JsonTreeItem* root_item = new JsonTreeItem(value, parent);

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
		return QModelIndex();

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
		return QModelIndex();
}

/**************************************************************************
 *
 * PcapTreeModel::parent
 *
 *************************************************************************/
QModelIndex PcapTreeModel::parent(const QModelIndex& index) const
{
	if (not index.isValid()) {
		return QModelIndex();
	}

	JsonTreeItem* child_item = static_cast<JsonTreeItem*>(index.internalPointer());
	JsonTreeItem* parent_item = child_item->parent();

	if (parent_item == _root_item)
		return QModelIndex();

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
QVariant PcapTreeModel::data(const QModelIndex& index, int role) const
{
	if (not index.isValid())
		return QVariant();

	// Display protocol data.
	switch (role) {
	case Qt::DisplayRole: {
		JsonTreeItem* item = static_cast<JsonTreeItem*>(index.internalPointer());
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
			return QVariant();
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
			return QVariant();
		}
		break;

	case Qt::CheckStateRole:
		if (index.column() == 0) {
			// Show checkbox only for the first column.
			JsonTreeItem* item = static_cast<JsonTreeItem*>(index.internalPointer());

			if (item->is_selected())
				return Qt::PartiallyChecked;

			return Qt::Unchecked;
		}
	}

	return QVariant();
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
			return QVariant();
		}
	}
	return QVariant();
}

/**************************************************************************
 *
 * PcapTreeModel::flags
 *
 *************************************************************************/
Qt::ItemFlags PcapTreeModel::flags(const QModelIndex& index) const
{
	if (!index.isValid()) {
		return Qt::NoItemFlags;
	}

	return QAbstractItemModel::flags(index);
}

/**************************************************************************
 *
 * PcapTreeModel::load
 *
 *************************************************************************/
bool PcapTreeModel::load(QString filename, bool& canceled)
{
	// we load a dictionnary of all the protocols knwown in tshark.
	// we have stored this dictionnary in a json file.
	// If it doesn't exist we create a new one and load it.
	const std::string& protocols_dict_file = protocols_dict_path();

	static rapidjson::Document protocols_dict = [&]() {
		if (not file_exists(protocols_dict_file)) {
			create_protocols_dict(protocols_dict_file);
		}
		return parse_protocol_dict(protocols_dict_file);
	}();

	std::vector<std::string> cmd;
	cmd.emplace_back("tshark");
	cmd.emplace_back("-q");
	cmd.emplace_back("-zio,phs,frame");
	cmd.emplace_back("-r-");

	// TODO : rename "extract_csv" into "parallel_exec"
	splitted_files_t files =
	    pvpcap::extract_csv(pvpcap::split_pcap(filename.toStdString(),
	                                           PVRush::PVNrawCacheManager::nraw_dir().toStdString(),
	                                           false /* no need to preserve flows */, canceled),
	                        cmd, canceled);

	// merge documents and delete intermediary files
	std::vector<rapidjson::Document> trees(files.size());
#pragma omp parallel for
	for (size_t i = 0; i < files.size(); i++) {
		trees[i] = create_protocols_tree(files[i].path(), protocols_dict);
		std::remove(files[i].path().c_str());
	}
	for (size_t i = 1; i < files.size(); i++) {
		merge_protocols_tree(trees[0], trees[i], trees[0].GetAllocator());
	}

	// remove parent temporary directory
	std::remove(
	    QFileInfo(QString::fromStdString(files[0].path())).absolutePath().toStdString().c_str());

	if (not canceled) {
		enrich_protocols_tree(*_json_data, trees[0], protocols_dict);
		reset();
		return true;
	} else {
		return false;
	}
}

void PcapTreeModel::reset()
{
	beginResetModel();
	_root_item = JsonTreeItem::load(_json_data);

	endResetModel();
}

} // namespace pvpcap
