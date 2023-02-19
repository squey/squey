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

#include "PVERFTreeModel.h"
#include "../../common/erf/PVERFAPI.h"

#include <pvkernel/core/serialize_numbers.h>

#include <QPalette>

/******************************************************************************
 *
 * PVRush::PVERFTreeItem
 *
 *****************************************************************************/
PVRush::PVERFTreeItem::PVERFTreeItem(const QList<QVariant>& data,
                                     bool is_node,
                                     PVERFTreeItem* parent /*= 0*/)
    : _is_node(is_node)
	, _parent(parent)
	, _data(data)
{

	if (_parent) {
		_parent->append_child(this);
	}
}

PVRush::PVERFTreeItem::~PVERFTreeItem()
{
	qDeleteAll(_children);
}

void PVRush::PVERFTreeItem::append_child(PVERFTreeItem* child)
{
#if STORE_INDEX
	_index = _children.size();
#endif
	_children.emplace_back(child);
}

const PVRush::PVERFTreeItem* PVRush::PVERFTreeItem::child(int row) const
{
	return _children[row];
}

int PVRush::PVERFTreeItem::child_count() const
{
	return _children.size();
}

int PVRush::PVERFTreeItem::row() const
{
#if STORE_INDEX
	return _index;
#else
	int index = 0;

	if (_parent) {
		auto it = std::find(_children.begin(), _children.end(), this);
		if (it != _children.end()) {
			index = std::distance(_children.begin(), it);
		}
	}

	return index;
#endif
}

const PVRush::PVERFTreeItem* PVRush::PVERFTreeItem::parent() const
{
	return _parent;
}

int PVRush::PVERFTreeItem::column_count() const
{
	return 1;
}

QVariant PVRush::PVERFTreeItem::data(int column) const
{
	return _data.value(column);
}

bool PVRush::PVERFTreeItem::is_path(const QString& path) const
{
	PVERFTreeItem* item = const_cast<PVERFTreeItem*>(this);
	QStringList parents = path.split(".");
	if (parents.isEmpty()) {
		return false;
	}
	QString name;
	QString parent_name;
	do {
		name = item->data(0).toString();
		parent_name = parents.back();

		parents.pop_back();
		item = item->parent();
	} while (item->parent() != nullptr and not parents.isEmpty() and name == parent_name);

	return parents.isEmpty();
}

QString PVRush::PVERFTreeItem::path() const
{
	PVERFTreeItem* item = const_cast<PVERFTreeItem*>(this);
	QStringList parents;
	do {
		parents.insert(0, item->data(0).toString());
		item = item->parent();
	} while (item->parent() != nullptr);

	return parents.join(".");
}

Qt::CheckState PVRush::PVERFTreeItem::state() const
{
	return _state;
}

void PVRush::PVERFTreeItem::set_state(Qt::CheckState state)
{
	_state = state;
}

PVRush::PVERFTreeItem* PVRush::PVERFTreeItem::selected_child(int row)
{
	PVRush::PVERFTreeItem* last_child = nullptr;

	for (const auto& child : _children) {
		if (child->state() == Qt::CheckState::PartiallyChecked) {
			if (row <= 0)
				return child;
			row--;
		}
		last_child = child;
	}
	return last_child;
}

int PVRush::PVERFTreeItem::selected_child_count() const
{
	int count = 0;
	for (const auto& child : _children) {
		if (child->state() != Qt::CheckState::Unchecked) {
			count++;
		}
	}
	return count;
}

int PVRush::PVERFTreeItem::selected_row() const
{
	int row = 0;

	if (_parent) {
		for (const auto& child : _parent->_children) {
			if (child->state() != Qt::CheckState::Unchecked) {
				if (child == this)
					return row;
				row++;
			}
		}
	}

	return row;
}

/******************************************************************************
 *
 * PVRush::PVERFTreeModel
 *
 *****************************************************************************/
PVRush::PVERFTreeModel::PVERFTreeModel(const QString& path)
	: _path(path)
{
	load(path);
}

bool PVRush::PVERFTreeModel::load(const QString& path)
{
	PVERFAPI erf(path.toStdString());

	_root_item.reset(new PVERFTreeItem(QList<QVariant>({path}), true));

	erf.visit_nodes<PVERFTreeItem>(
	    _root_item.get(), [&](const std::string& name, bool is_leaf, PVERFTreeItem* parent) {
		    PVRush::PVERFTreeItem* item =
		        new PVERFTreeItem(QList<QVariant>({name.c_str()}), not is_leaf, parent);
		    if (item->is_path("post.singlestate.states")) {
			    item->set_type(PVRush::PVERFTreeItem::EType::STATES);
		    } else if (item->is_path("post.constant.entityresults.NODE") or
		               item->is_path("post.singlestate.entityresults.NODE")) {
			    item->set_type(PVRush::PVERFTreeItem::EType::NODE);
		    }
		    return item;
	    });

	return true;
}

rapidjson::Document
PVRush::PVERFTreeModel::save(ENodesType nodes_type /*= ENodesType::SELECTED*/) const
{
	rapidjson::Document doc;
	rapidjson::Document::AllocatorType& alloc = doc.GetAllocator();

	visit_nodes<rapidjson::Value>(
	    nodes_type, &doc.SetObject(), [&](QModelIndex index, rapidjson::Value* parent) {
		    PVERFTreeItem* item = static_cast<PVERFTreeItem*>(index.internalPointer());

		    rapidjson::Value node_name;
		    std::string name = index.data().toString().toStdString();
		    node_name.SetString(name.c_str(), alloc);

		    if (item->is_node()) {
			    if (item->type() != PVERFTreeItem::EType::STATES) {
				    rapidjson::Value val;
				    if (item->children_are_leafs()) {
					    val.SetArray();
					    if (item->type() ==
					        PVERFTreeItem::EType::NODE) { // embed NODE groups in "groups" object
						    val.SetObject();

						    // "list"
						    if (nodes_type == ENodesType::SELECTED) {
							    rapidjson::Value node_list;
							    node_list.SetString(
							        item->user_data().toString().toStdString().c_str(), alloc);
							    val.AddMember("list", node_list, alloc);
						    }

						    // "groups"
						    rapidjson::Value groups_array;
						    groups_array.SetArray();
						    rapidjson::Value groups_name;
						    const std::string& groups_name_str = "groups";
						    groups_name.SetString(groups_name_str.c_str(), alloc);
						    val.AddMember(groups_name, groups_array, alloc);

						    parent->AddMember(node_name, val, alloc);
						    rapidjson::Value* child =
						        &(*parent)[name.c_str()][groups_name_str.c_str()];
						    return child;
					    }
				    } else {
					    val.SetObject();
				    }
				    parent->AddMember(node_name, val, alloc);
				    rapidjson::Value* child = &(*parent)[name.c_str()];
				    return child;
			    } else {
				    std::ostringstream list;
				    std::vector<size_t> values;
				    for (int i = 0; i < rowCount(index); i++) {
					    const QModelIndex& child = this->index(i, 0, index);
					    PVERFTreeItem* child_item =
					        static_cast<PVERFTreeItem*>(child.internalPointer());
					    if (child_item->state() == Qt::Checked or nodes_type == ENodesType::ALL) {
						    values.emplace_back(i);
					    }
				    }
				    PVCore::serialize_numbers(values.begin(), values.end(), list);
				    rapidjson::Value val;
				    val.SetString(list.str().c_str(), alloc);
				    parent->AddMember("states", val, alloc);
			    }
		    } else {
			    parent->PushBack(node_name, alloc);
		    }

		    return (rapidjson::Value*)nullptr;
	    });

	return doc;
}

bool PVRush::PVERFTreeModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
	if (not index.isValid()) {
		return false;
	}

	if (role == Qt::CheckStateRole) {
		PVERFTreeItem* item = static_cast<PVERFTreeItem*>(index.internalPointer());

		item->set_state((Qt::CheckState)value.toInt());

		Q_EMIT indexStateChanged(index);

		Q_EMIT layoutAboutToBeChanged();
		Q_EMIT layoutChanged();

		return true;
	}

	return QAbstractItemModel::setData(index, value, role);
}

QVariant PVRush::PVERFTreeModel::data(const QModelIndex& index, int role) const
{
	if (not index.isValid()) {
		return QVariant();
	}

	PVERFTreeItem* item = static_cast<PVERFTreeItem*>(index.internalPointer());

	if (role == Qt::CheckStateRole and index.column() == 0 and item->is_node()) {
		return item->state();
	}

	if (role != Qt::DisplayRole) {
		return QVariant();
	}

	return item->data(index.column());
}

Qt::ItemFlags PVRush::PVERFTreeModel::flags(const QModelIndex& index) const
{
	if (not index.isValid()) {
		return {};
	}

	Qt::ItemFlags flags = QAbstractItemModel::flags(index) | Qt::ItemIsEnabled;

	PVERFTreeItem* item = static_cast<PVERFTreeItem*>(index.internalPointer());
	if (item->is_node() and index.column() == 0) {
		flags |= Qt::ItemIsUserCheckable | Qt::ItemIsAutoTristate;
		flags &= ~Qt::ItemIsSelectable;
	}

	return flags;
}

QVariant PVRush::PVERFTreeModel::headerData(int section,
                                            Qt::Orientation orientation,
                                            int role /*= Qt::DisplayRole*/) const
{
	if (orientation == Qt::Horizontal && role == Qt::DisplayRole) {
		return _root_item->data(section);
	}

	return QVariant();
}

QModelIndex PVRush::PVERFTreeModel::index(int row,
                                          int column,
                                          const QModelIndex& parent /*= QModelIndex()*/) const
{
	if (not hasIndex(row, column, parent)) {
		return QModelIndex();
	}

	PVERFTreeItem* parent_item;

	if (not parent.isValid()) {
		parent_item = _root_item.get();
	} else {
		parent_item = static_cast<PVERFTreeItem*>(parent.internalPointer());
	}

	PVERFTreeItem* child_item = parent_item->child(row);
	if (child_item) {
		return createIndex(row, column, child_item);
	} else {
		return QModelIndex();
	}
}

QModelIndex PVRush::PVERFTreeModel::parent(const QModelIndex& index) const
{
	if (not index.isValid()) {
		return QModelIndex();
	}

	PVERFTreeItem* child_item = static_cast<PVERFTreeItem*>(index.internalPointer());
	PVERFTreeItem* parent_item = child_item->parent();

	if (parent_item == _root_item.get()) {
		return QModelIndex();
	}

	return createIndex(parent_item->row(), 0, parent_item);
}

int PVRush::PVERFTreeModel::rowCount(const QModelIndex& parent /*= QModelIndex()*/) const
{
	PVERFTreeItem* parent_item;
	if (parent.column() > 0) {
		return 0;
	}

	if (not parent.isValid()) {
		parent_item = _root_item.get();
	} else {
		parent_item = static_cast<PVERFTreeItem*>(parent.internalPointer());
	}

	return parent_item->child_count();
}

int PVRush::PVERFTreeModel::columnCount(const QModelIndex& parent /*= QModelIndex()*/) const
{
	if (parent.isValid()) {
		return static_cast<PVERFTreeItem*>(parent.internalPointer())->column_count();
	} else {
		return _root_item->column_count();
	}
}
