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

#include "PVOpcUaTreeModel.h"

#include "PVOpcUaTreeItem.h"

#include <QOpcUaClient>
#include <QPixmap>

namespace PVRush
{

PVOpcUaTreeModel::PVOpcUaTreeModel(QObject* parent) : QAbstractItemModel(parent) {}

void PVOpcUaTreeModel::setOpcUaClient(QOpcUaClient* client)
{
	beginResetModel();
	m_client = client;
	if (m_client)
		m_root_item.reset(
		    new PVOpcUaTreeItem(client->node("ns=0;i=84"), this /* model */, nullptr /* parent */));
	else
		m_root_item.reset(nullptr);
	endResetModel();
}

QOpcUaClient* PVOpcUaTreeModel::opcUaClient() const
{
	return m_client;
}

QVariant PVOpcUaTreeModel::data(const QModelIndex& index, int role) const
{
	if (!index.isValid())
		return {};

	auto item = static_cast<PVOpcUaTreeItem*>(index.internalPointer());

	if (role == Qt::DisplayRole) {
		return item->data(index.column());
	} else if (role == Qt::DecorationRole && index.column() == 0) {
		return item->icon(index.column());
	} else if (role == Qt::UserRole) {
		return item->user_data(index.column());
	}

	return {};
}

QVariant PVOpcUaTreeModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	if (role != Qt::DisplayRole)
		return {};

	if (orientation == Qt::Horizontal) {
		switch (section) {
			case 0: return QString("BrowseName");
			case 1:	return QString("Value");
			case 2:	return QString("NodeClass");
			case 3:	return QString("DataType");
			case 4:	return QString("NodeId");
			case 5:	return QString("DisplayName");
			case 6:	return QString("Description");
			default: return QString("Column %1").arg(section);
		}
	} else
		return QString("Row %1").arg(section);
}

QModelIndex PVOpcUaTreeModel::index(int row, int column, const QModelIndex& parent) const
{
	if (!hasIndex(row, column, parent))
		return {};

	PVOpcUaTreeItem* item = nullptr;

	if (!parent.isValid()) {
		item = m_root_item.get();
	} else {
		item = static_cast<PVOpcUaTreeItem*>(parent.internalPointer())->child(row);
	}

	if (item)
		return createIndex(row, column, item);
	else
		return {};
}

QModelIndex PVOpcUaTreeModel::parent(const QModelIndex& index) const
{
	if (!index.isValid())
		return {};

	auto child_item = static_cast<PVOpcUaTreeItem*>(index.internalPointer());
	auto parent_item = child_item->parentItem();

	if (child_item == m_root_item.get() || !parent_item)
		return {};

	return createIndex(parent_item->row(), 0, parent_item);
}

int PVOpcUaTreeModel::rowCount(const QModelIndex& parent) const
{
	PVOpcUaTreeItem* parent_item;
	if (!m_client)
		return 0;

	if (parent.column() > 0)
		return 0;

	if (!parent.isValid())
		return 1; // only one root item
	else
		parent_item = static_cast<PVOpcUaTreeItem*>(parent.internalPointer());

	if (!parent_item)
		return 0;

	return parent_item->childCount();
}

int PVOpcUaTreeModel::columnCount(const QModelIndex& parent) const
{
	if (parent.isValid())
		return static_cast<PVOpcUaTreeItem*>(parent.internalPointer())->columnCount();
	else if (m_root_item)
		return m_root_item->columnCount();
	else
		return 0;
}

Qt::ItemFlags PVOpcUaTreeModel::flags(const QModelIndex& index) const
{
	bool history_access = static_cast<PVOpcUaTreeItem*>(index.internalPointer())->has_history_access();
	if (history_access) {
		return QAbstractItemModel::flags(index);
	} else {
		return QAbstractItemModel::flags(index) & ~Qt::ItemIsSelectable;
	}
}

} // namespace PVRush
