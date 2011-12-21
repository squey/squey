#include "PVSerializeOptionsModel.h"
#include <pvkernel/core/PVSerializeArchiveOptions.h>

PVInspector::PVSerializeOptionsModel::PVSerializeOptionsModel(PVCore::PVSerializeArchiveOptions_p options, QObject* parent):
	QAbstractItemModel(parent),
	_options(options)
{
}

PVCore::PVSerializeObject::list_childs_t const& PVInspector::PVSerializeOptionsModel::get_childs_index(const QModelIndex& parent) const
{
	PVCore::PVSerializeObject* so_parent;
	if (parent.isValid()) {
		// Take the pointer from the parent
		so_parent = get_so_index(parent);
	}
	else {
		// This is the root of the tree.
		so_parent =  _options->get_root().get();
	}
	PVCore::PVSerializeObject::list_childs_t const& childs = so_parent->visible_childs();
	return childs;
}

QModelIndex PVInspector::PVSerializeOptionsModel::index(int row, int column, const QModelIndex& parent) const
{
	// Column is always 0 (see columnCount), but asserts it
	assert(column == 0);

	// Create indexes with a pointer to the corresponding PVSerializeObject in our tree
	PVCore::PVSerializeObject::list_childs_t const& childs = get_childs_index(parent);
	assert(row < childs.size());
	PVCore::PVSerializeObject* obj = childs.values().at(row).get();
	return createIndex(row, column, (void*) obj);
}

int PVInspector::PVSerializeOptionsModel::rowCount(const QModelIndex &index) const
{
	PVCore::PVSerializeObject::list_childs_t const& childs = get_childs_index(index);
	return childs.size();
}

int PVInspector::PVSerializeOptionsModel::columnCount(const QModelIndex& /*index*/) const
{
	// We have only one column, with the description of the objects
	return 1;
}

QVariant PVInspector::PVSerializeOptionsModel::data(const QModelIndex &index, int role) const
{
	PVCore::PVSerializeObject* obj = get_so_index(index);
	switch (role) {
		case Qt::DisplayRole:
			return QVariant(obj->description());
		case Qt::CheckStateRole:
		{
			if (!obj->is_optional()) {
				return QVariant();
			}
			bool checked = obj->must_write();
			return checked ? Qt::Checked : Qt::Unchecked;
		}
		default:
			break;
	};

	return QVariant();
}

//QVariant headerData(int section, Qt::Orientation orientation, int role) const;
Qt::ItemFlags PVInspector::PVSerializeOptionsModel::flags(const QModelIndex &index) const
{
	Qt::ItemFlags flags = Qt::ItemIsSelectable | Qt::ItemIsEnabled;
	PVCore::PVSerializeObject* obj = get_so_index(index);
	if (obj->is_optional()) {
		flags |= Qt::ItemIsUserCheckable | Qt::ItemIsEditable;
	}
	return flags;
}

QModelIndex PVInspector::PVSerializeOptionsModel::parent(const QModelIndex & index) const
{
	PVCore::PVSerializeObject* obj = get_so_index(index);
	PVCore::PVSerializeObject* parent = obj->parent();

	if (parent == _options->get_root().get()) {
		return QModelIndex();
	}

	// This is not optimal, but for now let's try it like that...
	
	// Find out the index of the parent within its parent's children list
	PVCore::PVSerializeObject* pp = parent->parent();
	PVCore::PVSerializeObject::list_childs_t const& childs = pp->childs();
	QList<PVCore::PVSerializeObject_p> childs_p = childs.values();
	QList<PVCore::PVSerializeObject_p>::const_iterator it;
	int idx = 0;
	bool found = false;
	for (it = childs_p.begin(); it != childs_p.end(); it++) {
		if (it->get() == parent) {
			found = true;
			break;
		}
		idx++;
	}
	assert(found);
	return createIndex(idx, 0, parent);
}

PVCore::PVSerializeObject* PVInspector::PVSerializeOptionsModel::get_so_index(const QModelIndex& index) const
{
	PVCore::PVSerializeObject* ret = static_cast<PVCore::PVSerializeObject*>(index.internalPointer());
	assert(ret);
	return ret;
}

bool PVInspector::PVSerializeOptionsModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
	if (role != Qt::CheckStateRole) {
		return false;
	}

	bool checked = value.toBool();
	PVCore::PVSerializeObject* so = get_so_index(index);
	so->set_write(checked);

	emit dataChanged(index, index);
	emitDataChangedChildren(index);

	return true;
}

void PVInspector::PVSerializeOptionsModel::emitDataChangedChildren(const QModelIndex& index_)
{
	int nchildren = rowCount(index_);
	if (nchildren == 0) {
		return;
	}
	
	QModelIndex first = index(0, 0, index_);
	QModelIndex last = index(nchildren-1, 0, index_);
	emit dataChanged(first, last);
	for (int i = 0; i < nchildren; i++) {
		QModelIndex child = index(i, 0, index_);
		emitDataChangedChildren(child);
	}
}
