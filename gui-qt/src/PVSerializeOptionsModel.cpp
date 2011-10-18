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
	PVCore::PVSerializeObject::list_childs_t const& childs = so_parent->childs();
	return childs;
}

QModelIndex PVInspector::PVSerializeOptionsModel::index(int row, int column, const QModelIndex& parent) const
{
	// Column is always 0 (see columnCount), but asserts it
	assert(column == 0);

	// Create indexes with a pointer to the corresponding PVSerilizeObject in our tree
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
		default:
			break;
	};

	return QVariant();
}

//QVariant headerData(int section, Qt::Orientation orientation, int role) const;
//Qt::ItemFlags flags(const QModelIndex &index) const;
QModelIndex PVInspector::PVSerializeOptionsModel::parent(const QModelIndex & index) const
{
	PVCore::PVSerializeObject* obj = get_so_index(index);
	PVCore::PVSerializeObject* parent = obj->parent().get();

	if (parent == _options->get_root().get()) {
		return QModelIndex();
	}

	// This is not optimal, but for now let's try it like that...
	
	// Find out the index of the parent within its parent's children list
	PVCore::PVSerializeObject* pp = parent->parent().get();
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
