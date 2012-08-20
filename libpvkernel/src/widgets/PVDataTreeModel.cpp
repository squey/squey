/**
 * \file PVViewsModel.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVDataTreeObject.h>
#include <pvkernel/widgets/PVDataTreeModel.h>

#include <QFont>

PVWidgets::PVDataTreeModel::PVDataTreeModel(PVCore::PVDataTreeObjectBase& root, QObject* parent):
	QAbstractItemModel(parent)
{
	_root = root.cast_with_children();
}

PVCore::PVDataTreeObjectBase* PVWidgets::PVDataTreeModel::get_object(QModelIndex const& index) const
{
	assert(index.isValid());
	return (static_cast<PVCore::PVDataTreeObjectBase*>(index.internalPointer()));
}

QModelIndex PVWidgets::PVDataTreeModel::index(int row, int column, const QModelIndex& parent) const
{
	// Column is always 0 (see columnCount), but asserts it
	assert(column == 0);

	PVCore::PVDataTreeObjectWithChildrenBase::children_base_t children;
	if (!parent.isValid()) {
		assert(_root);
		// Root element: get its children !
		children = _root->get_children_base();
	}
	else {
		// Get children of this parent !
		PVCore::PVDataTreeObjectWithChildrenBase* node_obj = get_object(parent)->cast_with_children();
		if (!node_obj) {
			return QModelIndex();
		}
		children = node_obj->get_children_base();
	}
	if (row >= children.size()) {
		return QModelIndex();
	}

	PVCore::PVDataTreeObjectBase* final_obj = children.at(row);
	return createIndex(row, column, final_obj);
}

int PVWidgets::PVDataTreeModel::rowCount(const QModelIndex &index) const
{
	if (!_root) {
		return 0;
	}

	if (!index.isValid()) {
		// Root object. Get its number of children.
		return _root->get_children_count();
	}
	
	PVCore::PVDataTreeObjectWithChildrenBase* node_obj = get_object(index)->cast_with_children();
	if (node_obj) {
		// This object has children, so return its count.
		return node_obj->get_children_count();
	}

	// In this case, there is no children left.
	return 0;
}

int PVWidgets::PVDataTreeModel::columnCount(const QModelIndex& /*index*/) const
{
	// We have only one column
	return 1;
}

QVariant PVWidgets::PVDataTreeModel::data(const QModelIndex &index, int role) const
{
	if (!index.isValid()) {
		return QVariant();
	}
	PVCore::PVDataTreeObjectBase* node_obj = get_object(index);
	switch (role) {
		case Qt::DisplayRole:
		{
			return node_obj->get_serialize_description();
		}
		default:
			break;
	};

	return QVariant();
}

Qt::ItemFlags PVWidgets::PVDataTreeModel::flags(const QModelIndex& /*index*/) const
{
	Qt::ItemFlags flags = Qt::ItemIsSelectable | Qt::ItemIsEnabled | Qt::ItemIsUserCheckable;
	return flags;
}

QModelIndex PVWidgets::PVDataTreeModel::parent(const QModelIndex & index) const
{
	if (!index.isValid()) {
		return QModelIndex();
	}

	PVCore::PVDataTreeObjectWithParentBase* const node_obj = get_object(index)->cast_with_parent();
	if (!node_obj) {
		return QModelIndex();
	}
	
	PVCore::PVDataTreeObjectBase* const node_parent = node_obj->get_parent_base();
	assert(_root);
	if (node_parent->cast_with_children() == _root) {
		// Parent is root element, thus no parent.
		return QModelIndex();
	}

	// We need to take the parent of node_parent, and get the index of node_parent into its children.
	PVCore::PVDataTreeObjectWithChildrenBase* const super_parent = node_parent->cast_with_parent()->get_parent_base()->cast_with_children();
	assert(super_parent);

	int idx_parent = 0;
	for (PVCore::PVDataTreeObjectBase* const c : super_parent->get_children_base()) {
		if (c == node_parent) {
			break;
		}
		idx_parent++;
	}

	return createIndex(idx_parent, 0, node_obj->get_parent_base());
}

QModelIndex PVWidgets::PVDataTreeModel::index_from_obj(PVCore::PVDataTreeObjectBase const* obj) const
{
	return index_from_obj_rec(QModelIndex(), _root, obj);
}

QModelIndex PVWidgets::PVDataTreeModel::index_from_obj_rec(QModelIndex const& cur, PVCore::PVDataTreeObjectWithChildrenBase const* idx_obj, PVCore::PVDataTreeObjectBase const* obj_test) const
{
	if (!idx_obj) {
		return QModelIndex();
	}

	if (cur.internalPointer() == obj_test) {
		return cur;
	}

	int row = 0;
	PVCore::PVDataTreeObjectWithChildrenBase::children_base_t children = idx_obj->get_children_base();
	for (PVCore::PVDataTreeObjectBase const* c: children) {
		if (c == obj_test) {
			return index(row, 0, cur);
		}
		row++;
	}

	// Go one level deeper
	row = 0;
	for (PVCore::PVDataTreeObjectBase const* c: children) {
		QModelIndex new_cur = index(row, 0, cur);
		QModelIndex test = index_from_obj_rec(new_cur, c->cast_with_children(), obj_test);
		if (test.isValid()) {
			assert(test.internalPointer() == obj_test);
			return test;
		}
		row++;
	}

	return QModelIndex();
}
