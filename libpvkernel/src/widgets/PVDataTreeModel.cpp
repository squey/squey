/**
 * \file PVViewsModel.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVDataTreeObject.h>
#include <pvkernel/widgets/PVDataTreeModel.h>

#include <QFont>

PVWidgets::PVDataTreeModel::PVDataTreeModel(PVCore::PVDataTreeObjectWithChildrenBase& root, QObject* parent):
	QAbstractItemModel(parent),
	_root(&root)
{
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
		// Root element: get its children !
		children = _root->get_children_base();
	}
	else {
		// Get children of this parent !
		PVCore::PVDataTreeObjectWithChildrenBase* node_obj = get_object(parent)->cast_with_children();
		assert(node_obj);
		children = node_obj->get_children_base();
	}
	assert(row < children.size());
	
	return createIndex(row, column, children.at(row));
}

int PVWidgets::PVDataTreeModel::rowCount(const QModelIndex &index) const
{
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
		case Qt::FontRole:
		{
			/*
			if (!node_obj.is_plotted()) {
				return QVariant();
			}
			Picviz::PVPlotted* plotted = node_obj.as_plotted();
			if (plotted->current_view() == _src.current_view()) {
				QFont font;
				font.setBold(true);
				return font;
			}
			return QVariant();*/
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
