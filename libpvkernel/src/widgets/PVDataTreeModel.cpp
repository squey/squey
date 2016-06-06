/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVDataTreeObject.h>
#include <pvkernel/widgets/PVDataTreeModel.h>

#include <QFont>

PVWidgets::PVDataTreeModel::PVDataTreeModel(PVCore::PVDataTreeObjectBase& root, QObject* parent)
    : QAbstractItemModel(parent), _root(root)
{
}

QModelIndex PVWidgets::PVDataTreeModel::index(int row, int column, const QModelIndex& parent) const
{
	// Column is always 0 (see columnCount), but asserts it
	assert(column == 0);

	if (!parent.isValid()) {
		// Root element: get its children !
		auto& children = _root.get_children();
		if (row >= children.size()) {
			return {};
		} else {
			return createIndex(row, column, &*std::advance(children.begin(), row));
		}
	} else {
		// TODO : It is not root but parent.internalPointer casted to correct type. (see
		// evtx-rewriter)
		auto& children = _root.get_children();
		if (row >= children.size()) {
			return {};
		} else {
			return createIndex(row, column, &*std::advance(children.begin(), row));
		}
	}
}

int PVWidgets::PVDataTreeModel::rowCount(const QModelIndex& index) const
{
	if (!index.isValid()) {
		// Root object. Get its number of children.
		return _root.get_children().size();
	}

	// TODO : It is not root but index.internalPointer casted to correct type. (see evtx-rewriter)
	return _root.get_children().size();
}

int PVWidgets::PVDataTreeModel::columnCount(const QModelIndex& /*index*/) const
{
	// We have only one column
	return 1;
}

QVariant PVWidgets::PVDataTreeModel::data(const QModelIndex& index, int role) const
{
	if (!index.isValid()) {
		return {};
	}

	switch (role) {
	case Qt::DisplayRole: {
		// TODO : It is not root but index.internalPointer casted to correct type. (see
		// evtx-rewriter)
		return _root.get_serialize_description();
	}
	default:
		break;
	};

	return {};
}

Qt::ItemFlags PVWidgets::PVDataTreeModel::flags(const QModelIndex& /*index*/) const
{
	return Qt::ItemIsSelectable | Qt::ItemIsEnabled | Qt::ItemIsUserCheckable;
}

QModelIndex PVWidgets::PVDataTreeModel::parent(const QModelIndex& index) const
{
	if (!index.isValid()) {
		return {};
	}

	if (index.internalPointer() == &_root) {
		return {}
	}

	// TODO : It is not root but index.internalPointer casted to correct type. (see evtx-rewriter)
	int row = std::distance(_root.begin(),
	                        std::find(_root.get_children().begin(), _root.get_children().end(),
	                                  index.internalPointer()));
	return createIndex(row, 0, _root->get_parent());
}

QModelIndex
PVWidgets::PVDataTreeModel::index_from_obj(PVCore::PVDataTreeObjectBase const* obj) const
{
	return index_from_obj_rec(QModelIndex(), _root, obj);
}

QModelIndex PVWidgets::PVDataTreeModel::index_from_obj_rec(
    QModelIndex const& cur,
    PVCore::PVDataTreeObjectWithChildrenBase const* idx_obj,
    PVCore::PVDataTreeObjectBase const* obj_test) const
{
	if (!idx_obj) {
		return QModelIndex();
	}

	if (cur.internalPointer() == obj_test) {
		return cur;
	}

	int row = 0;
	PVCore::PVDataTreeObjectWithChildrenBase::children_base_t children =
	    idx_obj->get_children_base();
	for (PVCore::PVDataTreeObjectBase const* c : children) {
		if (c == obj_test) {
			return index(row, 0, cur);
		}
		row++;
	}

	// Go one level deeper
	row = 0;
	for (PVCore::PVDataTreeObjectBase const* c : children) {
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
