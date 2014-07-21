/**
 * \file PVStringSortProxyModel.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVGUIQT_PVSTRINGSORTPROXYMODEL_H
#define PVGUIQT_PVSTRINGSORTPROXYMODEL_H

#include <picviz/PVSortingFunc.h>

#include <pvguiqt/PVSortFilterProxyModel.h>

namespace PVGuiQt {

class PVStringSortProxyModel: public PVSortFilterProxyModel
{
public:
	PVStringSortProxyModel(QTableView* view, QObject* parent = nullptr);

	/**
	 * RH: the parameter has no constness because GCC seems to have a
	 * problem with purity and const methods...
	 */
	void set_qt_order_func(const Picviz::PVQtSortingFunc_flesser& lt_t);

	void set_default_qt_order_func();

protected:
	bool less_than(const QModelIndex& /*left*/, const QModelIndex& /*right*/) const override { return false; }

	bool is_equal(const QModelIndex& left, const QModelIndex &right) const override;

	void sort_indexes(int column, Qt::SortOrder order, vec_indexes_t& vec_idxes, tbb::task_group_context* ctxt = NULL) override;
	void filter_source_indexes(vec_indexes_t const& src_idxes_in, vec_indexes_t& src_idxes_out) override { src_idxes_out = src_idxes_in; }

private:
	mutable Picviz::PVQtSortingFunc_flesser _qt_lesser_f;
};

}

#endif
