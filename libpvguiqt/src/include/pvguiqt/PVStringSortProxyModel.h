/**
 * \file PVStringSortProxyModel.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVGUIQT_PVSTRINGSORTPROXYMODEL_H
#define PVGUIQT_PVSTRINGSORTPROXYMODEL_H

#include <pvguiqt/PVSortFilterProxyModel.h>

namespace PVGuiQt {

class PVStringSortProxyModel: public PVSortFilterProxyModel
{
public:
	PVStringSortProxyModel(QObject* parent = NULL):
		PVSortFilterProxyModel(parent)
	{ }

protected:
	bool less_than(const QModelIndex &left, const QModelIndex &right) const override { return false; }
	bool is_equal(const QModelIndex &left, const QModelIndex &right) const override; 

	void sort_indexes(int column, Qt::SortOrder order, vec_indexes_t& vec_idxes) override;
	void filter_source_indexes(vec_indexes_t const& src_idxes_in, vec_indexes_t& src_idxes_out) override { src_idxes_out = src_idxes_in; }
};

}

#endif
