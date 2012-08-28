/**
 * \file PVListingSortFilterProxyModel.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVINSPECTOR_PVLISTINGFILTERPROXYMODEL_H
#define PVINSPECTOR_PVLISTINGFILTERPROXYMODEL_H

#include <picviz/PVDefaultSortingFunc.h>
#include <picviz/PVSortingFunc.h>
#include <picviz/PVView_types.h>

#include <pvguiqt/PVSortFilterProxyModel.h>

namespace PVGuiQt {

class PVListingSortFilterProxyModel: public PVSortFilterProxyModel
{
public:
	PVListingSortFilterProxyModel(Picviz::PVView_sp& lib_view, QObject* parent = NULL);

public:
	void refresh_filter();

protected:
	bool less_than(const QModelIndex &left, const QModelIndex &right) const;
	bool is_equal(const QModelIndex &left, const QModelIndex &right) const;
	void sort(int column, Qt::SortOrder order);
	void sort_indexes(int column, Qt::SortOrder order, vec_indexes_t& vec_idxes);
	void filter_source_indexes(vec_indexes_t const& src_idxes_in, vec_indexes_t& src_idxes_out);

private:
	mutable Picviz::PVSortingFunc_fless _sort_f;
	mutable Picviz::PVSortingFunc_fequals _equals_f;
	Picviz::PVView const& _lib_view;

	// Temporary
	Picviz::PVDefaultSortingFunc _def_sort;

	Q_OBJECT
};

}

#endif
