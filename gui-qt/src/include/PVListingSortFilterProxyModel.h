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

#include <PVSortFilterProxyModel.h>

namespace Picviz {
class PVStateMachine;
}

namespace PVInspector {

class PVTabSplitter;

class PVListingSortFilterProxyModel: public PVSortFilterProxyModel
{
public:
	PVListingSortFilterProxyModel(PVTabSplitter* tab_parent, QObject* parent = NULL);

public:
	void refresh_filter();
	void reset_lib_view();

protected:
	bool less_than(const QModelIndex &left, const QModelIndex &right) const;
	bool is_equal(const QModelIndex &left, const QModelIndex &right) const;
	void sort(int column, Qt::SortOrder order);
	void sort_indexes(int column, Qt::SortOrder order, vec_indexes_t& vec_idxes);
	void filter_source_indexes(vec_indexes_t const& src_idxes_in, vec_indexes_t& src_idxes_out);

private:
	mutable Picviz::PVSortingFunc_fless _sort_f;
	mutable Picviz::PVSortingFunc_fequals _equals_f;
	Picviz::PVView* _lib_view;
	Picviz::PVStateMachine* _state_machine;
	PVTabSplitter* _tab_parent;

	// Temporary
	Picviz::PVDefaultSortingFunc _def_sort;

	Q_OBJECT
};

}

#endif
