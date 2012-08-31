/**
 * \file PVListingSortFilterProxyModel.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVINSPECTOR_PVLISTINGFILTERPROXYMODEL_H
#define PVINSPECTOR_PVLISTINGFILTERPROXYMODEL_H

#include <picviz/PVDefaultSortingFunc.h>
#include <picviz/PVSortingFunc.h>
#include <picviz/PVView.h>

#include <pvhive/PVCallHelper.h>
#include <pvhive/PVObserverSignal.h>

#include <pvguiqt/PVSortFilterProxyModel.h>

namespace PVGuiQt {

namespace __impl {

class PVSelFuncObserver: public PVHive::PVFuncObserverSignal<Picviz::PVView, FUNC(Picviz::PVView::process_from_selection)>
{
	Q_OBJECT

public:
	void about_to_be_updated(const arguments_deep_copy_type&) const override { emit about_to_refresh_sel(); }
	void update(const arguments_deep_copy_type&) const override { emit refresh_sel(); }

signals:
	void about_to_refresh_sel() const;
	void refresh_sel() const;
};

}

class PVListingSortFilterProxyModel: public PVSortFilterProxyModel
{
	Q_OBJECT

public:
	PVListingSortFilterProxyModel(Picviz::PVView_sp& lib_view, QObject* parent = NULL);

public slots:
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

	// Observers
	//__impl::PVSelFuncObserver _obs_sel;
	PVHive::PVObserverSignal<Picviz::PVLayer> _obs_output_layer;

	// Temporary
	Picviz::PVDefaultSortingFunc _def_sort;
};

}

#endif
