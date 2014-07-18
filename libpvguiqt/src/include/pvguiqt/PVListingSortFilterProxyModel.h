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

class PVListingSortFilterProxyModel;


namespace __impl {

struct PVListingVisibilityObserver: public PVHive::PVFuncObserver<Picviz::PVView, FUNC(Picviz::PVView::toggle_listing_unselected_visibility)>
{
	PVListingVisibilityObserver(PVGuiQt::PVListingSortFilterProxyModel* parent):
		_parent(parent)
	{ }

protected:
	virtual void update(arguments_type const& args) const;

private:
	PVGuiQt::PVListingSortFilterProxyModel* _parent;
};

struct PVListingVisibilityZombieObserver: public PVHive::PVFuncObserver<Picviz::PVView, FUNC(Picviz::PVView::toggle_listing_zombie_visibility)>
{
	PVListingVisibilityZombieObserver(PVGuiQt::PVListingSortFilterProxyModel* parent):
		_parent(parent)
	{ }

protected:
	virtual void update(arguments_type const& args) const;

private:
	PVGuiQt::PVListingSortFilterProxyModel* _parent;
};

}

class PVListingSortFilterProxyModel: public PVSortFilterProxyModel
{
	Q_OBJECT

public:
	PVListingSortFilterProxyModel(Picviz::PVView_sp& lib_view, QTableView* view, QObject* parent = NULL);

public slots:
	void refresh_filter();

protected:
	bool less_than(const QModelIndex &left, const QModelIndex &right) const;
	bool is_equal(const QModelIndex &left, const QModelIndex &right) const;
	void sort(int column, Qt::SortOrder order);
	void sort_indexes(int column, Qt::SortOrder order, vec_indexes_t& vec_idxes, tbb::task_group_context* ctxt = NULL);
	void filter_source_indexes(vec_indexes_t const& src_idxes_in, vec_indexes_t& src_idxes_out);

private:
	mutable Picviz::PVSortingFunc_fless _sort_f;
	mutable Picviz::PVSortingFunc_fequals _equals_f;
	Picviz::PVView const& _lib_view;

	// Observers
	//__impl::PVSelFuncObserver _obs_sel;
	PVHive::PVObserverSignal<Picviz::PVLayer> _obs_output_layer;
	PVHive::PVObserverSignal<Picviz::PVSelection> _obs_sel;
	__impl::PVListingVisibilityObserver _obs_vis;
	__impl::PVListingVisibilityZombieObserver _obs_zomb;
};

}

#endif
