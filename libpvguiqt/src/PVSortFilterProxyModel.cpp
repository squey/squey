/**
 * \file PVSortFilterProxyModel.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVAlgorithms.h>
#include <pvkernel/core/PVParallels.h>
#include <pvkernel/core/PVProgressBox.h>

#include <pvguiqt/PVSortFilterProxyModel.h>
#include <pvguiqt/PVSortFilterProxyModel_impl.h>

#include <algorithm>
#include <assert.h>

#include <tbb/tick_count.h>

#include <boost/bind.hpp>

PVGuiQt::PVSortFilterProxyModel::PVSortFilterProxyModel(QObject* parent):
	QAbstractProxyModel(parent)
{
	_sort_idx = -1;
	set_dynamic_sort(false);
	_sort_time = -1.0;
}

void PVGuiQt::PVSortFilterProxyModel::init_default_sort()
{
	int row_count = sourceModel()->rowCount();
	_vec_sort_m2s.clear();
	_vec_sort_m2s.reserve(row_count);
	for (int i = 0; i < row_count; i++) {
		_vec_sort_m2s.push_back(i);
	}
	_sort_idx = -1;
}

void PVGuiQt::PVSortFilterProxyModel::reset_to_default_ordering()
{
	init_default_sort();
	do_filter();
}

void PVGuiQt::PVSortFilterProxyModel::reset_to_default_ordering_or_reverse()
{
	if (_sort_idx == -1) {
		// Special case where a std::reverse is really wanted !
		emit layoutAboutToBeChanged();
		std::reverse(_vec_sort_m2s.begin(), _vec_sort_m2s.end());
		std::reverse(_vec_filtered_m2s.begin(), _vec_filtered_m2s.end());
		emit layoutChanged();
		_cur_order = (Qt::SortOrder) !_cur_order;
	}
	else {
		reset_to_default_ordering();
	}
}

bool PVGuiQt::PVSortFilterProxyModel::__reverse_sort_order()
{
	// We have to be invariant about this varaible, which may be modified by __do_sort.
	int sort_idx = _sort_idx;
	__impl::PVSortProxyComp comp(this, _sort_idx);
	// If the stable reverse code takes more than 1/5th of the stable code, use the stable reverse code
	assert(_sort_time != -1.0);
	if (_sort_time == -1.0) {
		// Should not happen !!
		__do_sort(_sort_idx, (Qt::SortOrder) !_cur_order);
		return true;
	}
	double time_adapt = _sort_time/2.0;
	PVCore::launch_adaptive(
			boost::bind(&PVCore::stable_sort_reverse<vec_indexes_t::iterator, __impl::PVSortProxyComp, void()>, _vec_sort_m2s.begin(), _vec_sort_m2s.end(), comp, boost::ref(boost::this_thread::interruption_point)),
			boost::bind(&PVSortFilterProxyModel::__do_sort, this, _sort_idx, (Qt::SortOrder) !_cur_order),
			boost::posix_time::milliseconds(time_adapt*1000)
			);
	_sort_idx = sort_idx;
	return true;
}

void PVGuiQt::PVSortFilterProxyModel::reverse_sort_order()
{
	// In-place reverse of our first cache
	bool changed = false;
	QWidget* parent_ = dynamic_cast<QWidget*>(QObject::parent());
	PVCore::PVProgressBox* box = new PVCore::PVProgressBox(tr("Reverse sorting order..."), parent_);
	box->set_enable_cancel(false);
	PVCore::PVProgressBox::progress(boost::bind(&PVGuiQt::PVSortFilterProxyModel::__reverse_sort_order, this), box, changed);
	if (changed) {
		// If the sorting order has changed, reprocess the filter
		do_filter();
	}
	_cur_order = (Qt::SortOrder) !_cur_order;
}

void PVGuiQt::PVSortFilterProxyModel::do_filter()
{
	vec_indexes_t tmp;
	filter_source_indexes(_vec_sort_m2s, tmp);
	if (tmp.size() == _vec_filtered_m2s.size()) {
		emit layoutAboutToBeChanged();
		_vec_filtered_m2s = tmp;
		emit layoutChanged();
	}
	else {
		beginResetModel();
		_vec_filtered_m2s = tmp;
		endResetModel();
	}
}

void PVGuiQt::PVSortFilterProxyModel::sort_indexes(int column, Qt::SortOrder order, vec_indexes_t& vec_idxes)
{
	if (order == Qt::AscendingOrder) {
		__impl::PVSortProxyAsc s(this, column);
		std::stable_sort(vec_idxes.begin(), vec_idxes.end(), s);
	}
	else {
		__impl::PVSortProxyDesc s(this, column);
		std::stable_sort(vec_idxes.begin(), vec_idxes.end(), s);
	}
}

void PVGuiQt::PVSortFilterProxyModel::__do_sort(int column, Qt::SortOrder order)
{
	tbb::tick_count start = tbb::tick_count::now();
	init_default_sort();
	sort_indexes(column, order, _vec_sort_m2s);
	tbb::tick_count end = tbb::tick_count::now();
	PVLOG_INFO("Sorting took %0.4f seconds.\n", (end-start).seconds());
	_sort_time =(end-start).seconds();
}

void PVGuiQt::PVSortFilterProxyModel::do_sort(int column, Qt::SortOrder order)
{
	assert(column >= 0 && column < sourceModel()->columnCount());
	QWidget* parent_ = dynamic_cast<QWidget*>(QObject::parent());
	PVCore::PVProgressBox* box = new PVCore::PVProgressBox(tr("Sorting..."), parent_);
	box->set_enable_cancel(false);
	PVCore::PVProgressBox::progress(boost::bind(&PVGuiQt::PVSortFilterProxyModel::__do_sort, this, column, order), box);
	_sort_idx = column;
	_cur_order = order;
}

void PVGuiQt::PVSortFilterProxyModel::sort(int column, Qt::SortOrder order)
{
	if (column == -1) {
		init_default_sort();
		do_filter();
		return;
	}

	if (_sort_idx == column && _cur_order != order) {
		if (_cur_order != order) {
			reverse_sort_order();
		}
		return;
	}

	do_sort(column, order);
	do_filter();
}

void PVGuiQt::PVSortFilterProxyModel::reprocess_source()
{
	if (_dyn_sort && _sort_idx >= 0 && _sort_idx < sourceModel()->columnCount()) {
		if (sourceModel()->rowCount() != _vec_sort_m2s.size()) {
			// Size has changed, recreate a default sort array.
			int sort_idx = _sort_idx;
			init_default_sort();
			_sort_idx = sort_idx;
		}
		do_sort(_sort_idx, _cur_order);
	}
	else {
		init_default_sort();
	}

	do_filter();
}

void PVGuiQt::PVSortFilterProxyModel::invalidate_sort()
{
	// Force a sort if suitable
	if (_sort_idx >= 0 && _sort_idx < sourceModel()->columnCount()) {
		do_sort(_sort_idx, _cur_order);
		do_filter();
	}
}

void PVGuiQt::PVSortFilterProxyModel::invalidate_filter()
{
	// Force a computation of the filter
	do_filter();
}

void PVGuiQt::PVSortFilterProxyModel::invalidate_all()
{
	// Force a sort if suitable
	if (sourceModel() != NULL && sourceModel()->rowCount() != _vec_sort_m2s.size()) {
		// Size has changed, recreate a default sort array.
		int sort_idx = _sort_idx;
		init_default_sort();
		_sort_idx = sort_idx;
	}
	if (_sort_idx >= 0 && _sort_idx < sourceModel()->columnCount()) {
		do_sort(_sort_idx, _cur_order);
	}
	// And reprocess the filter everytime
	do_filter();
}

QModelIndex PVGuiQt::PVSortFilterProxyModel::mapFromSource(QModelIndex const& src_idx) const
{
	// source to proxy
	if (!src_idx.isValid()) {
		return QModelIndex();
	}

	// Reverse search of the original index to save in the source index
	int search_src_row = src_idx.row();
	int proxy_row;
	for (proxy_row = 0; proxy_row < _vec_filtered_m2s.size(); proxy_row++) {
		int src_row = _vec_filtered_m2s.at(proxy_row);
		if (src_row == search_src_row) {
			return index(proxy_row, src_idx.column(), QModelIndex());
		}
	}

	return QModelIndex();
}

QModelIndex PVGuiQt::PVSortFilterProxyModel::mapToSource(QModelIndex const& src_idx) const
{
	// proxy to source
	if (!src_idx.isValid()) {
		return QModelIndex();
	}

	if (src_idx.row() == 0 && _vec_filtered_m2s.size() == 0) {
		// Special case where no lines are displayed but we still want header information !!
		// This function is called by QAbstractProxyModel::headerData to find out the good column.
		// TODO: we should use the axis combination in the proxy, and not in the model.
		return sourceModel()->index(0, src_idx.column(), QModelIndex());
	}

	if (src_idx.row() >= _vec_filtered_m2s.size()) {
		return QModelIndex();
	}

	return sourceModel()->index(_vec_filtered_m2s.at(src_idx.row()), src_idx.column(), QModelIndex());
}

void PVGuiQt::PVSortFilterProxyModel::filter_source_indexes(vec_indexes_t const& src_idxes_in, vec_indexes_t& src_idxes_out)
{
	vec_indexes_t::const_iterator it;
	src_idxes_out.clear();
	src_idxes_out.reserve(src_idxes_in.size());
	for (it = src_idxes_in.begin(); it != src_idxes_in.end(); it++) {
		int idx = *it;
		if (filter_source_index(idx)) {
			src_idxes_out.push_back(idx);
		}
	}
}

void PVGuiQt::PVSortFilterProxyModel::setSourceModel(QAbstractItemModel* model)
{
	beginResetModel();

	QAbstractItemModel* prev_model = sourceModel();
	if (prev_model) {
		disconnect(prev_model, SIGNAL(layoutAboutToBeChanged()), this, SLOT(src_layout_about_changed()));
		disconnect(prev_model, SIGNAL(layoutChanged()), this, SLOT(src_layout_changed()));
		disconnect(prev_model, SIGNAL(modelAboutToBeReset()), this, SLOT(src_model_about_reset()));
		disconnect(prev_model, SIGNAL(modelReset()), this, SLOT(src_model_reset()));
	}

	QAbstractProxyModel::setSourceModel(model);
	init_default_sort();
	do_filter();

	connect(model, SIGNAL(layoutAboutToBeChanged()), this, SLOT(src_layout_about_changed()));
	connect(model, SIGNAL(layoutChanged()), this, SLOT(src_layout_changed()));
	connect(model, SIGNAL(modelAboutToBeReset()), this,SLOT(src_model_about_reset()));
	connect(model, SIGNAL(modelReset()), this, SLOT(src_model_reset()));

	endResetModel();
}

int PVGuiQt::PVSortFilterProxyModel::rowCount(const QModelIndex& parent) const
{
	if (parent.isValid()) {
		return 0;
	}

	return _vec_filtered_m2s.size();
}

int PVGuiQt::PVSortFilterProxyModel::columnCount(const QModelIndex& parent) const
{
	return sourceModel()->columnCount(parent);
}

QModelIndex PVGuiQt::PVSortFilterProxyModel::index(int row, int col, const QModelIndex& parent) const
{
	if (parent.isValid()) {
		return QModelIndex();
	}

	return createIndex(row, col);
}

QModelIndex PVGuiQt::PVSortFilterProxyModel::parent(const QModelIndex& /*idx*/) const
{
	return QModelIndex();
}

void PVGuiQt::PVSortFilterProxyModel::src_layout_about_changed()
{
	emit layoutAboutToBeChanged();
}

void PVGuiQt::PVSortFilterProxyModel::src_layout_changed()
{
	reprocess_source();
	emit layoutChanged();
}

void PVGuiQt::PVSortFilterProxyModel::src_model_about_reset()
{
	beginResetModel();
}

void PVGuiQt::PVSortFilterProxyModel::src_model_reset()
{
	reprocess_source();
	endResetModel();
}