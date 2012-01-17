#include <pvkernel/core/general.h>

#include <PVSortFilterProxyModel.h>
#include <PVSortFilterProxyModel_impl.h>

#include <algorithm>
#include <assert.h>

PVInspector::PVSortFilterProxyModel::PVSortFilterProxyModel(QObject* parent):
	QAbstractProxyModel(parent)
{
	_sort_idx = -1;
	set_dynamic_sort(false);
}

void PVInspector::PVSortFilterProxyModel::init_default_sort()
{
	int row_count = sourceModel()->rowCount();
	_vec_sort_m2s.clear();
	_vec_sort_m2s.reserve(row_count);
	for (int i = 0; i < row_count; i++) {
		_vec_sort_m2s.push_back(i);
	}
}

void PVInspector::PVSortFilterProxyModel::reverse_sort_order()
{
	// In-place reverse of our 2 caches
	emit layoutAboutToBeChanged();
	std::reverse(_vec_sort_m2s.begin(), _vec_sort_m2s.end());
	std::reverse(_vec_filtered_m2s.begin(), _vec_filtered_m2s.end());
	_cur_order = (Qt::SortOrder) !_cur_order;
	emit layoutChanged();
}

void PVInspector::PVSortFilterProxyModel::do_filter()
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

void PVInspector::PVSortFilterProxyModel::do_sort(int column, Qt::SortOrder order)
{
	assert(column >= 0 && column < sourceModel()->columnCount());
	if (order == Qt::AscendingOrder) {
		__impl::PVSortProxyAsc s(this, column);
		std::stable_sort(_vec_sort_m2s.begin(), _vec_sort_m2s.end(), s);
	}
	else {
		__impl::PVSortProxyDesc s(this, column);
		std::stable_sort(_vec_sort_m2s.begin(), _vec_sort_m2s.end(), s);
	}
	_sort_idx = column;
	_cur_order = order;
}

void PVInspector::PVSortFilterProxyModel::sort(int column, Qt::SortOrder order)
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

void PVInspector::PVSortFilterProxyModel::reprocess_source()
{
	if (_dyn_sort && _sort_idx >= 0 && _sort_idx < sourceModel()->columnCount()) {
		do_sort(_sort_idx, _cur_order);
	}
	else {
		init_default_sort();
	}

	do_filter();
}

void PVInspector::PVSortFilterProxyModel::invalidate_sort()
{
	// Force a sort if suitable
	if (_sort_idx >= 0 && _sort_idx < sourceModel()->columnCount()) {
		do_sort(_sort_idx, _cur_order);
		do_filter();
	}
}

void PVInspector::PVSortFilterProxyModel::invalidate_filter()
{
	// Force a computation of the filter
	do_filter();
}

void PVInspector::PVSortFilterProxyModel::invalidate_all()
{
	// Force a sort if suitable
	if (_sort_idx >= 0 && _sort_idx < sourceModel()->columnCount()) {
		do_sort(_sort_idx, _cur_order);
	}
	// And reprocess the filter everytime
	do_filter();
}

QModelIndex PVInspector::PVSortFilterProxyModel::mapFromSource(QModelIndex const& src_idx) const
{
	// source to proxy
	if (!src_idx.isValid()) {
		return QModelIndex();
	}

	PVLOG_WARN("!!!! Reversed search done !!!!\n");
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

QModelIndex PVInspector::PVSortFilterProxyModel::mapToSource(QModelIndex const& src_idx) const
{
	// proxy to source
	if (!src_idx.isValid()) {
		return QModelIndex();
	}

	if (src_idx.row() >= _vec_filtered_m2s.size()) {
		return QModelIndex();
	}

	return sourceModel()->index(_vec_filtered_m2s.at(src_idx.row()), src_idx.column(), QModelIndex());
}

void PVInspector::PVSortFilterProxyModel::filter_source_indexes(vec_indexes_t const& src_idxes_in, vec_indexes_t& src_idxes_out)
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

void PVInspector::PVSortFilterProxyModel::setSourceModel(QAbstractItemModel* model)
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

int PVInspector::PVSortFilterProxyModel::rowCount(const QModelIndex& parent) const
{
	if (parent.isValid()) {
		return 0;
	}

	return _vec_filtered_m2s.size();
}

int PVInspector::PVSortFilterProxyModel::columnCount(const QModelIndex& parent) const
{
	return sourceModel()->columnCount(parent);
}

QModelIndex PVInspector::PVSortFilterProxyModel::index(int row, int col, const QModelIndex& parent) const
{
	if (parent.isValid()) {
		return QModelIndex();
	}

	return createIndex(row, col);
}

QModelIndex PVInspector::PVSortFilterProxyModel::parent(const QModelIndex& idx) const
{
	return QModelIndex();
}

void PVInspector::PVSortFilterProxyModel::src_layout_about_changed()
{
	emit layoutAboutToBeChanged();
}

void PVInspector::PVSortFilterProxyModel::src_layout_changed()
{
	reprocess_source();
	emit layoutChanged();
}

void PVInspector::PVSortFilterProxyModel::src_model_about_reset()
{
	beginResetModel();
}

void PVInspector::PVSortFilterProxyModel::src_model_reset()
{
	reprocess_source();
	endResetModel();
}
