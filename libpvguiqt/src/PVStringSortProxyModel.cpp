/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <picviz/PVDefaultSortingFunc.h>
#include <pvguiqt/PVStringSortProxyModel.h>
#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/core/picviz_bench.h>

#include <algorithm>
#include <cassert>

#include <tbb/tick_count.h>
#include <tbb/parallel_sort.h>

#include <QHeaderView>

/******************************************************************************
 *
 * PVGuiQt::PVStringSortProxyModel::PVStringSortProxyModel
 *
 *****************************************************************************/

PVGuiQt::PVStringSortProxyModel::PVStringSortProxyModel(QTableView* view, QObject* parent):
	QAbstractProxyModel(parent),
	_qt_lesser_f(Picviz::PVDefaultSortingFunc().qt_f_lesser()),
	_view(view),
	_sort_idx(-1)
{
}

/******************************************************************************
 *
 * PVGuiQt::PVStringSortProxyModel::init_default_sort
 *
 *****************************************************************************/
void PVGuiQt::PVStringSortProxyModel::init_default_sort()
{
	BENCH_START(b);

	int row_count = sourceModel()->rowCount();

	_vec_sort_m2s.resize(row_count);
	std::iota(_vec_sort_m2s.begin(), _vec_sort_m2s.end(), 0);

	BENCH_END(b, "PVStringSortProxyModel::init_default_sort", 1, 1, row_count, sizeof(int));
}

/******************************************************************************
 *
 * PVGuiQt::PVStringSortProxyModel::__reverse_sort_order
 *
 *****************************************************************************/
bool PVGuiQt::PVStringSortProxyModel::__reverse_sort_order(tbb::task_group_context* ctxt)
{
	// We have to be invariant about this variable, which may be modified by __do_sort.
	int sort_idx = _sort_idx;
	__do_sort(_sort_idx, (Qt::SortOrder) !_cur_order, ctxt);
	bool changed = !ctxt->is_group_execution_cancelled();
	if (changed) {
		_sort_idx = sort_idx;
	}

	return changed;
}

/******************************************************************************
 *
 * PVGuiQt::PVStringSortProxyModel::reverse_sort_order
 *
 *****************************************************************************/
bool PVGuiQt::PVStringSortProxyModel::reverse_sort_order()
{
	// In-place reverse of our first cache
	QWidget* parent_ = dynamic_cast<QWidget*>(QObject::parent());
	PVCore::PVProgressBox* box = new PVCore::PVProgressBox(tr("Reverse sorting order..."), parent_);
	box->set_enable_cancel(true);
	tbb::task_group_context ctxt;
	bool changed = PVCore::PVProgressBox::progress([&](){ __reverse_sort_order(&ctxt); }, ctxt, box);

	return changed;
}

/******************************************************************************
 *
 * PVGuiQt::PVStringSortProxyModel::__do_sort
 *
 *****************************************************************************/
void PVGuiQt::PVStringSortProxyModel::__do_sort(int column, Qt::SortOrder order, tbb::task_group_context* ctxt)
{
	tbb::tick_count start = tbb::tick_count::now();
	init_default_sort();
	sort_indexes(column, order, _vec_sort_m2s, ctxt);
	tbb::tick_count end = tbb::tick_count::now();
	PVLOG_INFO("Sorting took %0.4f seconds.\n", (end-start).seconds());
}

/******************************************************************************
 *
 * PVGuiQt::PVStringSortProxyModel::do_sort
 *
 *****************************************************************************/
bool PVGuiQt::PVStringSortProxyModel::do_sort(int column, Qt::SortOrder order)
{
	assert(column >= 0 && column < sourceModel()->columnCount());
	QWidget* parent_ = dynamic_cast<QWidget*>(QObject::parent());
	PVCore::PVProgressBox* box = new PVCore::PVProgressBox(tr("Sorting..."), parent_);
	box->set_enable_cancel(true);
	tbb::task_group_context ctxt;
	bool changed = PVCore::PVProgressBox::progress([&](){ __do_sort(column, order, &ctxt);}, ctxt, box);

	return changed;
}

/******************************************************************************
 *
 * PVGuiQt::PVStringSortProxyModel::sort
 *
 *****************************************************************************/
void PVGuiQt::PVStringSortProxyModel::sort(int column, Qt::SortOrder order)
{
	bool changed = false;
	emit layoutAboutToBeChanged();

	// This path could be call only from Qt (not checked)
	if (column == -1) {
		init_default_sort();
		emit layoutChanged();
		return;
	}

	if (_sort_idx == column && _cur_order != order) {
		changed = reverse_sort_order();
	}
	else {
		changed = do_sort(column, order);
	}

	if (changed) {
		_sort_idx = column;
		_cur_order = order;
	} else {
		_view->horizontalHeader()->blockSignals(true);
		_view->horizontalHeader()->setSortIndicator(_sort_idx, _cur_order);
		_view->horizontalHeader()->blockSignals(false);
	}

	_view->horizontalHeader()->setSortIndicatorShown(true);
	emit layoutChanged();
}

/******************************************************************************
 *
 * PVGuiQt::PVStringSortProxyModel::reprocess_source
 *
 *****************************************************************************/
void PVGuiQt::PVStringSortProxyModel::reprocess_source()
{
	emit layoutAboutToBeChanged();
	init_default_sort();
	emit layoutChanged();
}

/******************************************************************************
 *
 * PVGuiQt::PVStringSortProxyModel::invalidate_all
 *
 *****************************************************************************/
void PVGuiQt::PVStringSortProxyModel::invalidate_all()
{
	emit layoutAboutToBeChanged();
	// Force a sort if suitable
	if (sourceModel() != nullptr && sourceModel()->rowCount() != (int)_vec_sort_m2s.size()) {
		// Size has changed, recreate a default sort array.
		init_default_sort();
	}
	if (_sort_idx >= 0 && _sort_idx < sourceModel()->columnCount()) {
		do_sort(_sort_idx, _cur_order);
	}
	emit layoutChanged();
}

/******************************************************************************
 *
 * PVGuiQt::PVStringSortProxyModel::mapFromSource
 *
 *****************************************************************************/
QModelIndex PVGuiQt::PVStringSortProxyModel::mapFromSource(QModelIndex const& src_idx) const
{
	// source to proxy
	if (!src_idx.isValid()) {
		return QModelIndex();
	}

	// Reverse search of the original index to save in the source index
	int search_src_row = src_idx.row();
	for (int proxy_row = 0; proxy_row < (int)_vec_sort_m2s.size(); proxy_row++) {
		int src_row =_vec_sort_m2s[proxy_row];
		if (src_row == search_src_row) {
			return index(proxy_row, src_idx.column(), QModelIndex());
		}
	}

	return QModelIndex();
}

/******************************************************************************
 *
 * PVGuiQt::PVStringSortProxyModel::mapToSource
 *
 *****************************************************************************/
QModelIndex PVGuiQt::PVStringSortProxyModel::mapToSource(QModelIndex const& src_idx) const
{
	// proxy to source
	if (!src_idx.isValid()) {
		return QModelIndex();
	}

	if (src_idx.row() == 0 && _vec_sort_m2s.size() == 0) {
		// Special case where no lines are displayed but we still want header information !!
		// This function is called by QAbstractProxyModel::headerData to find out the good column.
		// TODO: we should use the axis combination in the proxy, and not in the model.
		return sourceModel()->index(0, src_idx.column(), QModelIndex());
	}

	if (src_idx.row() >= (int)_vec_sort_m2s.size()) {
		return QModelIndex();
	}

	return sourceModel()->index(_vec_sort_m2s[src_idx.row()], src_idx.column(), QModelIndex());
}

/******************************************************************************
 *
 * PVGuiQt::PVStringSortProxyModel::setSourceModel
 *
 *****************************************************************************/
void PVGuiQt::PVStringSortProxyModel::setSourceModel(QAbstractItemModel* model)
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

	connect(model, SIGNAL(layoutAboutToBeChanged()), this, SLOT(src_layout_about_changed()));
	connect(model, SIGNAL(layoutChanged()), this, SLOT(src_layout_changed()));
	connect(model, SIGNAL(modelAboutToBeReset()), this,SLOT(src_model_about_reset()));
	connect(model, SIGNAL(modelReset()), this, SLOT(src_model_reset()));

	endResetModel();
}

/******************************************************************************
 *
 * PVGuiQt::PVStringSortProxyModel::rowCount
 *
 *****************************************************************************/
int PVGuiQt::PVStringSortProxyModel::rowCount(const QModelIndex& parent) const
{
	return sourceModel()->rowCount(parent);
}

/******************************************************************************
 *
 * PVGuiQt::PVStringSortProxyModel::columnCount
 *
 *****************************************************************************/
int PVGuiQt::PVStringSortProxyModel::columnCount(const QModelIndex& parent) const
{
	return sourceModel()->columnCount(parent);
}

/******************************************************************************
 *
 * PVGuiQt::PVStringSortProxyModel::index
 *
 *****************************************************************************/
QModelIndex PVGuiQt::PVStringSortProxyModel::index(int row, int col, const QModelIndex&) const
{
	return createIndex(row, col);
}

/******************************************************************************
 *
 * PVGuiQt::PVStringSortProxyModel::parent
 *
 *****************************************************************************/
QModelIndex PVGuiQt::PVStringSortProxyModel::parent(const QModelIndex& /*idx*/) const
{
	return QModelIndex();
}

/******************************************************************************
 *
 * PVGuiQt::PVStringSortProxyModel::src_layout_about_changed
 *
 *****************************************************************************/
void PVGuiQt::PVStringSortProxyModel::src_layout_about_changed()
{
	emit layoutAboutToBeChanged();
}

/******************************************************************************
 *
 * PVGuiQt::PVStringSortProxyModel::src_layout_changed
 *
 *****************************************************************************/
void PVGuiQt::PVStringSortProxyModel::src_layout_changed()
{
	reprocess_source();
	emit layoutChanged();
}

/******************************************************************************
 *
 * PVGuiQt::PVStringSortProxyModel::src_model_about_reset
 *
 *****************************************************************************/
void PVGuiQt::PVStringSortProxyModel::src_model_about_reset()
{
	beginResetModel();
}

/******************************************************************************
 *
 * PVGuiQt::PVStringSortProxyModel::src_model_reset
 *
 *****************************************************************************/
void PVGuiQt::PVStringSortProxyModel::src_model_reset()
{
	reprocess_source();
	endResetModel();
}

/******************************************************************************
 *
 * PVGuiQt::PVStringSortProxyModel::set_qt_order_func
 *
 *****************************************************************************/
void PVGuiQt::PVStringSortProxyModel::set_qt_order_func(const Picviz::PVQtSortingFunc_flesser& lt_t)
{
	_qt_lesser_f = lt_t;
}

/******************************************************************************
 *
 * PVGuiQt::PVStringSortProxyModel::sort_indexes
 *
 *****************************************************************************/
void PVGuiQt::PVStringSortProxyModel::sort_indexes(int column, Qt::SortOrder order, vec_indexes_t& vec_idxes, tbb::task_group_context* /*ctxt*/)
{
	switch (column) {
		case 0: {
			if (order == Qt::AscendingOrder) {
				tbb::parallel_sort(vec_idxes.begin(), vec_idxes.end(),
					[&](int i1, int i2)
					{
						const int column_ = column;
						QAbstractItemModel* const src_model = this->sourceModel();
						return _qt_lesser_f(src_model->data(src_model->index(i1, column_)).toString(),
						                    src_model->data(src_model->index(i2, column_)).toString());
					});
			}
			else {
				tbb::parallel_sort(vec_idxes.begin(), vec_idxes.end(),
					[&](int i1, int i2)
					{
						const int column_ = column;
						QAbstractItemModel* const src_model = this->sourceModel();
						// s1 > s2 <=> s2 < s1 :-p
						return _qt_lesser_f(src_model->data(src_model->index(i2, column_)).toString(),
						                    src_model->data(src_model->index(i1, column_)).toString());
					});
			};
			break;
		}
		case 1: {
			if (order == Qt::AscendingOrder) {
				tbb::parallel_sort(vec_idxes.begin(), vec_idxes.end(),
					[&](int i1, int i2)
					{
						const int column_ = column;
						QAbstractItemModel* const src_model = this->sourceModel();
						const uint64_t s1 = src_model->data(src_model->index(i1, column_), Qt::UserRole).toULongLong();
						const uint64_t s2 = src_model->data(src_model->index(i2, column_), Qt::UserRole).toULongLong();
						return s1 < s2;
					});

			}
			else {
				tbb::parallel_sort(vec_idxes.begin(), vec_idxes.end(),
					[&](int i1, int i2)
					{
						const int column_ = column;
						QAbstractItemModel* const src_model = this->sourceModel();
						const size_t s1 = src_model->data(src_model->index(i1, column_), Qt::UserRole).toULongLong();
						const size_t s2 = src_model->data(src_model->index(i2, column_), Qt::UserRole).toULongLong();
						return s1 > s2;
					});
			}
			break;
		}
		default: {
			PVLOG_ERROR("No sorting function avaible for column %d\n", column);
			break;
		}
	}
}
