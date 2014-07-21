
#include <pvkernel/core/PVLogger.h>

#include <picviz/PVDefaultSortingFunc.h>

#include <pvguiqt/PVStringSortProxyModel.h>

#include <tbb/parallel_sort.h>

PVGuiQt::PVStringSortProxyModel::PVStringSortProxyModel(QTableView* view, QObject* parent) :
	PVSortFilterProxyModel(view, parent)
{
	set_default_qt_order_func();
}


void PVGuiQt::PVStringSortProxyModel::set_qt_order_func(const Picviz::PVQtSortingFunc_flesser& lt_t)
{
	_qt_lesser_f = lt_t;
}

void PVGuiQt::PVStringSortProxyModel::set_default_qt_order_func()
{
	Picviz::PVDefaultSortingFunc sf;
	set_qt_order_func(sf.qt_f_lesser());
}

bool PVGuiQt::PVStringSortProxyModel::is_equal(const QModelIndex& left, const QModelIndex& right) const
{
	return sourceModel()->data(left).toString() == sourceModel()->data(right).toString();
}

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
