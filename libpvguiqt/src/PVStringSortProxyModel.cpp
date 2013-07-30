#include <pvguiqt/PVStringSortProxyModel.h>
#include <tbb/parallel_sort.h>
#include <pvkernel/core/PVLogger.h>

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
						const QString s1 = src_model->data(src_model->index(i1, column_)).toString();
						const QString s2 = src_model->data(src_model->index(i2, column_)).toString();
						return s1 < s2;
					});
			}
			else {
				tbb::parallel_sort(vec_idxes.begin(), vec_idxes.end(),
					[&](int i1, int i2)
					{
						const int column_ = column;
						QAbstractItemModel* const src_model = this->sourceModel();
						const QString s1 = src_model->data(src_model->index(i1, column_)).toString();
						const QString s2 = src_model->data(src_model->index(i2, column_)).toString();
						return s1 > s2;
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
						const size_t s1 = src_model->data(src_model->index(i1, column_), Qt::UserRole).toUInt();
						const size_t s2 = src_model->data(src_model->index(i2, column_), Qt::UserRole).toUInt();
						return s1 < s2;
					});

			}
			else {
				tbb::parallel_sort(vec_idxes.begin(), vec_idxes.end(),
					[&](int i1, int i2)
					{
						const int column_ = column;
						QAbstractItemModel* const src_model = this->sourceModel();
						const size_t s1 = src_model->data(src_model->index(i1, column_), Qt::UserRole).toUInt();
						const size_t s2 = src_model->data(src_model->index(i2, column_), Qt::UserRole).toUInt();
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
