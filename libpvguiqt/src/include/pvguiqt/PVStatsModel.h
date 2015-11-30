/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2015
 */

#ifndef __PVGUIQT_PVSTATSMODEL_H__
#define __PVGUIQT_PVSTATSMODEL_H__

#include <pvguiqt/PVAbstractStatsModel.h>

#include <pvkernel/core/inendi_bench.h>

#include <pvcop/db/types.h>

namespace PVGuiQt
{

class PVStatsModel: public PVAbstractStatsModel
{
public:
	PVStatsModel(pvcop::db::array col1, pvcop::db::array col2, QWidget* parent = nullptr)
		: PVAbstractStatsModel(parent),
		 _col1(std::move(col1)),
		 _col2(std::move(col2)),
		 _sort(_col1.size()),
		 _native_sort(_sort.to_core_array()),
		 _sort_idx(-1)
	{
		std::iota(_native_sort.begin(), _native_sort.end(), 0);
	}

public:
	int rowCount(QModelIndex const& parent) const
	{
		if (parent.isValid()) {
			return 0;
		}

		return _col1.size();
	}

	QVariant data(QModelIndex const& index, int role) const
	{
		assert((size_t) index.row() < _col1.size());

		if (role == Qt::DisplayRole) {
			switch (index.column()) {
				case 0:
				{
					std::string const& str = _col1.at(sorted_index(index.row()));
					return QVariant(QString::fromUtf8(str.c_str(), str.size()));
				}
				break;
			}
		}
		else if (role == Qt::UserRole) {
			switch (index.column()) {
				case 1:
				{
					std::string const& str = _col2.at(sorted_index(index.row()));
					return QVariant(QString::fromUtf8(str.c_str(), str.size()));
				}
			}
		}

		return QVariant();
	}

	void sort(int col_idx, Qt::SortOrder order) override
	{
		assert(col_idx == 0 || col_idx == 1);

		if (_sort_idx != col_idx) {
			const pvcop::db::array& column = (col_idx == 0) ? _col1 : _col2;

			BENCH_START(sort);
			_sort.parallel_sort_on(column);
			BENCH_END(sort, "sort", column.size(), /*column.mem_size() / column.size()*/1, column.size(), /*column.mem_size() / column.size()*/1);
		}

		_sort_order = order;
		_sort_idx = col_idx;

		emit layoutChanged();
	}

private:
	inline size_t sorted_index(int row) const
	{
		// Invert sort order
		if (_sort_idx != -1 && _sort_order == Qt::SortOrder::DescendingOrder) {
			row = _sort.size() - row -1;
		}

	    return _native_sort[row];
	}

private:
	using type_index = typename pvcop::db::type_traits::type_id<pvcop::db::type_index>::type;

private:
	const pvcop::db::array _col1;
	const pvcop::db::array _col2;

	pvcop::db::indexes _sort;
	pvcop::core::array<type_index>& _native_sort;

	Qt::SortOrder _sort_order;
	int _sort_idx;
};

} // namespace PVGuiQt

#endif // __PVGUIQT_PVSTATSMODEL_H__
