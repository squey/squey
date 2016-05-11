#ifndef PVSORTFILTER_H
#define PVSORTFILTER_H

#include <Qt>
#include <vector>

#include <inendi/PVSelection.h>

#include <pvcop/db/array.h>

namespace PVGuiQt
{

class PVSortFilter
{
  public:
	PVSortFilter(size_t row_count)
	    : _filter(row_count)
	    , _sort(row_count)
	    , _sorted_column(PVCOL_INVALID_VALUE)
	    , _sort_order(Qt::SortOrder::AscendingOrder)
	{
		auto& sort = _sort.to_core_array();
		std::iota(sort.begin(), sort.end(), 0);
		std::iota(_filter.begin(), _filter.end(), 0);
	}

	PVRow row_pos_to_index(PVRow idx) const { return _filter[idx]; }

	PVRow row_pos_from_index(PVRow idx) const
	{
		return std::distance(_filter.begin(), std::find(_filter.begin(), _filter.end(), idx));
	}

	size_t size() const { return _filter.size(); }

	void set_filter(Inendi::PVSelection const& sel)
	{

		auto const& sort = _sort.to_core_array();
		_filter.resize(sort.size());

		// Push selected lines
		size_t copy_size = std::distance(
		    _filter.begin(), std::copy_if(sort.begin(), sort.end(), _filter.begin(),
		                                  [&](PVRow row) { return sel.get_line(row); }));
		_filter.resize(copy_size);

		if (_sort_order == Qt::DescendingOrder) {
			std::reverse(_filter.begin(), _filter.end());
		}
	}

	std::vector<PVRow> const& shown_lines() const { return _filter; }

	pvcop::db::indexes& sorting() { return _sort; }

	PVCol sorted_column() const { return _sorted_column; }

	void set_filter_as_sort()
	{
		auto const& sort = _sort.to_core_array();
		if (_sort_order != Qt::DescendingOrder) {
			std::copy(sort.begin(), sort.end(), _filter.begin());
		} else {
			std::copy(sort.begin(), sort.end(), _filter.rbegin());
		}
	}

	void set_sorted_meta(PVCol col, Qt::SortOrder order)
	{
		_sorted_column = col;
		_sort_order = order;
	}

  private:
	std::vector<PVRow> _filter; //!< Lines to use, map listing_row_id to nraw_row_id unsorted
	pvcop::db::indexes _sort;  //!< Sorted lines, map listing not filtered position to nraw position
	PVCol _sorted_column;      //!< The current sorted column
	Qt::SortOrder _sort_order; //!< The sort order of the current sorted column
};
}

#endif
