/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVSORTFILTER_H
#define PVSORTFILTER_H

#include <Qt>
#include <vector>
#include <numeric>

#include <squey/PVSelection.h>

#include <pvcop/db/array.h>
#include <pvhwloc.h>

#include <omp.h>

namespace PVGuiQt
{

class PVSortFilter
{
  public:
	explicit PVSortFilter(size_t row_count) : _filter(row_count), _sort(row_count)
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

	void set_filter(Squey::PVSelection const& sel)
	{
		size_t num_threads = pvhwloc::thread_count();

		std::vector<std::vector<PVRow>> filters(num_threads);
		size_t elts[num_threads];

		auto const& sort = _sort.to_core_array();
		_filter.resize(sort.size());

#pragma omp parallel num_threads(num_threads)
		{
			auto& filter = filters[omp_get_thread_num()];
			filter.resize(sort.size() / omp_get_num_threads() + 1);
			size_t i = 0;
#pragma omp for schedule(static) nowait
			for (auto it = sort.begin(); it < sort.end(); it++) {
				if (sel.get_line_fast(*it)) {
					filter[i++] = *it;
				}
			}

			// Update it at the end to avoid false-sharing
			elts[omp_get_thread_num()] = i;
		}

		auto begin = _filter.begin();
		for (size_t i = 0; i < num_threads; i++) {
			begin = std::copy_n(filters[i].begin(), elts[i], begin);
		}

		// Push selected lines
		_filter.resize(std::distance(_filter.begin(), begin));

		if (_sort_order == Qt::DescendingOrder) {
			std::reverse(_filter.begin(), _filter.end());
		}
	}

	std::vector<PVRow> const& shown_lines() const { return _filter; }

	pvcop::db::indexes& sorting() { return _sort; }

	PVCombCol sorted_column() const { return _sorted_column; }

	void set_filter_as_sort()
	{
		auto const& sort = _sort.to_core_array();
		if (_sort_order != Qt::DescendingOrder) {
			std::copy(sort.begin(), sort.end(), _filter.begin());
		} else {
			std::copy(sort.begin(), sort.end(), _filter.rbegin());
		}
	}

	void set_sorted_meta(PVCombCol col, Qt::SortOrder order)
	{
		_sorted_column = col;
		_sort_order = order;
	}

  private:
	std::vector<PVRow> _filter; //!< Lines to use, map listing_row_id to nraw_row_id unsorted
	pvcop::db::indexes _sort;  //!< Sorted lines, map listing not filtered position to nraw position
	PVCombCol _sorted_column;  //!< The current sorted column
	Qt::SortOrder _sort_order; //!< The sort order of the current sorted column
};
} // namespace PVGuiQt

#endif
