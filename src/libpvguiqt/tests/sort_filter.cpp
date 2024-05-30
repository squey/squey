//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/core/squey_assert.h>
#include <pvguiqt/PVSortFilter.h>

#include <pvcop/core/selected_array.h>

#include <random>
#include <chrono>
#include <iostream>

#ifdef SQUEY_BENCH
constexpr size_t SIZE = 200000000;
#else
constexpr size_t SIZE = 10000;
#endif

static bool is_sorted(const PVGuiQt::PVSortFilter& sf,
                      const pvcop::core::selected_array<pvcop::db::index_t>& sorted_sel_array)
{
	return std::is_sorted(std::begin(sorted_sel_array), std::end(sorted_sel_array),
	                      [&](pvcop::db::index_t i1, pvcop::db::index_t i2) {
		                      return sf.row_pos_from_index(i1) < sf.row_pos_from_index(i2);
		                  });
}

int main()
{
	PVGuiQt::PVSortFilter sf(SIZE);
	Squey::PVSelection sel(SIZE);
	sel.select_even();

	PV_VALID(sf.size(), SIZE);

	// Shuffle values as it is the worse but normal case)
	auto& sort = sf.sorting().to_core_array();
	const auto& sorted_sel_array = pvcop::core::make_selected_array(sort, sel);
	std::mt19937 g; // Init from seed to keep same random ordering
	std::shuffle(sort.begin(), sort.end(), g);

	auto start = std::chrono::steady_clock::now();
	sf.set_filter(sel);
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> diff(end - start);

	PV_VALID(sf.size(), SIZE / 2);

#ifndef SQUEY_BENCH
	PV_ASSERT_VALID(not is_sorted(sf, sorted_sel_array));
#endif

	start = std::chrono::steady_clock::now();
	sf.set_filter_as_sort();
	end = std::chrono::steady_clock::now();
	diff += (end - start);

#ifndef SQUEY_BENCH
	PV_ASSERT_VALID(is_sorted(sf, sorted_sel_array));
#endif

	std::cout << diff.count();
}
