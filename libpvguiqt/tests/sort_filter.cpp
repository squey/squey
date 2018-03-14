#include <pvkernel/core/inendi_assert.h>
#include <pvguiqt/PVSortFilter.h>

#include <pvcop/core/selected_array.h>

#include <random>
#include <chrono>
#include <iostream>

#ifdef INSPECTOR_BENCH
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
	Inendi::PVSelection sel(SIZE);
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

#ifndef INSPECTOR_BENCH
	PV_ASSERT_VALID(not is_sorted(sf, sorted_sel_array));
#endif

	start = std::chrono::steady_clock::now();
	sf.set_filter_as_sort();
	end = std::chrono::steady_clock::now();
	diff += (end - start);

#ifndef INSPECTOR_BENCH
	PV_ASSERT_VALID(is_sorted(sf, sorted_sel_array));
#endif

	std::cout << diff.count();
}
